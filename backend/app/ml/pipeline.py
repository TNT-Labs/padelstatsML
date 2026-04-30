"""Orchestrator: chains all ML stages into a single end-to-end pipeline.

Stage order and progress budget:
  1. Court calibration    2%  →  5%
  2. Player tracking      5%  → 35%
  3. Ball tracking       35%  → 65%
  4. Pose extraction     65%  → 72%   (skipped if weights absent)
  5. Event detection     72%  → 80%
  6. Shot classification 80%  → 90%
  7. Stats aggregation   90%  → 100%

Each stage is isolated and can be tested individually.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from app.ml.ball import BallTracker, smooth_trajectory
from app.ml.court import calibrate_from_video
from app.ml.crops import extract_player_crops
from app.ml.events import detect_events, EventType
from app.ml.players import PlayerTracker, normalize_player_ids
from app.ml.pose import PoseTracker
from app.ml.shots import classify_shot
from app.ml.stats import compute_stats, detect_rallies


@dataclass
class PipelineConfig:
    yolo_weights:      str
    tracknet_weights:  str | None
    device:            str  = "cpu"
    player_stride:     int  = 2
    ball_stride:       int  = 2
    # Optional: path to YOLOv8-pose weights for arm/wrist keypoints.
    # When the file is absent the pose stage is skipped gracefully.
    pose_weights:      str  = "yolov8n-pose.pt"


class AnalysisPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(
        self,
        video_path: str,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict:
        """Run full analysis pipeline.

        Args:
            video_path:        local path to video file
            progress_callback: called with (percent 0-100, message)

        Returns:
            Serialisable dict ready for DB persistence (see MatchStats model).
        """
        def report(p: int, msg: str) -> None:
            if progress_callback:
                progress_callback(p, msg)

        # ── 1. Court calibration ─────────────────────────────────────────────
        report(2, "Calibrating court…")
        calibration = calibrate_from_video(video_path)
        if calibration is None:
            raise RuntimeError(
                "Court calibration failed — ensure the full court is visible "
                "from a fixed, elevated camera position."
            )
        if calibration.estimated:
            print("[Pipeline] Warning: using estimated court corners — "
                  "accuracy will be reduced.")
        report(5, "Court calibrated" + (" (estimated)" if calibration.estimated else ""))

        # FPS for downstream calculations
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # ── 2. Player tracking ───────────────────────────────────────────────
        report(8, "Tracking players…")
        player_tracker = PlayerTracker(self.config.yolo_weights, self.config.device)
        all_player_dets: list[list] = []
        players_by_frame: dict[int, list] = {}

        for enum_idx, dets in enumerate(
            player_tracker.track_video(video_path, calibration, self.config.player_stride)
        ):
            all_player_dets.append(dets)
            for d in dets:
                players_by_frame.setdefault(d.frame_idx, []).append(d)
            if enum_idx % 150 == 0:
                progress = min(35, 8 + enum_idx // 50)
                report(progress, "Tracking players…")

        id_mapping = normalize_player_ids(all_player_dets)
        report(35, f"Identified {len(id_mapping)} player track(s)")

        # ── 3. Ball tracking ─────────────────────────────────────────────────
        report(38, "Tracking ball…")
        ball_tracker = BallTracker(self.config.tracknet_weights, self.config.device)
        if not ball_tracker.using_neural_model:
            report(38, "Tracking ball (MOG2 fallback — no TrackNet weights)…")
        ball_track = list(ball_tracker.track_video(video_path, stride=self.config.ball_stride))
        ball_track = smooth_trajectory(ball_track, max_gap=self.config.ball_stride * 3)
        report(65, f"Ball track: {sum(1 for b in ball_track if b.pos_px)} detections")

        # ── 4. Pose extraction (optional) ────────────────────────────────────
        report(66, "Extracting player poses…")
        pose_tracker = PoseTracker(self.config.pose_weights, self.config.device)
        pose_by_frame: dict[int, dict[int, object]] = {}
        if pose_tracker.available:
            # Only run on frames that will be used for shot classification
            hit_frame_candidates = [
                b.frame_idx for b in ball_track if b.pos_px is not None
            ]
            pose_by_frame = pose_tracker.extract_poses(
                video_path, players_by_frame, hit_frame_candidates[:500]
            )
            report(72, f"Pose extracted for {len(pose_by_frame)} frames")
        else:
            report(72, "Pose stage skipped (no weights)")

        # ── 5. Event detection ───────────────────────────────────────────────
        report(73, "Detecting events…")
        events = detect_events(ball_track, players_by_frame, calibration=calibration, fps=fps)
        report(80, f"Detected {len(events)} events")

        # ── 6. Shot classification ───────────────────────────────────────────
        report(82, "Classifying shots…")
        classified_shots = []
        for ev in events:
            if ev.type != EventType.HIT or ev.player_id is None:
                continue

            # Court position of hitting player
            player_court_pos = None
            for p in players_by_frame.get(ev.frame_idx, []):
                if p.track_id == ev.player_id:
                    player_court_pos = p.foot_court
                    break

            # Pose keypoints at hit frame (None if pose stage was skipped)
            pose_kpts = None
            frame_poses = pose_by_frame.get(ev.frame_idx, {})
            if ev.player_id in frame_poses:
                pose_kpts = frame_poses[ev.player_id]

            cs = classify_shot(ev, pose_kpts, ball_track, player_court_pos,
                               fps=fps, calibration=calibration)
            classified_shots.append(cs)
        report(90, f"Classified {len(classified_shots)} shots")

        # ── 7. Stats aggregation ─────────────────────────────────────────────
        report(92, "Aggregating stats…")
        rallies = detect_rallies(events)
        result  = compute_stats(
            player_detections=all_player_dets,
            classified_shots=classified_shots,
            rallies=rallies,
            id_mapping=id_mapping,
            calibration=calibration,
            fps=fps,
            ball_track=ball_track,
        )

        # ── 8. Player crop extraction ────────────────────────────────────────
        report(98, "Extracting player thumbnails…")
        result["player_crops_data"] = extract_player_crops(
            video_path, all_player_dets, id_mapping
        )

        report(100, "Done")
        return result
