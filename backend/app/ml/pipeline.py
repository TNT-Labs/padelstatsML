"""Orchestrator: lega tutti gli stadi ML in pipeline end-to-end.

Pattern: ogni stadio è isolato e testabile. Questo modulo li compone e gestisce
il progresso (callback per aggiornare DB).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2

from app.ml.ball import BallTracker, smooth_trajectory
from app.ml.court import calibrate_from_video
from app.ml.events import detect_events
from app.ml.players import PlayerTracker, normalize_player_ids
from app.ml.shots import classify_shot
from app.ml.stats import compute_stats, detect_rallies


@dataclass
class PipelineConfig:
    yolo_weights: str
    tracknet_weights: str | None
    device: str = "cuda"
    player_stride: int = 1


class AnalysisPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(
        self,
        video_path: str,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict:
        """Esegue analisi completa.

        Args:
            video_path: path locale al video
            progress_callback: chiamato con (percent, message) per updates

        Returns:
            dict con stats serializzate, pronto per persistenza
        """
        def report(p: int, msg: str) -> None:
            if progress_callback:
                progress_callback(p, msg)

        # 1. Court calibration (5%)
        report(2, "Calibrating court...")
        calibration = calibrate_from_video(video_path)
        if calibration is None:
            raise RuntimeError(
                "Court detection failed. Camera positioning may be incorrect — "
                "the entire court must be visible from a fixed elevated position."
            )
        report(5, "Court calibrated")

        # FPS for distance/time calculations
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # 2. Player tracking (5% -> 35%)
        report(10, "Tracking players...")
        player_tracker = PlayerTracker(self.config.yolo_weights, self.config.device)
        all_player_dets: list[list] = []
        players_by_frame: dict[int, list] = {}
        for frame_idx, dets in enumerate(player_tracker.track_video(video_path, calibration, self.config.player_stride)):
            all_player_dets.append(dets)
            for d in dets:
                players_by_frame.setdefault(d.frame_idx, []).append(d)
            if frame_idx % 200 == 0:
                # Progress proporzionale (stima 30% al completamento di player tracking)
                report(min(35, 10 + frame_idx // 100), "Tracking players...")

        id_mapping = normalize_player_ids(all_player_dets)
        report(35, f"Identified {len(id_mapping)} players")

        # 3. Ball tracking (35% -> 70%)
        report(40, "Tracking ball...")
        ball_tracker = BallTracker(self.config.tracknet_weights, self.config.device)
        ball_track = list(ball_tracker.track_video(video_path))
        ball_track = smooth_trajectory(ball_track, max_gap=5)
        report(70, "Ball trajectory complete")

        # 4. Event detection (70% -> 80%)
        report(72, "Detecting events...")
        events = detect_events(ball_track, players_by_frame)
        report(80, f"Detected {len(events)} events")

        # 5. Shot classification (80% -> 90%)
        report(82, "Classifying shots...")
        # Per MVP: passiamo None per pose_keypoints (skip pose stage)
        # In produzione: aggiungere YOLOv8-pose stage e collegare qui
        classified_shots = []
        for ev in events:
            from app.ml.events import EventType
            if ev.type != EventType.HIT or ev.player_id is None:
                continue
            # Trova posizione court del giocatore al frame del hit
            player_pos = None
            for p in players_by_frame.get(ev.frame_idx, []):
                if p.track_id == ev.player_id:
                    player_pos = p.foot_court
                    break
            cs = classify_shot(ev, None, ball_track, player_pos)
            classified_shots.append(cs)
        report(90, f"Classified {len(classified_shots)} shots")

        # 6. Stats aggregation (90% -> 100%)
        report(92, "Aggregating stats...")
        rallies = detect_rallies(events)
        result = compute_stats(
            player_detections=all_player_dets,
            classified_shots=classified_shots,
            rallies=rallies,
            id_mapping=id_mapping,
            calibration=calibration,
            fps=fps,
        )
        report(100, "Done")

        return result
