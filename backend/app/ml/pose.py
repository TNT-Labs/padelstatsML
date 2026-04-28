"""YOLOv8-Pose keypoint extraction for shot classification.

Only processes frames within a window of each HIT event, not the full video —
this keeps the extra inference pass cheap (typically <5% of total frames).

Output: {frame_idx: {track_id: np.ndarray shape (17, 3)}}
  The 17-point COCO keypoint layout (x, y, confidence per point).

Matching strategy: pose bounding boxes are matched to known player track_ids
via foot-point proximity (mid-bottom of pose bbox vs player foot_px).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.ml.players import PlayerDetection


# COCO keypoint indices used by shot classifier
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_WRIST    = 9
KP_R_WRIST    = 10


class PoseTracker:
    """Run YOLOv8-Pose on a sparse subset of frames."""

    def __init__(self, weights: str, device: str = "cpu") -> None:
        self._model = None
        self._device = device
        if Path(weights).exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(weights)
            except Exception as exc:
                print(f"[PoseTracker] Load failed: {exc} — pose stage disabled")

    @property
    def available(self) -> bool:
        return self._model is not None

    def extract_poses(
        self,
        video_path: str,
        players_by_frame: dict[int, list[PlayerDetection]],
        hit_frames: list[int],
        window: int = 4,
        foot_match_px: float = 120.0,
    ) -> dict[int, dict[int, np.ndarray]]:
        """Extract pose keypoints for frames near each HIT event.

        Args:
            video_path:       source video
            players_by_frame: {frame_idx: [PlayerDetection, …]}
            hit_frames:       frame indices of HIT events
            window:           look ±window frames around each hit
            foot_match_px:    max distance (px) for pose→player matching

        Returns:
            {frame_idx: {track_id: keypoints_17x3}}
        """
        if not self.available or not hit_frames:
            return {}

        # Collect the union of frames to process
        target: set[int] = set()
        for f in hit_frames:
            for d in range(-window, window + 1):
                target.add(f + d)

        result: dict[int, dict[int, np.ndarray]] = {}

        cap = cv2.VideoCapture(video_path)
        frame_idx = -1

        while True:
            ok, frame = cap.read()
            frame_idx += 1
            if not ok:
                break
            if frame_idx not in target:
                continue

            res = self._model(frame, verbose=False, device=self._device, classes=[0])
            if not res or res[0].keypoints is None:
                continue

            kpts_all = res[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes    = res[0].boxes
            if boxes is None or len(kpts_all) == 0:
                continue

            players_here = players_by_frame.get(frame_idx, [])
            frame_poses: dict[int, np.ndarray] = {}

            for i, kpts in enumerate(kpts_all):
                if i >= len(boxes.xyxy):
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                foot_x = (x1 + x2) / 2.0
                foot_y = y2

                # Match to the closest known player track
                best_tid: int | None = None
                best_dist = float("inf")
                for p in players_here:
                    d = np.hypot(p.foot_px[0] - foot_x, p.foot_px[1] - foot_y)
                    if d < best_dist:
                        best_dist = d
                        best_tid = p.track_id

                if best_tid is not None and best_dist < foot_match_px:
                    frame_poses[best_tid] = kpts   # shape (17, 3)

            if frame_poses:
                result[frame_idx] = frame_poses

        cap.release()
        return result
