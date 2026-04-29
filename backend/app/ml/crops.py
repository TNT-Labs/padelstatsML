"""Extract one representative crop image per detected player.

Picks the highest-confidence detection from the middle third of the video
for each canonical player ID, crops with padding, and returns JPEG bytes.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.ml.players import PlayerDetection

_CROP_W = 160
_CROP_H = 240
_PAD    = 24   # pixel padding around the bbox


def extract_player_crops(
    video_path: str,
    all_player_dets: list[list[PlayerDetection]],
    id_mapping: dict[int, int],
) -> dict[int, bytes]:
    """Return {canonical_player_id: jpeg_bytes} for each of the 4 players.

    Selects the detection with the highest confidence from the middle third
    of the video so we avoid the warm-up frames at the start.
    """
    total_frames = len(all_player_dets)
    mid_start = total_frames // 3
    mid_end   = total_frames * 2 // 3

    # best detection per canonical player: (confidence, frame_idx, bbox_px)
    best: dict[int, tuple[float, int, tuple[float, float, float, float]]] = {}

    for chunk_idx, frame_dets in enumerate(all_player_dets):
        in_middle = mid_start <= chunk_idx < mid_end
        for det in frame_dets:
            pid = id_mapping.get(det.track_id)
            if pid is None:
                continue
            # Prefer middle-third frames; give them a 2× confidence bonus
            score = det.confidence * (2.0 if in_middle else 1.0)
            if pid not in best or score > best[pid][0]:
                best[pid] = (score, det.frame_idx, det.bbox_px)

    if not best:
        return {}

    # Group required frame indices → read each frame once
    frame_to_players: dict[int, list[int]] = {}
    for pid, (_, frame_idx, _) in best.items():
        frame_to_players.setdefault(frame_idx, []).append(pid)

    crops: dict[int, bytes] = {}
    cap = cv2.VideoCapture(video_path)
    for frame_idx in sorted(frame_to_players):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        for pid in frame_to_players[frame_idx]:
            _, _, bbox = best[pid]
            x1, y1, x2, y2 = bbox
            x1 = max(0, int(x1) - _PAD)
            y1 = max(0, int(y1) - _PAD)
            x2 = min(w, int(x2) + _PAD)
            y2 = min(h, int(y2) + _PAD)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (_CROP_W, _CROP_H), interpolation=cv2.INTER_AREA)
            ok2, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok2:
                crops[pid] = buf.tobytes()
    cap.release()
    return crops
