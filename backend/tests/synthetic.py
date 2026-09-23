"""Helpers that build synthetic matches, so pipeline stages can be tested
without a real video or a detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.detect import Detection
from app.ml.tracking import Observation, _torso_histogram

FRAME_W, FRAME_H = 1920, 1080

# Four visually distinct shirts, so colour re-identification has something to
# work with — as it does in a real match, where the two pairs wear different kit.
SHIRT_BGR = [(40, 40, 220), (220, 60, 40), (40, 200, 220), (200, 40, 200)]


@dataclass
class SyntheticPlayer:
    court_xy: tuple[float, float]
    shirt: int


def render_frame(calibration, players: list[SyntheticPlayer]) -> tuple[np.ndarray, list[Detection]]:
    """Paint players as coloured rectangles and return matching detections."""
    frame = np.full((FRAME_H, FRAME_W, 3), 70, dtype=np.uint8)
    detections: list[Detection] = []

    for player in players:
        foot = calibration.court_to_pixels(np.array([player.court_xy], dtype=np.float32))[0]
        fx, fy = float(foot[0]), float(foot[1])
        # Apparent height shrinks with distance from the camera, which for a
        # camera behind the baseline means with image Y.
        height = 90.0 + 130.0 * (fy / FRAME_H)
        width = height * 0.38
        x1, y1 = fx - width / 2.0, fy - height
        x2, y2 = fx + width / 2.0, fy

        ix1, iy1 = int(np.clip(x1, 0, FRAME_W - 1)), int(np.clip(y1, 0, FRAME_H - 1))
        ix2, iy2 = int(np.clip(x2, 1, FRAME_W)), int(np.clip(y2, 1, FRAME_H))
        frame[iy1:iy2, ix1:ix2] = SHIRT_BGR[player.shirt % len(SHIRT_BGR)]

        detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=0.85))

    return frame, detections


def make_observation(
    calibration,
    court_xy: tuple[float, float],
    timestamp_s: float,
    shirt: int,
    frame_index: int | None = None,
    confidence: float = 0.85,
) -> Observation:
    frame, detections = render_frame(calibration, [SyntheticPlayer(court_xy, shirt)])
    detection = detections[0]
    return Observation(
        frame_index=frame_index if frame_index is not None else int(round(timestamp_s * 5)),
        timestamp_s=timestamp_s,
        bbox=detection.bbox,
        foot_px=detection.foot_px,
        foot_court=court_xy,
        confidence=confidence,
        color=_torso_histogram(frame, detection.bbox),
    )


def walk(
    calibration,
    shirt: int,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    t0: float,
    t1: float,
    hz: float = 5.0,
) -> list[Observation]:
    """A player moving in a straight line between two court positions."""
    steps = max(int(round((t1 - t0) * hz)), 1)
    out: list[Observation] = []
    for i in range(steps + 1):
        f = i / steps
        t = t0 + (t1 - t0) * f
        xy = (
            start_xy[0] + (end_xy[0] - start_xy[0]) * f,
            start_xy[1] + (end_xy[1] - start_xy[1]) * f,
        )
        out.append(make_observation(calibration, xy, t, shirt))
    return out
