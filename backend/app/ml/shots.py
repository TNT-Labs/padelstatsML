"""Shot type classification: smash, volée, bandeja, altro.

Classification hierarchy (most → least specific):
  SMASH   — arm above head (pose) OR ball very high + high post-hit speed + back/net position
  BANDEJA — arm high (pose) OR high ball + mid-court + moderate post-hit speed
  VOLLEY  — near net + ball NOT high (set before bounce) + low-medium speed
  OTHER   — drive, lob, globo, vibora

Pose keypoints (COCO 17-point format) improve accuracy significantly when
available from the YOLOv8-Pose stage.  When pose=None the classifier falls
back to ball-trajectory + court-position features only.

In production: replace decision tree with XGBoost/MLP trained on a few
thousand annotated shots with temporal features.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from app.ml.ball import BallDetection
from app.ml.events import Event

if TYPE_CHECKING:
    from app.ml.court import CourtCalibration


class ShotType(str, Enum):
    SMASH   = "smash"
    VOLLEY  = "volley"
    BANDEJA = "bandeja"
    OTHER   = "other"


@dataclass
class ClassifiedShot:
    event: Event
    shot_type: ShotType
    confidence: float


# ── Court-position constants ─────────────────────────────────────────────────
# Net is at 10 m.  A player is "at net" if within 3 m of it (7–13 m range).
_NET_ZONE_LO = 7.0
_NET_ZONE_HI = 13.0
_BACK_THRESHOLD = 4.5   # < 4.5 m or > 15.5 m from baselines = back court


def classify_shot(
    hit_event: Event,
    pose_keypoints: np.ndarray | None,        # 17×3 (x, y, conf) COCO, or None
    ball_track: list[BallDetection],
    player_court_pos: tuple[float, float] | None,
    fps: float = 30.0,
    calibration: "CourtCalibration | None" = None,
) -> ClassifiedShot:
    """Classify a single HIT event into a shot type.

    Args:
        hit_event:        the HIT Event to classify
        pose_keypoints:   COCO 17-point keypoints for the hitting player at
                          the hit frame (x, y, conf), or None
        ball_track:       full ball trajectory for context
        player_court_pos: (x_m, y_m) in court coordinates, or None
    """
    if hit_event.player_id is None:
        return ClassifiedShot(hit_event, ShotType.OTHER, 0.0)

    # ── Features from ball trajectory ────────────────────────────────────────
    ball_y_before  = _ball_y_window(ball_track, hit_event.frame_idx, before=12)
    hit_y          = hit_event.ball_pos_px[1]
    # In image coords: smaller Y = higher up.  ball_was_high → ball was above hit point.
    ball_was_high  = (ball_y_before < hit_y - 30) if ball_y_before else False

    post_speed_px = _ball_speed_after(ball_track, hit_event.frame_idx, after=6)
    if calibration is not None and fps > 0:
        # Convert to m/s: resolution- and fps-independent thresholds
        post_speed_ms = _ball_speed_after_ms(
            ball_track, hit_event.frame_idx, after=6, calibration=calibration, fps=fps
        )
        high_speed   = post_speed_ms > 18.0   # m/s — padel smash
        medium_speed = post_speed_ms > 6.0
    else:
        high_speed   = post_speed_px > 20.0
        medium_speed = post_speed_px > 8.0

    # ── Features from player court position ──────────────────────────────────
    is_at_net  = False
    is_at_back = False
    if player_court_pos is not None:
        y = player_court_pos[1]
        is_at_net  = _NET_ZONE_LO < y < _NET_ZONE_HI
        is_at_back = y < _BACK_THRESHOLD or y > (20.0 - _BACK_THRESHOLD)

    # ── Features from pose keypoints (COCO) ──────────────────────────────────
    # Only trust keypoints whose detection confidence exceeds the threshold.
    # Using low-confidence keypoints produces more misclassifications than
    # ignoring them entirely.
    arm_above_head = False
    arm_high       = False
    _KPT_CONF = 0.35
    if pose_keypoints is not None:
        try:
            # COCO indices: 5=L-shoulder, 6=R-shoulder, 9=L-wrist, 10=R-wrist
            ls_y, ls_c = pose_keypoints[5, 1], pose_keypoints[5, 2]
            rs_y, rs_c = pose_keypoints[6, 1], pose_keypoints[6, 2]
            lw_y, lw_c = pose_keypoints[9, 1], pose_keypoints[9, 2]
            rw_y, rw_c = pose_keypoints[10, 1], pose_keypoints[10, 2]
            if max(lw_c, rw_c) > _KPT_CONF:
                # Use the more confident wrist; ignore shoulder if uncertain
                wrist_y    = lw_y if lw_c >= rw_c else rw_y
                shoulder_y = (
                    min(ls_y, rs_y)
                    if min(ls_c, rs_c) > _KPT_CONF
                    else wrist_y + 50   # conservative: assume shoulder is below
                )
                arm_above_head = wrist_y < shoulder_y - 25
                arm_high       = wrist_y < shoulder_y + 15
        except (IndexError, ValueError):
            pass

    # ── Decision tree (pose-aware when available) ────────────────────────────
    if arm_above_head and ball_was_high:
        return ClassifiedShot(hit_event, ShotType.SMASH, 0.82)

    if ball_was_high and high_speed and (is_at_back or is_at_net):
        return ClassifiedShot(hit_event, ShotType.SMASH, 0.62)

    if arm_high and is_at_back and ball_was_high:
        return ClassifiedShot(hit_event, ShotType.BANDEJA, 0.65)

    if ball_was_high and medium_speed and is_at_back:
        return ClassifiedShot(hit_event, ShotType.BANDEJA, 0.50)

    if is_at_net and not arm_above_head and not ball_was_high:
        return ClassifiedShot(hit_event, ShotType.VOLLEY, 0.60)

    if is_at_net and medium_speed:
        return ClassifiedShot(hit_event, ShotType.VOLLEY, 0.45)

    return ClassifiedShot(hit_event, ShotType.OTHER, 0.40)


# ── Ball trajectory helpers ──────────────────────────────────────────────────

def _ball_y_window(ball_track: list[BallDetection], frame_idx: int, before: int) -> float | None:
    """Median Y position of ball in the `before` frames preceding hit."""
    ys = [
        b.pos_px[1]
        for b in ball_track
        if b.pos_px is not None and frame_idx - before <= b.frame_idx < frame_idx
    ]
    return float(np.median(ys)) if ys else None


def _ball_speed_after(ball_track: list[BallDetection], frame_idx: int, after: int) -> float:
    """Mean ball speed (px/frame) for `after` frames following the hit."""
    pts = [
        b.pos_px
        for b in ball_track
        if b.pos_px is not None and frame_idx <= b.frame_idx <= frame_idx + after
    ]
    if len(pts) < 2:
        return 0.0
    speeds = [
        np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        for i in range(1, len(pts))
    ]
    return float(np.mean(speeds))


def _ball_speed_after_ms(
    ball_track: list[BallDetection],
    frame_idx: int,
    after: int,
    calibration: "CourtCalibration",
    fps: float,
) -> float:
    """Mean ball speed in m/s for `after` frames following the hit.

    Converts pixel positions to court metres via the homography, then
    multiplies by fps — gives a resolution- and zoom-independent value.
    """
    pts_px = np.array(
        [b.pos_px for b in ball_track
         if b.pos_px is not None and frame_idx <= b.frame_idx <= frame_idx + after],
        dtype=np.float32,
    )
    if len(pts_px) < 2:
        return 0.0
    pts_m = calibration.pixel_to_court(pts_px)
    dists = np.linalg.norm(np.diff(pts_m, axis=0), axis=1)
    return float(dists.mean() * fps)
