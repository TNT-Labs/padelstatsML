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

import numpy as np

from app.ml.ball import BallDetection
from app.ml.events import Event


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
    pose_keypoints: np.ndarray | None,   # 17×3 (x, y, conf) COCO, or None
    ball_track: list[BallDetection],
    player_court_pos: tuple[float, float] | None,
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

    post_speed     = _ball_speed_after(ball_track, hit_event.frame_idx, after=6)
    high_speed     = post_speed > 20.0   # px/frame — smash threshold
    medium_speed   = post_speed > 8.0

    # ── Features from player court position ──────────────────────────────────
    is_at_net  = False
    is_at_back = False
    if player_court_pos is not None:
        y = player_court_pos[1]
        is_at_net  = _NET_ZONE_LO < y < _NET_ZONE_HI
        is_at_back = y < _BACK_THRESHOLD or y > (20.0 - _BACK_THRESHOLD)

    # ── Features from pose keypoints (COCO) ──────────────────────────────────
    arm_above_head = False
    arm_high       = False
    if pose_keypoints is not None:
        try:
            # COCO indices: 5=L-shoulder, 6=R-shoulder, 9=L-wrist, 10=R-wrist
            ls_y = pose_keypoints[5, 1]
            rs_y = pose_keypoints[6, 1]
            lw_y = pose_keypoints[9, 1]
            rw_y = pose_keypoints[10, 1]
            min_wrist    = min(lw_y, rw_y)
            min_shoulder = min(ls_y, rs_y)
            arm_above_head = min_wrist < min_shoulder - 25   # wrist clearly above shoulder
            arm_high       = min_wrist < min_shoulder + 15   # wrist near or above shoulder
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
