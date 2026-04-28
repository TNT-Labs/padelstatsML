"""Shot type classification: smash, volée, bandeja, altro.

Strategia MVP (rule-based su pose + ball context):
- SMASH: braccio sopra la testa (wrist Y < shoulder Y - threshold) + palla alta
        prima del colpo + alta velocità ball post-hit
- VOLEE: palla colpita PRIMA del rimbalzo + giocatore vicino alla rete (Y court < 7m
         o Y court > 13m a seconda del lato) + braccio frontale
- BANDEJA: posizione tra smash e volée (braccio alto ma non sopra la testa) +
           giocatore in fondo campo + traiettoria post-hit più piatta
- ALTRO: drive, bandeja-rovescio, vibora, etc. (non classificati nel MVP)

In produzione: classificatore ML (XGBoost o piccolo MLP) su feature engineerate
da pose temporal window (es. 1 secondo prima/dopo hit) + ball features.
Dataset: sequenze annotate di colpi, anche pochi mila esempi sufficienti.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.ml.ball import BallDetection
from app.ml.events import Event


class ShotType(str, Enum):
    SMASH = "smash"
    VOLLEY = "volley"
    BANDEJA = "bandeja"
    OTHER = "other"


@dataclass
class ClassifiedShot:
    event: Event
    shot_type: ShotType
    confidence: float


def classify_shot(
    hit_event: Event,
    pose_keypoints: dict[int, np.ndarray] | None,  # {frame_idx: 17x3 keypoints (x, y, conf)}
    ball_track: list[BallDetection],
    player_court_pos: tuple[float, float] | None,
) -> ClassifiedShot:
    """Classifica un singolo evento HIT.

    Args:
        hit_event: l'evento HIT
        pose_keypoints: keypoints YOLOv8-pose per il giocatore al frame del hit
        ball_track: trajectory della palla (per analizzare velocità pre/post hit)
        player_court_pos: posizione giocatore in court coords (m)
    """
    if hit_event.player_id is None:
        return ClassifiedShot(hit_event, ShotType.OTHER, 0.0)

    # Estrai contesto: ball Y prima del hit (per detect altezza palla)
    ball_y_before = _ball_y_window(ball_track, hit_event.frame_idx, before=10)

    # Pose features se disponibili
    arm_above_head = False
    arm_high = False
    if pose_keypoints is not None and hit_event.frame_idx in pose_keypoints:
        kpts = pose_keypoints[hit_event.frame_idx]
        # COCO keypoints: 5=left_shoulder, 6=right_shoulder, 9=left_wrist, 10=right_wrist
        # Y axis: smaller = higher in image
        try:
            r_shoulder_y = kpts[6, 1]
            r_wrist_y = kpts[10, 1]
            l_shoulder_y = kpts[5, 1]
            l_wrist_y = kpts[9, 1]

            min_wrist_y = min(r_wrist_y, l_wrist_y)
            min_shoulder_y = min(r_shoulder_y, l_shoulder_y)

            arm_above_head = min_wrist_y < min_shoulder_y - 30
            arm_high = min_wrist_y < min_shoulder_y + 10
        except (IndexError, ValueError):
            pass

    # Court position: Y court = dove sta il giocatore lungo il campo (0..20m)
    is_at_net = False
    is_at_back = False
    if player_court_pos is not None:
        y_court = player_court_pos[1]
        # Net is at 10m (mid). Player vicino rete: 7..10 oppure 10..13
        is_at_net = 7.0 < y_court < 13.0
        is_at_back = y_court < 4.0 or y_court > 16.0

    ball_was_high = ball_y_before < hit_event.ball_pos_px[1] - 50  # palla scendeva → era più alta

    # Decision tree
    if arm_above_head and ball_was_high and is_at_back:
        return ClassifiedShot(hit_event, ShotType.SMASH, 0.75)
    if arm_above_head and ball_was_high and is_at_net:
        return ClassifiedShot(hit_event, ShotType.SMASH, 0.65)
    if arm_high and is_at_back and ball_was_high:
        return ClassifiedShot(hit_event, ShotType.BANDEJA, 0.6)
    if is_at_net and not arm_above_head:
        return ClassifiedShot(hit_event, ShotType.VOLLEY, 0.55)

    return ClassifiedShot(hit_event, ShotType.OTHER, 0.4)


def _ball_y_window(ball_track: list[BallDetection], frame_idx: int, before: int) -> float:
    """Mediana Y della palla nei `before` frame precedenti."""
    relevant = [
        b.pos_px[1]
        for b in ball_track
        if b.pos_px is not None and frame_idx - before <= b.frame_idx < frame_idx
    ]
    if not relevant:
        return 0.0
    return float(np.median(relevant))
