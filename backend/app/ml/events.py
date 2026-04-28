"""Event detection da trajectory di palla + giocatori.

Eventi rilevati (rule-based per MVP):
- HIT: brusco cambio di direzione/velocità della palla in prossimità di un giocatore
- BOUNCE: cambio segno della componente verticale (Y) della velocità palla
- WALL_HIT: velocity reversal perpendicular to a court boundary when ball is near that wall

Approccio: calcola velocità frame-by-frame, identifica picchi di accelerazione
(z-score sulla magnitudo), classifica per contesto spaziale.

In produzione: TCN o Transformer su sequenze di features (ball pos + 4 player
poses) trainato su match annotati. Molto più accurato.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from app.ml.ball import BallDetection
from app.ml.players import PlayerDetection

if TYPE_CHECKING:
    from app.ml.court import CourtCalibration


class EventType(str, Enum):
    HIT = "hit"
    BOUNCE = "bounce"
    WALL_HIT = "wall_hit"


@dataclass
class Event:
    type: EventType
    frame_idx: int
    ball_pos_px: tuple[float, float]
    player_id: int | None  # only for HIT
    confidence: float


def detect_events(
    ball_track: list[BallDetection],
    players_by_frame: dict[int, list[PlayerDetection]],
    proximity_threshold_px: float = 80.0,
    min_velocity_change: float = 30.0,
    calibration: "CourtCalibration | None" = None,
    wall_margin_m: float = 0.8,
) -> list[Event]:
    """Pipeline event detection.

    Args:
        ball_track: trajectory della palla (post smoothing)
        players_by_frame: {frame_idx: [PlayerDetection, ...]}
        proximity_threshold_px: distanza max ball-player per HIT
        min_velocity_change: pixel/frame, soglia per detection cambio direzione
        calibration: court homography; required for WALL_HIT detection
        wall_margin_m: distance from court edge (meters) to classify as near-wall
    """
    valid = [(b.frame_idx, b.pos_px) for b in ball_track if b.pos_px is not None]
    if len(valid) < 5:
        return []

    frames = np.array([f for f, _ in valid])
    positions = np.array([p for _, p in valid])

    velocities = np.diff(positions, axis=0)
    accels = np.linalg.norm(np.diff(velocities, axis=0), axis=1)

    events: list[Event] = []

    # 1. HIT: acceleration peak + nearest player within threshold
    for i, acc in enumerate(accels):
        if acc < min_velocity_change:
            continue

        peak_frame = int(frames[i + 1])
        peak_pos = positions[i + 1]

        nearest_player, nearest_dist = _find_nearest_player(
            peak_pos, players_by_frame.get(peak_frame, [])
        )

        if nearest_player is not None and nearest_dist < proximity_threshold_px:
            conf = min(1.0, proximity_threshold_px / max(nearest_dist, 1.0)) * 0.5 + 0.3
            events.append(
                Event(
                    type=EventType.HIT,
                    frame_idx=peak_frame,
                    ball_pos_px=(float(peak_pos[0]), float(peak_pos[1])),
                    player_id=nearest_player,
                    confidence=float(conf),
                )
            )

    # 2. BOUNCE: Y-velocity sign change (image coords: +Y = downward)
    for i in range(1, len(velocities)):
        if velocities[i - 1, 1] > 5 and velocities[i, 1] < -5:
            events.append(
                Event(
                    type=EventType.BOUNCE,
                    frame_idx=int(frames[i]),
                    ball_pos_px=(float(positions[i, 0]), float(positions[i, 1])),
                    player_id=None,
                    confidence=0.6,
                )
            )

    # 3. WALL_HIT: velocity reversal perpendicular to nearest court boundary
    if calibration is not None:
        wall_events = _detect_wall_hits(
            frames, positions, velocities, calibration, wall_margin_m
        )
        events.extend(wall_events)

    events = _dedup_events(events, min_gap_frames=5)
    events.sort(key=lambda e: e.frame_idx)
    return events


def _detect_wall_hits(
    frames: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    calibration: "CourtCalibration",
    wall_margin_m: float,
) -> list[Event]:
    """Detect ball-wall contacts using court-coordinate velocity reversal.

    A wall hit is identified when:
    - The ball is within `wall_margin_m` of a court boundary in court coords
    - The velocity component perpendicular to that boundary reverses sign
      between consecutive frames

    We check indices i in [1, len(velocities)-1] which correspond to position
    index i+1 — the frame where the reversal occurs.
    """
    from app.ml.court import COURT_WIDTH_M, COURT_LENGTH_M

    court_pts = calibration.pixel_to_court(positions)  # Nx2 in metres

    wall_events: list[Event] = []
    n = len(velocities)

    for i in range(1, n):
        pos_idx = i  # positions[i] is between velocities[i-1] and velocities[i]
        cx, cy = court_pts[pos_idx]

        near_left = cx < wall_margin_m
        near_right = cx > COURT_WIDTH_M - wall_margin_m
        near_bottom = cy < wall_margin_m
        near_top = cy > COURT_LENGTH_M - wall_margin_m

        if not (near_left or near_right or near_bottom or near_top):
            continue

        # Velocity in court coords (approximate using pixel velocities scaled)
        # We use pixel velocities directly since the sign relationship is preserved
        # (assuming camera is roughly axis-aligned with the court)
        vx_prev, vy_prev = velocities[i - 1]
        vx_curr, vy_curr = velocities[i]

        x_reversal = (
            (near_left and vx_prev < -3 and vx_curr > 3) or
            (near_right and vx_prev > 3 and vx_curr < -3)
        )
        y_reversal = (
            (near_bottom and vy_prev < -3 and vy_curr > 3) or
            (near_top and vy_prev > 3 and vy_curr < -3)
        )

        if not (x_reversal or y_reversal):
            continue

        wall_events.append(
            Event(
                type=EventType.WALL_HIT,
                frame_idx=int(frames[pos_idx]),
                ball_pos_px=(float(positions[pos_idx, 0]), float(positions[pos_idx, 1])),
                player_id=None,
                confidence=0.65,
            )
        )

    return wall_events


def _find_nearest_player(
    ball_pos: np.ndarray, players: list[PlayerDetection]
) -> tuple[int | None, float]:
    if not players:
        return None, float("inf")
    distances = [
        (p.track_id, float(np.linalg.norm(np.array(p.foot_px) - ball_pos))) for p in players
    ]
    nearest = min(distances, key=lambda x: x[1])
    return nearest


def _dedup_events(events: list[Event], min_gap_frames: int) -> list[Event]:
    """Remove duplicate events of the same type within min_gap_frames."""
    if not events:
        return events
    by_type: dict[EventType, list[Event]] = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)

    out: list[Event] = []
    for evs in by_type.values():
        evs.sort(key=lambda e: e.frame_idx)
        last_kept_frame = -(10 ** 9)
        for e in evs:
            if e.frame_idx - last_kept_frame >= min_gap_frames:
                out.append(e)
                last_kept_frame = e.frame_idx
    return out
