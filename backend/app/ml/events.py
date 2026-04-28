"""Event detection da trajectory di palla + giocatori.

Eventi rilevati (rule-based per MVP):
- HIT: brusco cambio di direzione/velocità della palla in prossimità di un giocatore
- BOUNCE: cambio segno della componente verticale (Y) della velocità palla
- WALL_HIT: palla raggiunge i muri laterali/fondo (proximity al bordo court)

Approccio: calcola velocità frame-by-frame, identifica picchi di accelerazione
(z-score sulla magnitudo), classifica per contesto spaziale.

In produzione: TCN o Transformer su sequenze di features (ball pos + 4 player
poses) trainato su match annotati. Molto più accurato.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.ml.ball import BallDetection
from app.ml.players import PlayerDetection


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
) -> list[Event]:
    """Pipeline event detection.

    Args:
        ball_track: trajectory della palla (post smoothing)
        players_by_frame: {frame_idx: [PlayerDetection, ...]}
        proximity_threshold_px: distanza max ball-player per HIT
        min_velocity_change: pixel/frame, soglia per detection cambio direzione
    """
    # Estrai posizioni come array numpy (skip frames senza ball)
    valid = [(b.frame_idx, b.pos_px) for b in ball_track if b.pos_px is not None]
    if len(valid) < 5:
        return []

    frames = np.array([f for f, _ in valid])
    positions = np.array([p for _, p in valid])

    # Velocità (differenze finite)
    velocities = np.diff(positions, axis=0)

    # Acceleration magnitude (cambio di velocità)
    accels = np.linalg.norm(np.diff(velocities, axis=0), axis=1)

    events: list[Event] = []

    # 1. HIT detection: peak di accelerazione + giocatore vicino
    for i, acc in enumerate(accels):
        if acc < min_velocity_change:
            continue

        peak_frame = int(frames[i + 1])
        peak_pos = positions[i + 1]

        # Trova giocatore più vicino in quel frame
        nearest_player, nearest_dist = _find_nearest_player(
            peak_pos, players_by_frame.get(peak_frame, [])
        )

        if nearest_player is not None and nearest_dist < proximity_threshold_px:
            # Confidence proporzionale a inverso distanza
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

    # 2. BOUNCE detection: cambio segno componente Y velocità
    for i in range(1, len(velocities)):
        # Y crescente = palla scende (image coords). Bounce = passaggio da +Y a -Y
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

    # Dedup eventi troppo vicini (stesso evento detectato 2 volte)
    events = _dedup_events(events, min_gap_frames=5)
    events.sort(key=lambda e: e.frame_idx)
    return events


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
    """Rimuove eventi dello stesso tipo entro min_gap_frames frames."""
    if not events:
        return events
    by_type: dict[EventType, list[Event]] = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)

    out: list[Event] = []
    for evs in by_type.values():
        evs.sort(key=lambda e: e.frame_idx)
        last_kept_frame = -10**9
        for e in evs:
            if e.frame_idx - last_kept_frame >= min_gap_frames:
                out.append(e)
                last_kept_frame = e.frame_idx
    return out
