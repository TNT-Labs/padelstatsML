"""Event detection from ball trajectory + player positions.

Events detected (rule-based for MVP):
  HIT      — sharp acceleration peak in ball trajectory near a player
  BOUNCE   — Y-velocity sign change (ball changes vertical direction)
  WALL_HIT — velocity reversal perpendicular to a court boundary

Critical fix: player tracking runs with vid_stride > 1, so
`players_by_frame` only contains entries for every Nth frame (0, 2, 4, …
with stride=2).  Ball tracking runs on every frame.  Without
`get_players_near_frame`, most HIT events would have no player candidate
because the ball's peak frame falls *between* two player frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from app.ml.ball import BallDetection
from app.ml.players import PlayerDetection, get_players_near_frame

if TYPE_CHECKING:
    from app.ml.court import CourtCalibration


class EventType(str, Enum):
    HIT      = "hit"
    BOUNCE   = "bounce"
    WALL_HIT = "wall_hit"


@dataclass
class Event:
    type: EventType
    frame_idx: int
    ball_pos_px: tuple[float, float]
    player_id: int | None  # only meaningful for HIT
    confidence: float


# ── Main entry point ─────────────────────────────────────────────────────────

def detect_events(
    ball_track: list[BallDetection],
    players_by_frame: dict[int, list[PlayerDetection]],
    proximity_threshold_px: float = 100.0,
    min_velocity_change: float = 25.0,
    calibration: "CourtCalibration | None" = None,
    wall_margin_m: float = 0.8,
) -> list[Event]:
    """Detect HIT, BOUNCE, and WALL_HIT events from ball + player trajectories.

    Args:
        ball_track: smoothed ball detections (output of smooth_trajectory)
        players_by_frame: {frame_idx: [PlayerDetection, …]}
        proximity_threshold_px: max ball-player pixel distance for a HIT
        min_velocity_change: minimum acceleration magnitude (px/frame²)
        calibration: court homography; enables WALL_HIT detection
        wall_margin_m: metres from court edge that counts as "near wall"
    """
    valid = [(b.frame_idx, b.pos_px) for b in ball_track if b.pos_px is not None]
    if len(valid) < 5:
        return []

    frames    = np.array([f for f, _ in valid])
    positions = np.array([p for _, p in valid])
    velocities = np.diff(positions, axis=0)
    accels     = np.linalg.norm(np.diff(velocities, axis=0), axis=1)

    events: list[Event] = []

    # ── 1. HIT: acceleration peak + nearby player ────────────────────────────
    for i, acc in enumerate(accels):
        if acc < min_velocity_change:
            continue

        peak_frame = int(frames[i + 1])
        peak_pos   = positions[i + 1]

        # Use nearest-frame lookup to handle stride gaps
        nearby_players = get_players_near_frame(players_by_frame, peak_frame)
        nearest_pid, nearest_dist = _find_nearest_player(peak_pos, nearby_players)

        if nearest_pid is not None and nearest_dist < proximity_threshold_px:
            conf = min(1.0, proximity_threshold_px / max(nearest_dist, 1.0)) * 0.5 + 0.3
            events.append(Event(
                type=EventType.HIT,
                frame_idx=peak_frame,
                ball_pos_px=(float(peak_pos[0]), float(peak_pos[1])),
                player_id=nearest_pid,
                confidence=float(conf),
            ))

    # ── 2. BOUNCE: Y-velocity sign change ────────────────────────────────────
    # In image coords +Y is downward. Bounce: ball going down (vy > 0) then up (vy < 0)
    for i in range(1, len(velocities)):
        if velocities[i - 1, 1] > 4 and velocities[i, 1] < -4:
            events.append(Event(
                type=EventType.BOUNCE,
                frame_idx=int(frames[i]),
                ball_pos_px=(float(positions[i, 0]), float(positions[i, 1])),
                player_id=None,
                confidence=0.6,
            ))

    # ── 3. WALL_HIT: velocity reversal near court boundary ───────────────────
    if calibration is not None:
        events.extend(
            _detect_wall_hits(frames, positions, velocities, calibration, wall_margin_m)
        )

    events = _dedup_events(events, min_gap_frames=5)
    events.sort(key=lambda e: e.frame_idx)
    return events


# ── Wall-hit detection ───────────────────────────────────────────────────────

def _detect_wall_hits(
    frames: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    calibration: "CourtCalibration",
    wall_margin_m: float,
) -> list[Event]:
    """Detect ball-wall contacts via court-coordinate proximity + velocity reversal."""
    from app.ml.court import COURT_WIDTH_M, COURT_LENGTH_M

    court_pts = calibration.pixel_to_court(positions)  # Nx2 metres
    wall_events: list[Event] = []

    for i in range(1, len(velocities)):
        cx, cy = court_pts[i]
        near_l = cx < wall_margin_m
        near_r = cx > COURT_WIDTH_M  - wall_margin_m
        near_b = cy < wall_margin_m
        near_t = cy > COURT_LENGTH_M - wall_margin_m

        if not (near_l or near_r or near_b or near_t):
            continue

        vx0, vy0 = velocities[i - 1]
        vx1, vy1 = velocities[i]

        x_rev = (near_l and vx0 < -3 and vx1 > 3) or (near_r and vx0 > 3 and vx1 < -3)
        y_rev = (near_b and vy0 < -3 and vy1 > 3) or (near_t and vy0 > 3 and vy1 < -3)

        if not (x_rev or y_rev):
            continue

        wall_events.append(Event(
            type=EventType.WALL_HIT,
            frame_idx=int(frames[i]),
            ball_pos_px=(float(positions[i, 0]), float(positions[i, 1])),
            player_id=None,
            confidence=0.65,
        ))

    return wall_events


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_nearest_player(
    ball_pos: np.ndarray,
    players: list[PlayerDetection],
) -> tuple[int | None, float]:
    if not players:
        return None, float("inf")
    distances = [
        (p.track_id, float(np.linalg.norm(np.array(p.foot_px) - ball_pos)))
        for p in players
    ]
    return min(distances, key=lambda x: x[1])


def _dedup_events(events: list[Event], min_gap_frames: int) -> list[Event]:
    """Remove same-type events closer than min_gap_frames apart."""
    if not events:
        return events
    by_type: dict[EventType, list[Event]] = {}
    for e in events:
        by_type.setdefault(e.type, []).append(e)

    out: list[Event] = []
    for evs in by_type.values():
        evs.sort(key=lambda e: e.frame_idx)
        last = -(10 ** 9)
        for e in evs:
            if e.frame_idx - last >= min_gap_frames:
                out.append(e)
                last = e.frame_idx
    return out
