"""Final statistics aggregation from all pipeline stages.

Stats computed:
  1. Distance travelled (m)       — sum of step lengths in court coords
  2. Position heatmap             — 10×20 cell histogram normalised to weights
  3. Shot counts by type          — from classified_shots
  4. Winners / errors per player  — ball-trajectory heuristic at rally end
  5. Rally list                   — frame segments with last hitter

Winners/errors heuristic (replaces the previous always-winner placeholder):
  - If ball's last detected position in a rally is within 0.5 m of a court
    boundary → likely hit out or into the wall → ERROR for last hitter.
  - Otherwise → ball stayed in the court when the rally ended → WINNER.
  - If no ball data is available for the rally → no attribution.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from app.ml.court import CourtCalibration, COURT_WIDTH_M, COURT_LENGTH_M
from app.ml.events import Event, EventType
from app.ml.players import PlayerDetection
from app.ml.shots import ClassifiedShot, ShotType

if TYPE_CHECKING:
    from app.ml.ball import BallDetection

# Ball out-of-bounds margin: if last ball pos is within this distance from
# any court boundary, we classify the rally end as an error.
_OUT_MARGIN_M = 0.6


def compute_stats(
    player_detections: list[list[PlayerDetection]],
    classified_shots: list[ClassifiedShot],
    rallies: list[tuple[int, int, int | None]],   # (start_frame, end_frame, last_hitter_track_id)
    id_mapping: dict[int, int],                   # YOLO track_id → canonical 0..3
    calibration: CourtCalibration,
    fps: float,
    ball_track: list["BallDetection"] | None = None,
) -> dict:
    n_players = 4

    # ── 1. Distance (court coords, metres) ──────────────────────────────────
    distance_m: dict[int, float] = defaultdict(float)
    last_pos: dict[int, tuple[float, float]] = {}

    for frame_dets in player_detections:
        for d in frame_dets:
            pid = id_mapping.get(d.track_id)
            if pid is None:
                continue
            if pid in last_pos:
                lx, ly = last_pos[pid]
                step = float(np.hypot(d.foot_court[0] - lx, d.foot_court[1] - ly))
                if step < 5.0:            # discard teleport artifacts
                    distance_m[pid] += step
            last_pos[pid] = d.foot_court

    # ── 2. Heatmaps (10×20 grid, 1 m cells, sparse) ─────────────────────────
    heatmaps_acc: dict[int, np.ndarray] = {i: np.zeros((10, 20)) for i in range(n_players)}
    for frame_dets in player_detections:
        for d in frame_dets:
            pid = id_mapping.get(d.track_id)
            if pid is None:
                continue
            ix = int(np.clip(d.foot_court[0], 0, 9.99))
            iy = int(np.clip(d.foot_court[1], 0, 19.99))
            heatmaps_acc[pid][ix, iy] += 1

    heatmaps_serialized: dict[str, list[list[float]]] = {}
    for pid, hm in heatmaps_acc.items():
        total = hm.sum()
        if total == 0:
            heatmaps_serialized[str(pid)] = []
            continue
        norm = hm / total
        points = [
            [float(ix + 0.5), float(iy + 0.5), float(norm[ix, iy])]
            for ix in range(10)
            for iy in range(20)
            if norm[ix, iy] > 0.001
        ]
        heatmaps_serialized[str(pid)] = points

    # ── 3. Shot counts by type ───────────────────────────────────────────────
    shots_by_player: dict[int, dict[str, int]] = {
        i: {st.value: 0 for st in ShotType} for i in range(n_players)
    }
    for cs in classified_shots:
        if cs.event.player_id is None:
            continue
        pid = id_mapping.get(cs.event.player_id)
        if pid is None:
            continue
        shots_by_player[pid][cs.shot_type.value] += 1

    # ── 4. Winners / errors ──────────────────────────────────────────────────
    # Build a lookup: frame_idx → ball position (metres) for quick access
    ball_court_by_frame: dict[int, tuple[float, float]] = {}
    if ball_track:
        valid_ball = [b for b in ball_track if b.pos_px is not None]
        if valid_ball:
            pxs = np.array([b.pos_px for b in valid_ball], dtype=np.float32)
            court_pts = calibration.pixel_to_court(pxs)
            for b, cp in zip(valid_ball, court_pts):
                ball_court_by_frame[b.frame_idx] = (float(cp[0]), float(cp[1]))

    winners: dict[int, int] = defaultdict(int)
    errors:  dict[int, int] = defaultdict(int)

    for start_f, end_f, last_hitter_tid in rallies:
        if last_hitter_tid is None:
            continue
        pid = id_mapping.get(last_hitter_tid)
        if pid is None:
            continue

        outcome = _classify_rally_outcome(start_f, end_f, ball_court_by_frame)
        if outcome == "winner":
            winners[pid] += 1
        elif outcome == "error":
            errors[pid] += 1
        # "unknown" → no attribution

    # ── 5. Assemble result ───────────────────────────────────────────────────
    per_player: dict[str, dict] = {}
    for pid in range(n_players):
        per_player[str(pid)] = {
            "distance_m": round(distance_m[pid], 2),
            "winners":    winners[pid],
            "errors":     errors[pid],
            "shots":      shots_by_player[pid],
        }

    return {
        "per_player": per_player,
        "heatmaps":   heatmaps_serialized,
        "rallies": [
            {
                "start_frame": s,
                "end_frame":   e,
                "last_player": id_mapping.get(p) if p is not None else None,
            }
            for s, e, p in rallies
        ],
        "court_calibration": calibration.to_dict(),
    }


def _classify_rally_outcome(
    start_frame: int,
    end_frame: int,
    ball_court_by_frame: dict[int, tuple[float, float]],
    tail_frames: int = 20,
) -> str:
    """Return 'winner', 'error', or 'unknown' for a rally.

    Looks at ball positions in the last `tail_frames` of the rally.
    If the ball was last seen within _OUT_MARGIN_M of any court boundary
    → error (ball went out or into the wall).
    Otherwise → winner (ball stayed in court; opponent failed to return).
    """
    # Gather ball positions from the tail of the rally
    tail_start = end_frame - tail_frames
    positions = [
        pos for f, pos in ball_court_by_frame.items()
        if tail_start <= f <= end_frame + 10   # slight look-ahead
    ]

    if not positions:
        return "unknown"

    # Use the last known position
    x, y = positions[-1]

    near_boundary = (
        x < _OUT_MARGIN_M or x > COURT_WIDTH_M  - _OUT_MARGIN_M or
        y < _OUT_MARGIN_M or y > COURT_LENGTH_M - _OUT_MARGIN_M
    )

    return "error" if near_boundary else "winner"


def detect_rallies(
    events: list[Event],
    min_rally_gap_frames: int = 60,
) -> list[tuple[int, int, int | None]]:
    """Segment events into rallies separated by pauses ≥ min_rally_gap_frames."""
    if not events:
        return []

    sorted_evs = sorted(events, key=lambda e: e.frame_idx)
    rallies: list[tuple[int, int, int | None]] = []
    start      = sorted_evs[0].frame_idx
    last       = sorted_evs[0].frame_idx
    last_hitter: int | None = (
        sorted_evs[0].player_id if sorted_evs[0].type == EventType.HIT else None
    )

    for ev in sorted_evs[1:]:
        if ev.frame_idx - last > min_rally_gap_frames:
            rallies.append((start, last, last_hitter))
            start       = ev.frame_idx
            last_hitter = None
        last = ev.frame_idx
        if ev.type == EventType.HIT:
            last_hitter = ev.player_id

    rallies.append((start, last, last_hitter))
    return rallies
