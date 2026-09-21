"""Tier 1 metrics: everything derivable from player positions on a validated
homography, and nothing else.

What is deliberately NOT here
-----------------------------
Shot types (smash / bandeja / volley), winners, errors and ball speed. All
four require ball tracking, which a Pi 5 cannot do at a useful frame rate,
and all four were previously produced by heuristics that returned confident
numbers with no relationship to the match. Removing them is the point of this
rewrite: five measured numbers are worth more than twenty plausible ones.

Every metric below carries its own reliability signal. `tracked_ratio` says
how much of the match a player was actually followed; `data_quality` says how
much to trust the whole set. A consumer that ignores them will still get
numbers, but the UI does not ignore them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, CourtCalibration
from app.ml.identity import IdentityResult, PlayerTrack
from app.ml.rallies import Rally

# Heatmap resolution. 0.5 m cells give 20x40 = 800 cells, detailed enough to
# show whether a player holds the net position and small enough to serialise.
CELL_M = 0.5
GRID_X = int(COURT_WIDTH_M / CELL_M)
GRID_Y = int(COURT_LENGTH_M / CELL_M)

# Court zones measured as distance from the net.
ZONE_NET_M = 3.0
ZONE_MID_M = 6.0


@dataclass(frozen=True)
class MetricsInput:
    players: list[PlayerTrack]
    rallies: list[Rally]
    identity: IdentityResult
    calibration: CourtCalibration
    sample_hz: float
    frames_sampled: int
    analysed_s: float
    max_speed_ms: float
    detector_model: str


def compute_metrics(data: MetricsInput) -> dict:
    """Assemble the full analysis result ready for JSON persistence."""
    per_player: dict[str, dict] = {}
    heatmaps: dict[str, list[list[float]]] = {}

    sample_period = 1.0 / data.sample_hz if data.sample_hz > 0 else 0.2
    max_step_s = sample_period * 2.5
    expected_samples = max(int(round(data.analysed_s * data.sample_hz)), 1)

    for player in data.players:
        stats, grid = _player_metrics(player, data, max_step_s, expected_samples)
        per_player[str(player.player_id)] = stats
        heatmaps[str(player.player_id)] = _serialise_grid(grid)

    summary = _summary(data)
    quality = _data_quality(data, per_player)

    return {
        "per_player": per_player,
        "heatmaps": heatmaps,
        "rallies": [r.to_dict() for r in data.rallies],
        "summary": summary,
        "data_quality": quality,
    }


# ── Per player ───────────────────────────────────────────────────────────────

def _player_metrics(
    player: PlayerTrack,
    data: MetricsInput,
    max_step_s: float,
    expected_samples: int,
) -> tuple[dict, np.ndarray]:
    observations = player.observations
    positions = np.array([o.foot_court for o in observations], dtype=np.float64)
    times = np.array([o.timestamp_s for o in observations], dtype=np.float64)

    grid = np.zeros((GRID_X, GRID_Y), dtype=np.float64)
    ix = np.clip((positions[:, 0] / CELL_M).astype(int), 0, GRID_X - 1)
    iy = np.clip((positions[:, 1] / CELL_M).astype(int), 0, GRID_Y - 1)
    np.add.at(grid, (ix, iy), 1.0)

    distance_total = 0.0
    distance_rally = 0.0
    speeds: list[float] = []
    rally_time = 0.0
    rejected_steps = 0

    for i in range(1, len(observations)):
        dt = times[i] - times[i - 1]
        if dt <= 1e-3 or dt > max_step_s:
            continue                      # gap in tracking, not a real step
        step = float(np.linalg.norm(positions[i] - positions[i - 1]))
        speed = step / dt
        if speed > data.max_speed_ms:
            rejected_steps += 1           # homography noise or an id swap
            continue

        distance_total += step
        speeds.append(speed)
        if _in_any_rally(times[i], data.rallies):
            distance_rally += step
            rally_time += dt

    zone = _zone_distribution(positions[:, 1])
    occupied_cells = int(np.count_nonzero(grid))

    speeds_arr = np.array(speeds, dtype=np.float64)
    avg_speed = float(distance_rally / rally_time) if rally_time > 0.5 else 0.0
    # 95th percentile rather than the max: one bad association should not
    # define a player's top speed.
    peak_speed = float(np.percentile(speeds_arr, 95)) if speeds_arr.size >= 10 else 0.0

    stats = {
        "team": player.team,
        "samples": len(observations),
        "tracked_ratio": round(min(1.0, len(observations) / expected_samples), 3),
        "distance_m": round(distance_total, 1),
        "distance_rally_m": round(distance_rally, 1),
        "avg_speed_ms": round(avg_speed, 2),
        "peak_speed_ms": round(peak_speed, 2),
        "coverage_m2": round(occupied_cells * CELL_M * CELL_M, 1),
        "zone_pct": zone,
        "rejected_steps": rejected_steps,
        "source_tracklets": len(player.source_tracklets),
    }
    return stats, grid


def _zone_distribution(ys: np.ndarray) -> dict[str, float]:
    """Share of samples spent at the net, mid-court and back, by distance
    from the net line. Position is the single most coached aspect of padel,
    so this is the metric that earns the analysis its keep."""
    if ys.size == 0:
        return {"net": 0.0, "mid": 0.0, "back": 0.0}
    distance_from_net = np.abs(ys - NET_Y_M)
    net = float((distance_from_net <= ZONE_NET_M).mean())
    mid = float(((distance_from_net > ZONE_NET_M) & (distance_from_net <= ZONE_MID_M)).mean())
    back = float((distance_from_net > ZONE_MID_M).mean())
    return {"net": round(net, 3), "mid": round(mid, 3), "back": round(back, 3)}


def _serialise_grid(grid: np.ndarray) -> list[list[float]]:
    total = float(grid.sum())
    if total <= 0:
        return []
    norm = grid / total
    # Drop cells below 0.05% of the player's time: they are single samples and
    # would triple the payload for no visible difference.
    threshold = 0.0005
    return [
        [
            round((x + 0.5) * CELL_M, 2),
            round((y + 0.5) * CELL_M, 2),
            round(float(norm[x, y]), 5),
        ]
        for x in range(GRID_X)
        for y in range(GRID_Y)
        if norm[x, y] > threshold
    ]


def _in_any_rally(t: float, rallies: list[Rally]) -> bool:
    return any(r.contains(t) for r in rallies)


# ── Match level ──────────────────────────────────────────────────────────────

def _summary(data: MetricsInput) -> dict:
    durations = np.array([r.duration_s for r in data.rallies], dtype=np.float64)
    total_rally_s = float(durations.sum()) if durations.size else 0.0

    return {
        "analysed_s": round(data.analysed_s, 1),
        "rallies_count": len(data.rallies),
        "total_rally_s": round(total_rally_s, 1),
        "avg_rally_s": round(float(durations.mean()), 1) if durations.size else 0.0,
        "median_rally_s": round(float(np.median(durations)), 1) if durations.size else 0.0,
        "longest_rally_s": round(float(durations.max()), 1) if durations.size else 0.0,
        "active_ratio": round(total_rally_s / data.analysed_s, 3) if data.analysed_s > 0 else 0.0,
        "players_found": len(data.players),
    }


def _data_quality(data: MetricsInput, per_player: dict[str, dict]) -> dict:
    warnings = list(data.identity.warnings)
    calib = data.calibration

    if calib.net_error_m is not None and calib.net_error_m > 0.5:
        warnings.append(
            f"La rete indicata cade a {calib.net_error_m:.1f} m dalla posizione attesa: "
            "gli angoli del campo sono probabilmente imprecisi."
        )

    poorly_tracked = [
        pid for pid, stats in per_player.items() if stats["tracked_ratio"] < 0.5
    ]
    if poorly_tracked:
        warnings.append(
            "Giocatori seguiti per meno della metà della partita: "
            + ", ".join(f"P{int(p) + 1}" for p in sorted(poorly_tracked))
            + ". Distanza e copertura sono sottostimate per questi giocatori."
        )

    if data.sample_hz < 4.0:
        warnings.append(
            f"Campionamento a {data.sample_hz:.1f} Hz: la distanza percorsa è "
            "sottostimata di circa il 15%."
        )

    return {
        "calibration_source": calib.source.value,
        "net_error_m": round(calib.net_error_m, 2) if calib.net_error_m is not None else None,
        "sample_hz": round(data.sample_hz, 2),
        "frames_sampled": data.frames_sampled,
        "players_found": len(data.players),
        "clusters_found": data.identity.clusters_found,
        "side_changes": data.identity.side_changes,
        "detector_model": data.detector_model,
        "metrics_tier": 1,
        "excluded_metrics": [
            "tipo di colpo",
            "vincenti/errori",
            "velocità palla",
            "punteggio",
        ],
        "warnings": warnings,
    }
