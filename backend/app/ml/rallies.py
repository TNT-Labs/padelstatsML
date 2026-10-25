"""Rally segmentation from player movement.

The old implementation segmented rallies from ball events, which meant rally
boundaries inherited every ball-tracking error. On a Pi there is no ball
tracker at all, so rallies are derived from what we can actually measure well:
how fast the four players are moving.

Between points players walk, stand and talk; during a point all four move
continuously. A smoothed mean court speed separates the two cleanly and needs
no ball, no audio and no scoreboard.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.ml.identity import PlayerTrack
from app.ml.smoothing import SmoothedPath

# Lowest threshold the match may choose: below it the filtered speed of a
# player standing still is no longer told from slow play.
ADAPTIVE_FLOOR_MS = 0.4
# Once started, a rally lasts while the mean speed stays above this share of
# the threshold that started it (1.0: a single threshold, as before).
HYSTERESIS_RATIO = 0.6
# The two kinds of moment — playing, between points — are told apart on
# this range of mean speeds; faster is play whatever the split.
_ADAPTIVE_RANGE_MS = 3.0


@dataclass(frozen=True)
class Rally:
    index: int
    start_s: float
    end_s: float
    players_tracked: int

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def contains(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
            "players_tracked": self.players_tracked,
        }


def detect_rallies(
    players: list[PlayerTrack],
    sample_hz: float,
    speed_threshold_ms: float = 1.1,
    min_duration_s: float = 3.0,
    merge_gap_s: float = 3.0,
    max_speed_ms: float = 8.0,
    min_players: int = 2,
    paths: dict[int, SmoothedPath] | None = None,
    hysteresis: float = HYSTERESIS_RATIO,
) -> list[Rally]:
    """Return the intervals during which a point is being played.

    With `paths` (app.ml.smoothing) the speeds are the filtered ones: the
    raw steps carry the detector's jitter, which made players standing
    between points look like they were moving."""
    series = _activity(players, sample_hz, max_speed_ms, paths)
    if series is None:
        return []
    times, smoothed, counts = series
    return _rallies(times, smoothed, counts, speed_threshold_ms, min_duration_s, merge_gap_s, min_players,
                    hysteresis)


def activity_threshold(
    players: list[PlayerTrack],
    sample_hz: float,
    ceiling_ms: float,
    max_speed_ms: float = 8.0,
    min_players: int = 2,
    paths: dict[int, SmoothedPath] | None = None,
) -> float:
    """The speed that separates play from the time between points, read
    from the match itself, never above `ceiling_ms`.

    With the jitter filtered out, a fixed threshold is a guess at how fast
    these players move: amateurs in a slow rally average well under the
    1.1 m/s that suited the raw speeds, and their rallies were lost. The
    match's mean speeds form two groups — standing or walking between
    points, moving in play — and the split that keeps each group tightest
    (Otsu's) is where one ends and the other begins."""
    series = _activity(players, sample_hz, max_speed_ms, paths)
    if series is None:
        return ceiling_ms
    _, smoothed, counts = series
    values = np.clip(smoothed[counts >= min_players], 0.0, _ADAPTIVE_RANGE_MS)
    if values.size < 20:
        return ceiling_ms
    histogram, edges = np.histogram(values, bins=60, range=(0.0, _ADAPTIVE_RANGE_MS))
    weights = histogram.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    total, total_sum = weights.sum(), float((weights * centres).sum())
    best, split = -1.0, ceiling_ms
    below = below_sum = 0.0
    for i in range(len(weights) - 1):
        below += weights[i]
        below_sum += weights[i] * centres[i]
        above = total - below
        if below == 0 or above == 0:
            continue
        between = below * above * (below_sum / below - (total_sum - below_sum) / above) ** 2
        if between > best:
            best, split = between, float(edges[i + 1])
    return float(min(ceiling_ms, max(split, ADAPTIVE_FLOOR_MS)))


def _activity(players, sample_hz, max_speed_ms, paths):
    """(times, mean speed smoothed over ~1 s, players counted) or None."""
    if not players or sample_hz <= 0:
        return None

    speeds_by_time: dict[float, list[float]] = defaultdict(list)
    sample_period = 1.0 / sample_hz
    # A step spanning more than two sample periods means the player was lost;
    # the implied speed is meaningless, so it is not evidence of activity.
    max_step_s = sample_period * 2.5

    for player in players:
        path = paths.get(player.player_id) if paths is not None else None
        if path is not None:
            for t, speed in zip(path.times, path.speeds):
                if speed <= max_speed_ms:
                    speeds_by_time[round(float(t), 3)].append(float(speed))
            continue
        observations = player.observations
        for prev, cur in zip(observations, observations[1:]):
            dt = cur.timestamp_s - prev.timestamp_s
            if dt <= 1e-3 or dt > max_step_s:
                continue
            step = float(
                np.hypot(
                    cur.foot_court[0] - prev.foot_court[0],
                    cur.foot_court[1] - prev.foot_court[1],
                )
            )
            speed = step / dt
            if speed > max_speed_ms:
                continue          # homography noise or an identity swap
            speeds_by_time[round(cur.timestamp_s, 3)].append(speed)

    if not speeds_by_time:
        return None

    times = np.array(sorted(speeds_by_time), dtype=np.float64)
    mean_speed = np.array(
        [float(np.mean(speeds_by_time[round(t, 3)])) for t in times], dtype=np.float64
    )
    counts = np.array([len(speeds_by_time[round(t, 3)]) for t in times], dtype=np.int32)

    window = max(3, int(round(sample_hz)) | 1)        # ~1 s, forced odd
    return times, _moving_median(mean_speed, window), counts


def _rallies(times, smoothed, counts, speed_threshold_ms, min_duration_s, merge_gap_s, min_players,
             hysteresis=HYSTERESIS_RATIO):
    enough = counts >= min_players
    if hysteresis >= 1.0:
        active = (smoothed >= speed_threshold_ms) & enough
    else:
        # A rally starts above the threshold and lasts while the play stays
        # above a lower one: the slow moments of a point — volleys at the
        # net, a lob watched — no longer cut it in pieces, and walking
        # between points still does not start one.
        low = max(ADAPTIVE_FLOOR_MS, hysteresis * speed_threshold_ms)
        active = np.zeros(len(times), dtype=bool)
        playing = False
        for i in range(len(times)):
            if not enough[i]:
                playing = False
            elif playing:
                playing = smoothed[i] >= low
            else:
                playing = smoothed[i] >= speed_threshold_ms
            active[i] = playing
    segments = _segments_from_mask(times, active)
    segments = _merge_close(segments, merge_gap_s)
    segments = [(a, b) for a, b in segments if (b - a) >= min_duration_s]

    rallies: list[Rally] = []
    for i, (start, end) in enumerate(segments):
        inside = (times >= start) & (times <= end)
        tracked = int(np.max(counts[inside])) if np.any(inside) else 0
        rallies.append(Rally(index=i, start_s=float(start), end_s=float(end), players_tracked=tracked))
    return rallies


def _moving_median(values: np.ndarray, window: int) -> np.ndarray:
    """Median filter with edge padding. Median, not mean: a single spurious
    high speed from an identity swap must not create a rally."""
    if window <= 1 or len(values) < window:
        return values
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(strided, axis=1)


def _segments_from_mask(times: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    start: float | None = None
    for t, flag in zip(times, mask):
        if flag and start is None:
            start = float(t)
        elif not flag and start is not None:
            segments.append((start, float(t)))
            start = None
    if start is not None:
        segments.append((start, float(times[-1])))
    return segments


def _merge_close(segments: list[tuple[float, float]], max_gap_s: float) -> list[tuple[float, float]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap_s:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged
