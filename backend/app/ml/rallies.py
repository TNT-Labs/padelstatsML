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
    merge_gap_s: float = 1.5,
    max_speed_ms: float = 8.0,
    min_players: int = 2,
) -> list[Rally]:
    """Return the intervals during which a point is being played."""
    if not players or sample_hz <= 0:
        return []

    speeds_by_time: dict[float, list[float]] = defaultdict(list)
    sample_period = 1.0 / sample_hz
    # A step spanning more than two sample periods means the player was lost;
    # the implied speed is meaningless, so it is not evidence of activity.
    max_step_s = sample_period * 2.5

    for player in players:
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
        return []

    times = np.array(sorted(speeds_by_time), dtype=np.float64)
    mean_speed = np.array(
        [float(np.mean(speeds_by_time[round(t, 3)])) for t in times], dtype=np.float64
    )
    counts = np.array([len(speeds_by_time[round(t, 3)]) for t in times], dtype=np.int32)

    window = max(3, int(round(sample_hz)) | 1)        # ~1 s, forced odd
    smoothed = _moving_median(mean_speed, window)

    active = (smoothed >= speed_threshold_ms) & (counts >= min_players)
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
