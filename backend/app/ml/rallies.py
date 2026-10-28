"""Rally segmentation from player movement.

The old implementation segmented rallies from ball events, which meant rally
boundaries inherited every ball-tracking error. On a Pi there is no ball
tracker at all, so rallies are derived from what we can actually measure well:
how fast the four players are moving.

Between points players walk, stand and talk; during a point all four move
continuously. A smoothed mean court speed separates the two cleanly and needs
no ball, no audio and no scoreboard.

Every point starts with a serve, and before it the four stand still in
their places: the server and the receiver at the back of their halves, the
partners waiting. A point that ends and the next served a few seconds later
used to stay one rally — the pause too short, or walking to the places
lively enough, to cut them. The serve formation followed by the start of
play cuts them where the new point begins.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.ml.court import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M
from app.ml.identity import PlayerTrack
from app.ml.smoothing import SmoothedPath

# Lowest threshold the match may choose: below it the filtered speed of a
# player standing still is no longer told from slow play.
ADAPTIVE_FLOOR_MS = 0.4
# Once started, a rally lasts while the mean speed stays above this share of
# the threshold that started it (1.0: a single threshold, as before).
HYSTERESIS_RATIO = 0.6
# Serve formation: every tracked player — at least three, both halves — has
# stood (median filtered speed below SERVE_STILL_MS, never above
# SERVE_STILL_MAX_MS) for SERVE_STILL_S, and the places are those of a
# serve: a player at the back of one half receives, the other half has its
# server at the back (diagonally opposite the receiver) or, the server not
# in sight, the partner alone.
SERVE_STILL_S = 1.0
SERVE_STILL_MS = 0.6
SERVE_STILL_MAX_MS = 1.2
# A player present for at least this share of the window counts.
SERVE_PRESENT_SHARE = 0.6
# Behind the service line (3.05 m from the back wall), with the far feet'
# error.
SERVE_BACK_M = 4.0
# Within this distance of the centre line the side is not told.
SERVE_CENTRE_M = 0.5
# Play must start within this time of the formation breaking up: otherwise
# the players were standing, not waiting for a serve.
SERVE_ONSET_MAX_S = 3.0
# Play that stops for a new formation sooner than this was not a point (a
# fault, a ball hit across between points): no serve.
SERVE_MIN_PLAY_S = 3.0
# A rally starting this close to a serve was opened by it.
SERVE_MATCH_S = 1.0
# The two kinds of moment — playing, between points — are told apart on
# this range of mean speeds; faster is play whatever the split.
_ADAPTIVE_RANGE_MS = 3.0


@dataclass(frozen=True)
class Rally:
    index: int
    start_s: float
    end_s: float
    players_tracked: int
    # Opened by a serve the players were seen lining up for.
    serve: bool = False

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
            "serve": self.serve,
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
    segments = _segments(times, smoothed, counts, speed_threshold_ms, merge_gap_s, min_players, hysteresis)
    serves = serve_starts(players, sample_hz, times, smoothed, speed_threshold_ms, paths)
    segments = _split_at_serves(segments, serves, times, smoothed, speed_threshold_ms, min_duration_s)
    return _numbered(segments, times, counts, min_duration_s, [served for _, served in serves])


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


def _segments(times, smoothed, counts, speed_threshold_ms, merge_gap_s, min_players,
              hysteresis=HYSTERESIS_RATIO) -> list[tuple[float, float]]:
    """Moments of play, pauses shorter than `merge_gap_s` bridged."""
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
    return _merge_close(_segments_from_mask(times, active), merge_gap_s)


def _numbered(segments, times, counts, min_duration_s, serves=()) -> list[Rally]:
    segments = [(a, b) for a, b in segments if (b - a) >= min_duration_s]
    rallies: list[Rally] = []
    for i, (start, end) in enumerate(segments):
        inside = (times >= start) & (times <= end)
        tracked = int(np.max(counts[inside])) if np.any(inside) else 0
        served = any(abs(t - start) <= SERVE_MATCH_S for t in serves)
        rallies.append(Rally(index=i, start_s=float(start), end_s=float(end), players_tracked=tracked,
                             serve=served))
    return rallies


def serve_starts(
    players: list[PlayerTrack],
    sample_hz: float,
    times: np.ndarray,
    smoothed: np.ndarray,
    speed_threshold_ms: float,
    paths: dict[int, SmoothedPath] | None = None,
) -> list[tuple[float, float]]:
    """(formation start, serve) of every serve found: the time the players
    took their places, and the time play started from them — the first
    moment after the formation broke up at which the mean speed `smoothed`
    reached `speed_threshold_ms`.

    Formations are read on the players' filtered paths; without them, on
    the raw positions, whose jitter rarely lets anybody look still."""
    if len(times) == 0 or sample_hz <= 0:
        return []
    places = [_aligned(player, times, sample_hz, paths) for player in players]
    window = max(2, int(round(SERVE_STILL_S * sample_hz)))
    formed = np.zeros(len(times), dtype=bool)
    for i in range(window - 1, len(times)):
        if times[i] - times[i - window + 1] > SERVE_STILL_S * 1.5:
            continue          # samples missing: not a second of watching
        formed[i] = _is_formation(places, slice(i - window + 1, i + 1), window)

    runs: list[tuple[int, int]] = []          # (first, last) index of each formation
    for i in np.flatnonzero(formed):
        if runs and runs[-1][1] == i - 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))

    serves: list[tuple[float, float]] = []
    for n, (first, last) in enumerate(runs):
        onset = next(
            (k for k in range(last + 1, len(times))
             if smoothed[k] >= speed_threshold_ms or times[k] - times[last] > SERVE_ONSET_MAX_S),
            None,
        )
        if onset is None or smoothed[onset] < speed_threshold_ms:
            continue
        served = float(times[onset])
        later = [times[f] - SERVE_STILL_S for f, _ in runs[n + 1:] if times[f] - SERVE_STILL_S > served]
        if later and later[0] - served < SERVE_MIN_PLAY_S:
            continue
        formed_at = max(float(times[0]), float(times[first]) - SERVE_STILL_S)
        if serves and serves[-1][1] == served:
            continue          # two formations broken up by the same serve
        serves.append((formed_at, served))
    return serves


def _aligned(player, times, sample_hz, paths) -> tuple[np.ndarray, np.ndarray]:
    """(positions (n, 2), speeds (n,)) of one player on the activity grid,
    NaN where the player was not seen."""
    path = paths.get(player.player_id) if paths is not None else None
    if path is not None:
        when, where, speed = path.times, path.positions, path.speeds
    else:
        obs = player.observations
        when = np.array([o.timestamp_s for o in obs], dtype=np.float64)
        where = np.array([o.foot_court for o in obs], dtype=np.float64).reshape(-1, 2)
        speed = np.full(len(obs), np.nan)
        if len(obs) > 1:
            dt = np.diff(when)
            steps = np.linalg.norm(np.diff(where, axis=0), axis=1)
            speed[1:] = np.where(dt > 1e-3, steps / np.maximum(dt, 1e-3), np.nan)
    positions = np.full((len(times), 2), np.nan)
    speeds = np.full(len(times), np.nan)
    if len(when) == 0:
        return positions, speeds
    after = np.clip(np.searchsorted(when, times), 0, len(when) - 1)
    before = np.maximum(after - 1, 0)
    k = np.where(np.abs(when[before] - times) < np.abs(when[after] - times), before, after)
    near = np.abs(when[k] - times) <= 0.5 / sample_hz
    positions[near] = where[k[near]]
    speeds[near] = speed[k[near]]
    return positions, speeds


def _is_formation(places, window: slice, size: int) -> bool:
    """Whether the players stand as before a serve during `window`."""
    seen: list[tuple[float, float]] = []
    for positions, speeds in places:
        present = ~np.isnan(speeds[window])
        if present.sum() < SERVE_PRESENT_SHARE * size:
            continue
        moving = speeds[window][present]
        if np.median(moving) > SERVE_STILL_MS or moving.max() > SERVE_STILL_MAX_MS:
            return False
        x, y = np.median(positions[window][present], axis=0)
        seen.append((float(x), float(y)))
    if len(seen) < 3:
        return False
    halves = ([p for p in seen if p[1] < NET_Y_M], [p for p in seen if p[1] >= NET_Y_M])
    for receiving in (0, 1):
        receivers = [p for p in halves[receiving] if _at_back(p) and _side(p) != 0]
        serving = halves[1 - receiving]
        if not receivers or not serving:
            continue
        servers = [p for p in serving if _at_back(p)]
        if len(serving) == 1 and not servers:
            return True       # the partner at the net; the server out of sight
        if any(_side(s) * _side(r) < 0 for s in servers for r in receivers):
            return True
    return False


def _at_back(place: tuple[float, float]) -> bool:
    y = place[1]
    return y <= SERVE_BACK_M or y >= COURT_LENGTH_M - SERVE_BACK_M


def _side(place: tuple[float, float]) -> int:
    """-1 / +1 for the two sides of the centre line, 0 on it."""
    offset = place[0] - COURT_WIDTH_M / 2.0
    return 0 if abs(offset) < SERVE_CENTRE_M else (1 if offset > 0 else -1)


def _split_at_serves(segments, serves, times, smoothed, speed_threshold_ms, min_duration_s):
    """Cut every moment of play at the serves inside it: the point before
    ends where play last reached the threshold before the players took
    their places for the next serve."""
    out: list[tuple[float, float]] = []
    for start, end in segments:
        for formed, served in serves:
            if not (start + min_duration_s <= served <= end - min_duration_s):
                continue
            before = (times > start) & (times < formed) & (smoothed >= speed_threshold_ms)
            finish = float(times[before][-1]) if before.any() else formed
            out.append((start, max(start, finish)))
            start = served
        out.append((start, end))
    return out


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
