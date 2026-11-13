"""From the ball found frame by frame to the ball's flights.

The network answers on each frame alone: most answers are the ball, some
are not — a ball on the next court, a light in the roof, a shoe — and a
line drawn through all of them jumps across the picture. The real ball
moves a little from one frame to the next, along a path; a false answer
does not come back frame after frame where a ball in flight would be.

So the answers are chained into tracklets: each joins the tracklet whose
last point it is near enough to (MAX_STEP of the picture's height per
frame, across gaps of up to MAX_GAP frames: the ball behind a player), the
nearest to where that tracklet was heading. Tracklets too short (MIN_POINTS)
are noise; tracklets that never move (MIN_TRAVEL) are something still —
a light, a ball lying on the floor — and both are dropped.

Then, for runs that kept the heat map's weaker peaks (ball_weak.jsonl),
each flight is followed further (recover): into its gaps, where a peak lies
on the line between the points around it, and past its ends, frame by
frame, where a peak (or an answer the chaining left out) lies where the
flight was heading — by the rule that chains the flight itself. The ball
coming to the far pair is two pixels and often the second brightest thing
in the map: it was missing in the half second before more than half of
their hits. Without ball_weak.jsonl nothing is recovered: the flights are
those of before. How far to trust the weaker peaks is RECOVERY.

All in image pixels, as fractions of the picture's height: the same rules
for any resolution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Per frame at 30 fps: a fast ball near the camera crosses a few metres of
# picture in a second; 8% of the height a frame is about 2.5 pictures a second.
MAX_STEP = 0.08
MAX_GAP = 6                  # frames: 0.2 s at 30 fps
MIN_POINTS = 5
MIN_TRAVEL = 0.02            # of the height, end to end of the tracklet's extent
# Once a flight has a direction, the next point must be where it was going:
# within this much of the height, plus half of what it moves in the gap.
# A hit or a bounce turns it more than that: there a new flight starts,
# and a stray answer a few frames on is not taken for the ball.
ON_COURSE = 0.03
# Recovering: in a gap, within this much of the height of the straight line
# between the points around it; past an end, at most this many frames on
# (0.5 s at 30 fps) — a weak track followed further drifts onto noise.
IN_GAP = 0.015
RECOVER_MAX = 15


@dataclass(frozen=True)
class Recovery:
    """How the flights are followed with the weaker peaks: peaks at least
    `min_peak` (the answers no flight took count as 1), into the gaps
    and/or past the ends, the distances allowed scaled by `tolerance`."""
    min_peak: float = 0.15
    gaps: bool = True
    ends: bool = True
    tolerance: float = 1.0


# Off until a setting is shown, on the marked frames (training/ball/
# peaks_check.py), to add more ball than noise: as first written (gaps and
# ends, 0.15, full tolerance) it added 13 wrong points for every 8 right
# ones. The weaker peaks are stored anyway: the setting chosen then reaches
# the runs analysed now through Aggiorna statistiche.
RECOVERY = Recovery(gaps=False, ends=False)


@dataclass
class Flight:
    """A tracklet of the ball: its points in frame order."""
    frames: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)

    def add(self, frame: int, t: float, x: float, y: float) -> None:
        self.frames.append(frame)
        self.times.append(t)
        self.xs.append(x)
        self.ys.append(y)

    def expected(self, frame: int) -> tuple[float, float]:
        """Where it would be at `frame`, going on as it was going."""
        if len(self.frames) < 2:
            return self.xs[-1], self.ys[-1]
        span = self.frames[-1] - self.frames[-2]
        ahead = (frame - self.frames[-1]) / max(span, 1)
        return (self.xs[-1] + (self.xs[-1] - self.xs[-2]) * ahead,
                self.ys[-1] + (self.ys[-1] - self.ys[-2]) * ahead)

    def speed(self) -> float:
        """Pixels per frame between its last two points."""
        if len(self.frames) < 2:
            return 0.0
        span = max(self.frames[-1] - self.frames[-2], 1)
        return math.hypot(self.xs[-1] - self.xs[-2], self.ys[-1] - self.ys[-2]) / span

    def travel(self) -> float:
        return math.hypot(max(self.xs) - min(self.xs), max(self.ys) - min(self.ys))


def flights(found: list[tuple[int, float, float, float, float]], height: int, weak=None,
            recovery: Recovery | None = None) -> list[Flight]:
    """The ball's flights from the raw answers (frame, time, x, y, peak),
    in frame order, `height` the picture's height in pixels; with `weak`
    (artifacts.read_ball_weak) followed further (recover)."""
    step = MAX_STEP * height
    active: list[Flight] = []
    done: list[Flight] = []
    for frame, t, x, y, _peak in found:
        still = []
        for flight in active:
            (still if frame - flight.frames[-1] <= MAX_GAP else done).append(flight)
        active = still
        best, best_d = None, math.inf
        for flight in active:
            gap = frame - flight.frames[-1]
            if gap <= 0 or math.hypot(x - flight.xs[-1], y - flight.ys[-1]) > step * gap:
                continue
            ex, ey = flight.expected(frame)
            d = math.hypot(x - ex, y - ey)
            if len(flight.frames) >= 2 and d > ON_COURSE * height + 0.5 * flight.speed() * gap:
                continue
            if d < best_d:
                best, best_d = flight, d
        if best is None:
            best = Flight()
            active.append(best)
        best.add(frame, t, x, y)
    done += active
    keep = [f for f in done if len(f.frames) >= MIN_POINTS and f.travel() >= MIN_TRAVEL * height]
    keep.sort(key=lambda f: f.frames[0])
    if weak is not None:
        recover(keep, found, weak, height, recovery or RECOVERY)
    return keep


Candidates = dict[int, list[tuple[float, float, float]]]   # frame -> [(time, x, y)]


def recover(kept: list[Flight], found, weak, height: int, how: Recovery = RECOVERY) -> int:
    """Extend the flights `kept` (in place) with the weaker peaks `weak`
    (frame -> (time, [(x, y, peak)])) and the answers in `found` no flight
    took. Returns how many points were added."""
    taken = {fr for f in kept for fr in f.frames}
    pool: Candidates = {}
    for frame, t, x, y, _peak in found:
        if frame not in taken:
            pool.setdefault(frame, []).append((t, x, y))
    for frame, (t, peaks) in weak.items():
        if frame not in taken:
            pool.setdefault(frame, []).extend((t, x, y) for x, y, p in peaks if p >= how.min_peak)
    added = 0
    for flight in kept:
        if how.gaps:
            added += _fill_gaps(flight, pool, taken, height, how.tolerance)
        if how.ends:
            added += _extend(flight, pool, taken, height, True, how.tolerance)
            added += _extend(flight, pool, taken, height, False, how.tolerance)
    return added


def _nearest(pool: Candidates, frame: int, ex: float, ey: float, limit: float):
    best, best_d = None, limit
    for c in pool.get(frame, ()):
        d = math.hypot(c[1] - ex, c[2] - ey)
        if d <= best_d:
            best, best_d = c, d
    return best


def _take(flight: Flight, pool: Candidates, taken: set[int], frame: int, c) -> None:
    flight.add(frame, *c)
    taken.add(frame)
    pool.pop(frame, None)


def _fill_gaps(flight: Flight, pool: Candidates, taken: set[int], height: int, tolerance: float = 1.0) -> int:
    added = 0
    points = list(zip(flight.frames, flight.xs, flight.ys))
    for (f0, x0, y0), (f1, x1, y1) in zip(points, points[1:]):
        for frame in range(f0 + 1, f1):
            if frame in taken:
                continue
            k = (frame - f0) / (f1 - f0)
            c = _nearest(pool, frame, x0 + (x1 - x0) * k, y0 + (y1 - y0) * k,
                         IN_GAP * height * tolerance)
            if c is not None:
                _take(flight, pool, taken, frame, c)
                added += 1
    if added:
        _sort(flight)
    return added


def _extend(flight: Flight, pool: Candidates, taken: set[int], height: int, forward: bool,
            tolerance: float = 1.0) -> int:
    """Past one end, by the rule the chaining uses: near enough the last
    point for the gap, where the flight was heading."""
    step = MAX_STEP * height
    sign = 1 if forward else -1
    pts = list(zip(flight.frames, flight.times, flight.xs, flight.ys))
    if not forward:
        pts.reverse()
    end = pts[-1][0]
    added = []
    while abs(pts[-1][0] - end) < RECOVER_MAX:
        (fa, _, xa, ya), (fb, _, xb, yb) = pts[-2], pts[-1]
        span = max(abs(fb - fa), 1)
        vx, vy = (xb - xa) / span, (yb - ya) / span
        speed = math.hypot(vx, vy)
        found = None
        for gap in range(1, MAX_GAP + 1):
            frame = fb + sign * gap
            if frame in taken or abs(frame - end) > RECOVER_MAX:
                break
            ex, ey = xb + vx * gap, yb + vy * gap
            c = _nearest(pool, frame, ex, ey, (ON_COURSE * height + 0.5 * speed * gap) * tolerance)
            if c is not None and math.hypot(c[1] - xb, c[2] - yb) <= step * gap:
                found = (frame, c)
                break
        if found is None:
            break
        frame, c = found
        taken.add(frame)
        pool.pop(frame, None)
        pts.append((frame, *c))
        added.append((frame, *c))
    for frame, t, x, y in added:
        flight.add(frame, t, x, y)
    if added:
        _sort(flight)
    return len(added)


def _sort(flight: Flight) -> None:
    order = sorted(range(len(flight.frames)), key=lambda i: flight.frames[i])
    flight.frames = [flight.frames[i] for i in order]
    flight.times = [flight.times[i] for i in order]
    flight.xs = [flight.xs[i] for i in order]
    flight.ys = [flight.ys[i] for i in order]


def track_payload(found: list[tuple[int, float, float, float, float]], width: int, height: int,
                  weak=None) -> dict:
    """The ball for the web player: per flight its times and points, in
    video pixels rounded to the pixel."""
    out = flights(found, height, weak)
    return {
        "frame": [width, height],
        "seen": sum(len(f.frames) for f in out),
        "raw": len(found),
        "flights": [
            {"t": [round(t, 3) for t in f.times],
             "x": [round(x) for x in f.xs],
             "y": [round(y) for y in f.ys]}
            for f in out
        ],
    }


# Bump when the flights are made differently: older cached files are rebuilt.
CACHE_VERSION = 3


def track_gz(directory, width: int, height: int) -> bytes | None:
    """track_payload of the run in `directory`, gzip-compressed JSON,
    cached beside ball.jsonl; None when the run followed no ball."""
    import gzip
    import json
    from pathlib import Path

    from app.ml.artifacts import BALL_FILE, BALL_WEAK_FILE, read_ball, read_ball_weak

    directory = Path(directory)
    source = directory / BALL_FILE
    if not source.exists():
        return None
    cache = directory / f"ball-track-v{CACHE_VERSION}.json.gz"
    sources = [p for p in (source, directory / BALL_WEAK_FILE) if p.exists()]
    if cache.exists() and all(cache.stat().st_mtime >= p.stat().st_mtime for p in sources):
        return cache.read_bytes()
    data = gzip.compress(json.dumps(track_payload(read_ball(directory), width, height, read_ball_weak(directory)),
                                    separators=(",", ":")).encode(), compresslevel=6)
    part = cache.with_name(cache.name + ".part")
    part.write_bytes(data)
    part.replace(cache)
    return data
