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

    def travel(self) -> float:
        return math.hypot(max(self.xs) - min(self.xs), max(self.ys) - min(self.ys))


def flights(found: list[tuple[int, float, float, float, float]], height: int) -> list[Flight]:
    """The ball's flights from the raw answers (frame, time, x, y, peak),
    in frame order, `height` the picture's height in pixels."""
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
            if d < best_d:
                best, best_d = flight, d
        if best is None:
            best = Flight()
            active.append(best)
        best.add(frame, t, x, y)
    done += active
    keep = [f for f in done if len(f.frames) >= MIN_POINTS and f.travel() >= MIN_TRAVEL * height]
    return sorted(keep, key=lambda f: f.frames[0])


def track_payload(found: list[tuple[int, float, float, float, float]], width: int, height: int) -> dict:
    """The ball for the web player: per flight its times and points, in
    video pixels rounded to the pixel."""
    out = flights(found, height)
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
CACHE_VERSION = 1


def track_gz(directory, width: int, height: int) -> bytes | None:
    """track_payload of the run in `directory`, gzip-compressed JSON,
    cached beside ball.jsonl; None when the run followed no ball."""
    import gzip
    import json
    from pathlib import Path

    from app.ml.artifacts import BALL_FILE, read_ball

    directory = Path(directory)
    source = directory / BALL_FILE
    if not source.exists():
        return None
    cache = directory / f"ball-track-v{CACHE_VERSION}.json.gz"
    if cache.exists() and cache.stat().st_mtime >= source.stat().st_mtime:
        return cache.read_bytes()
    data = gzip.compress(json.dumps(track_payload(read_ball(directory), width, height),
                                    separators=(",", ":")).encode(), compresslevel=6)
    part = cache.with_name(cache.name + ".part")
    part.write_bytes(data)
    part.replace(cache)
    return data
