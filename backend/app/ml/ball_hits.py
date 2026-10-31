"""The hits of a rally: when the ball left a racket, and whose.

The ball's flights (app/ml/ball_track.py) break where its course turns
sharply: at a hit, a bounce, the glass. A hit is the turn that happens at a
player: the flight starts at them, or the flight before ends at them and the
next one starts a moment later (the ball behind the body at the stroke).

"At a player" is in the picture: the ball inside the player's box widened
by REACH_W box widths on each side (the arm and the racket) and REACH_UP box
heights above the top (an overhead), and not in the bottom quarter of the
box, where the ball bounces at the feet rather than leaves the racket.

One stroke can open more than one flight (the ball hidden for a frame, a
bounce at the feet just before): hits by the same player within SAME_PLAYER_S
are one, and by two team-mates within SAME_TEAM_S too — the ball cannot go
to the other side of the net and back that fast. Only hits inside a rally
count (from RALLY_LEAD_S before its start: the serve).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

REACH_W = 0.6
REACH_UP = 0.5
MIN_REACH = 0.04           # of the picture's height: far players are small
FEET = 0.75                # below this share of the box: the feet
SAME_PLAYER_S = 0.6
SAME_TEAM_S = 0.35
HIDDEN_S = 0.3             # a flight that ends at a player, the next one starts within this
MAX_BOX_GAP_S = 0.5        # boxes interpolated across at most this
RALLY_LEAD_S = 1.0


@dataclass(frozen=True)
class Hit:
    t: float
    player: int
    team: int


class PlayerBoxes:
    """A player's boxes in time, interpolated between samples."""

    def __init__(self, player_id: int, team: int, samples: list[tuple[float, tuple]]) -> None:
        self.player_id, self.team = player_id, team
        samples = sorted(samples, key=lambda s: s[0])
        self.times = [s[0] for s in samples]
        self.boxes = [s[1] for s in samples]

    def at(self, t: float):
        i = bisect.bisect_right(self.times, t) - 1
        if i < 0:
            return self.boxes[0] if self.times and self.times[0] - t <= MAX_BOX_GAP_S / 2 else None
        if i + 1 < len(self.times) and self.times[i + 1] - self.times[i] <= MAX_BOX_GAP_S:
            f = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
            a, b = self.boxes[i], self.boxes[i + 1]
            return tuple(a[k] + (b[k] - a[k]) * f for k in range(4))
        return self.boxes[i] if t - self.times[i] <= MAX_BOX_GAP_S / 2 else None


def reach(box, x: float, y: float, height: int) -> float | None:
    """How far (x, y) is from the player's racket zone, in box heights; None
    when outside it."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    side = max(REACH_W * w, MIN_REACH * height)
    if not (x1 - side <= x <= x2 + side and y1 - max(REACH_UP * h, MIN_REACH * height) <= y <= y1 + FEET * h):
        return None
    cx, cy = (x1 + x2) / 2, y1 + 0.25 * h          # about the shoulders
    return math.hypot(x - cx, y - cy) / max(h, 1.0)


def _who(players: list[PlayerBoxes], t: float, x: float, y: float, height: int) -> PlayerBoxes | None:
    best, best_d = None, math.inf
    for player in players:
        box = player.at(t)
        if box is None:
            continue
        d = reach(box, x, y, height)
        if d is not None and d < best_d:
            best, best_d = player, d
    return best


def find_hits(flights, players: list[PlayerBoxes], rallies: list[dict], height: int) -> list[Hit]:
    """The hits, in time order, inside the rallies."""
    candidates: list[Hit] = []
    previous = None
    for flight in flights:
        t, x, y = flight.times[0], flight.xs[0], flight.ys[0]
        who = _who(players, t, x, y, height)
        if who is None and previous is not None and 0 < t - previous.times[-1] <= HIDDEN_S:
            # Hidden at the stroke: where the flight before ended.
            t, x, y = previous.times[-1], previous.xs[-1], previous.ys[-1]
            who = _who(players, t, x, y, height)
        if who is not None:
            candidates.append(Hit(round(t, 2), who.player_id, who.team))
        previous = flight

    spans = [(r["start_s"] - RALLY_LEAD_S, r["end_s"]) for r in rallies]
    out: list[Hit] = []
    for hit in sorted(candidates, key=lambda h: h.t):
        if not any(a <= hit.t <= b for a, b in spans):
            continue
        if out:
            last = out[-1]
            if last.player == hit.player and hit.t - last.t <= SAME_PLAYER_S:
                continue
            if last.team == hit.team and hit.t - last.t <= SAME_TEAM_S:
                continue
        out.append(hit)
    return out


def attach_hits(result: dict, identity, directory, height: int) -> None:
    """Hits into the result, when the run followed the ball: per rally the
    list [time, player], per player the count. Nothing without the ball."""
    from app.ml.artifacts import read_ball
    from app.ml.ball_track import flights

    found = read_ball(directory)
    if not found or not height:
        return
    players = [
        PlayerBoxes(p.player_id, p.team, [(float(o.timestamp_s), tuple(float(v) for v in o.bbox))
                                          for o in p.observations])
        for p in identity.players if p.observations
    ]
    hits = find_hits(flights(found, height), players, result["rallies"], height)
    for rally in result["rallies"]:
        rally["hits"] = [[h.t, h.player] for h in hits
                         if rally["start_s"] - RALLY_LEAD_S <= h.t <= rally["end_s"]]
    for player_id, stats in result["per_player"].items():
        stats["hits"] = sum(h.player == int(player_id) for h in hits)
