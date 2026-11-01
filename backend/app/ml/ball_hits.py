"""The hits of a rally, and who made them, from the ball and the players.

A hit is not where the ball's path breaks near a player: with the camera low
behind one pair, the near players cover half the picture and the ball
passes "on" them in nearly every flight; the glass and the floor turn the
ball as sharply as a racket does; and at the stroke the ball is often
behind the body. Rules on those failed on a real match (the hits found
alternated between the pairs half of the time, by chance).

So the hits are learnt from hits a person marked in the video ("Segna i
colpi", training/hits/train_hits.py), in three models stored in
hit_model.json:

1. per ball point, how likely a hit happened there: how the ball's
   velocity turns (before and after, over HIT_WINDOW_S), the gaps around it
   (hidden at the stroke), where it is against the vanishing point of the
   court, and how near the nearest player of each half is to it;
2. for a likely hit, which half hit it;
3. within that half, which of the two partners.

The candidates of a rally are then read as a sequence that alternates
between the halves — the ball crosses the net between two hits — at least
MIN_GAP_S apart (decode). On the match it was trained on, measured on
rallies left out of training: 81% of the hits found, 90% of those found
real, the half right 97% of the time and the partner 91%. A far hit is
found up to 0.45 s late: the ball turns visibly a few frames after it.

All image measures are fractions of the picture's height.
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

HIT_WINDOW_S = 0.2
MIN_GAP_S = 0.5
CANDIDATE_P = 0.08
CANDIDATE_NMS_S = 0.3
HIT_COST = 0.1
WHO_BEFORE_S = 0.5
WHO_AFTER_S = 0.1
RALLY_LEAD_S = 1.0
MODEL_PATH = Path(__file__).with_name("hit_model.json")
NO_PLAYER = [9.0, 9.0, 9.0, 0.0]
N_FEATURES = 23


@dataclass(frozen=True)
class Hit:
    t: float
    player: int
    far: bool


class PlayerTrack:
    """A player's boxes and half of the court in time."""

    def __init__(self, player_id: int, samples: list[tuple[float, tuple, bool]], team: int = 0) -> None:
        samples = sorted(samples, key=lambda s: s[0])
        self.player_id = player_id
        self.team = team
        self.times = [s[0] for s in samples]
        self.boxes = [s[1] for s in samples]
        self.far_half = [s[2] for s in samples]

    @classmethod
    def from_player(cls, player) -> "PlayerTrack":
        """From an identity.PlayerTrack: far half when beyond the net (Y > 10 m)."""
        return cls(player.player_id, [
            (float(o.timestamp_s), tuple(float(v) for v in o.bbox), float(o.foot_court[1]) > 10.0)
            for o in player.observations
        ], int(player.team))

    def box(self, t: float, max_gap: float = 2.0):
        i = bisect.bisect_right(self.times, t) - 1
        if i < 0:
            return self.boxes[0] if self.times and self.times[0] - t <= max_gap / 2 else None
        if i + 1 < len(self.times) and self.times[i + 1] - self.times[i] <= max_gap:
            f = (t - self.times[i]) / (self.times[i + 1] - self.times[i])
            a, b = self.boxes[i], self.boxes[i + 1]
            return tuple(a[k] + (b[k] - a[k]) * f for k in range(4))
        return self.boxes[i] if t - self.times[i] <= max_gap / 2 else None

    def far(self, t: float) -> bool:
        i = max(0, bisect.bisect_right(self.times, t) - 1)
        return self.far_half[i] if self.far_half else False


def _placement(box, x: float, y: float, height: int) -> list[float]:
    """Where the ball is against a player's box: distance (box heights, the
    width counting half), offsets from the box's centre and top, box size."""
    w, h = box[2] - box[0], box[3] - box[1]
    dx = (x - (box[0] + box[2]) / 2) / max(w, 1.0)
    dy = (y - box[1]) / max(h, 1.0)
    return [math.hypot(dx * 0.5, dy), dx, dy, h / height]


def _nearest(players: list[PlayerTrack], t: float, x: float, y: float, far: bool, height: int) -> list[float]:
    best = None
    for p in players:
        box = p.box(t)
        if box is None or p.far(t) != far:
            continue
        f = _placement(box, x, y, height)
        if best is None or f[0] < best[0]:
            best = f
    return best if best is not None else list(NO_PLAYER)


def _slope(t: np.ndarray, v: np.ndarray) -> float:
    tc = t - t.mean()
    return float((tc * (v - v.mean())).sum() / max((tc * tc).sum(), 1e-12))


def point_features(T: np.ndarray, X: np.ndarray, Y: np.ndarray, indices, players: list[PlayerTrack],
                   vp: tuple[float, float], height: int) -> np.ndarray:
    """The features of the ball points `indices` (T, X, Y: every point of
    the match's flights, in time order)."""
    rows = []
    for i in indices:
        t = T[i]
        lo = bisect.bisect_left(T, t - HIT_WINDOW_S)
        hi = bisect.bisect_right(T, t + HIT_WINDOW_S)
        ok = i - lo >= 2 and hi - i - 1 >= 2
        vb = np.array([_slope(T[lo:i], X[lo:i]), _slope(T[lo:i], Y[lo:i])]) if i - lo >= 2 else np.zeros(2)
        va = (np.array([_slope(T[i + 1:hi], X[i + 1:hi]), _slope(T[i + 1:hi], Y[i + 1:hi])])
              if hi - i - 1 >= 2 else np.zeros(2))
        nb, na = float(np.linalg.norm(vb)), float(np.linalg.norm(va))
        sb, sa = nb / height, na / height
        gp = t - T[i - 1] if i > 0 else 1.0
        gn = T[i + 1] - t if i + 1 < len(T) else 1.0
        rows.append([
            float(ok), sb, sa, math.log((sa + .02) / (sb + .02)), float(vb @ va / (nb * na + 1e-6)),
            float(np.sign(vb[0]) != np.sign(va[0])), float(np.sign(vb[1]) != np.sign(va[1])),
            (va[1] - vb[1]) / height, (va[0] - vb[0]) / height, min(gp, 1.0), min(gn, 1.0),
            float(i - lo), float(hi - i - 1), (Y[i] - vp[1]) / height, (X[i] - vp[0]) / height,
            *_nearest(players, t, X[i], Y[i], False, height),
            *_nearest(players, t, X[i], Y[i], True, height),
        ])
    return np.asarray(rows, dtype=np.float64).reshape(-1, N_FEATURES)


def partner_features(T, X, Y, a: PlayerTrack, b: PlayerTrack, t: float, height: int) -> list[float]:
    """Partner a against partner b at a hit: each one's nearest approach of
    the ball around it (WHO_BEFORE_S before to WHO_AFTER_S after)."""
    lo = bisect.bisect_left(T, t - WHO_BEFORE_S)
    hi = bisect.bisect_right(T, t + WHO_AFTER_S)
    idx = range(lo, hi) if hi > lo else [int(np.argmin(np.abs(T - t)))]

    def one(p: PlayerTrack) -> list[float]:
        best = None
        for i in idx:
            box = p.box(T[i])
            if box is None:
                continue
            f = _placement(box, X[i], Y[i], height) + [(X[i] - (box[0] + box[2]) / 2) / height]
            if best is None or f[0] < best[0]:
                best = f
        return best if best is not None else [9.0, 9.0, 9.0, 0.0, 9.0]

    fa, fb = one(a), one(b)
    return fa + fb + [fa[0] - fb[0], abs(fa[4]) - abs(fb[4])]


# ── The stored models ───────────────────────────────────────────────────────

class Trees:
    """A gradient-boosted classifier exported from scikit-learn: the
    probability of the positive class."""

    def __init__(self, data: dict) -> None:
        self.init = float(data["init"])
        self.rate = float(data["rate"])
        self.trees = [{k: np.asarray(v) for k, v in tree.items()} for tree in data["trees"]]

    def proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)          # as scikit-learn compares
        rows = np.arange(len(X))
        raw = np.full(len(X), self.init)
        for tree in self.trees:
            left, right, feature = tree["left"], tree["right"], tree["feature"]
            threshold, value = tree["threshold"], tree["value"]
            node = np.zeros(len(X), dtype=np.int64)
            inner = left[node] != -1
            while inner.any():                         # all rows down one level at a time
                go_left = X[rows, np.maximum(feature[node], 0)] <= threshold[node]
                node = np.where(inner, np.where(go_left, left[node], right[node]), node)
                inner = left[node] != -1
            raw += self.rate * value[node]
        return 1.0 / (1.0 + np.exp(-raw))


class Linear:
    def __init__(self, data: dict) -> None:
        self.coef = np.asarray(data["coef"], dtype=np.float64)
        self.intercept = float(data["intercept"])

    def proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(np.asarray(X, dtype=np.float64) @ self.coef + self.intercept)))


@dataclass
class HitModel:
    hit: Trees
    far: Trees
    partner: Linear
    measured: dict


@lru_cache(maxsize=2)
def load_model(path: str = str(MODEL_PATH)) -> HitModel | None:
    file = Path(path)
    if not file.exists():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    return HitModel(Trees(data["hit"]), Trees(data["far"]), Linear(data["partner"]), data.get("measured", {}))


# ── Reading the hits ────────────────────────────────────────────────────────

def candidates(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> list[tuple[float, float, float]]:
    """The likeliest points, one per CANDIDATE_NMS_S: (time, p hit, p far)."""
    keep: list[int] = []
    for i in np.argsort(-p):
        if p[i] < CANDIDATE_P:
            break
        if all(abs(t[i] - t[k]) > CANDIDATE_NMS_S for k in keep):
            keep.append(int(i))
    keep.sort()
    return [(float(t[i]), float(p[i]), float(q[i])) for i in keep]


def decode(cands: list[tuple[float, float, float]], cost: float = HIT_COST,
           gap: float = MIN_GAP_S) -> list[tuple[float, bool]]:
    """The best sequence of hits alternating between the halves: each
    candidate counts its probability of being a hit by that half, less
    `cost`. (time, far half) per hit."""
    best: dict[tuple[int, int], tuple[float, tuple[int, int] | None]] = {}
    for i, (t, p, q) in enumerate(cands):
        for far in (0, 1):
            gain = p * (q if far else 1.0 - q) - cost
            score, prev = gain, None
            for j in range(i):
                if t - cands[j][0] < gap:
                    continue
                before = best.get((j, 1 - far))
                if before is not None and before[0] + gain > score:
                    score, prev = before[0] + gain, (j, 1 - far)
            best[(i, far)] = (score, prev)
    if not best:
        return []
    key = max(best, key=lambda k: best[k][0])
    if best[key][0] <= 0:
        return []
    out = []
    while key is not None:
        out.append((cands[key[0]][0], bool(key[1])))
        key = best[key][1]
    return out[::-1]


def find_hits(T, X, Y, players: list[PlayerTrack], rallies: list[dict], vp, height: int,
              model: HitModel) -> list[Hit]:
    hits: list[Hit] = []
    for rally in rallies:
        lo = bisect.bisect_left(T, rally["start_s"] - RALLY_LEAD_S)
        hi = bisect.bisect_right(T, rally["end_s"])
        if hi - lo < 3:
            continue
        idx = list(range(lo, hi))
        F = point_features(T, X, Y, idx, players, vp, height)
        p, q = model.hit.proba(F), model.far.proba(F)
        for t, far in decode(candidates(p, q, T[lo:hi])):
            side = pair_at(players, t, far)
            if not side:
                continue
            if len(side) == 2:
                a, b = sorted(side, key=lambda pl: pl.player_id)
                first = model.partner.proba(np.array([partner_features(T, X, Y, a, b, t, height)]))[0]
                who = a if first >= 0.5 else b
            else:
                who = side[0]
            hits.append(Hit(round(t, 2), who.player_id, far))
    return hits


def pair_at(players: list[PlayerTrack], t: float, far: bool) -> list[PlayerTrack]:
    """The two partners on that half at `t`, by player id: the pair of
    whoever is seen there, the partner too when not seen at that moment."""
    seen = [p for p in players if p.box(t) is not None and p.far(t) == far]
    if not seen:
        return []
    team = seen[0].team
    return sorted([p for p in players if p.team == team], key=lambda p: p.player_id)


def ball_points(found, height: int):
    """Every point of the ball's flights, in time order: T, X, Y."""
    from app.ml.ball_track import flights

    pts = sorted((t, x, y) for f in flights(found, height) for t, x, y in zip(f.times, f.xs, f.ys))
    return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), np.array([p[2] for p in pts]))


def vanishing_point(calibration) -> tuple[float, float]:
    """Where the lines along the court meet in the picture."""
    v = np.asarray(calibration.homography_inv, dtype=np.float64) @ np.array([0.0, 1.0, 0.0])
    if abs(v[2]) < 1e-9:
        return (calibration.frame_size[0] / 2, 0.0)
    return (float(v[0] / v[2]), float(v[1] / v[2]))


# ── Rallies and hits for the statistics ─────────────────────────────────────

# Rallies found from the players' movement are where the ball is looked for,
# this far around them; a rally is then from its first hit (the serve) to its
# last, plus TAIL_S for the ball's last flight. Players keep moving for
# seconds after a point: from movement alone a rally ended 8 s after its
# last hit (median, on the marked match), from the hits 0 s.
EXTEND_S = 3.0
# Inside one rally found from movement, the ball not hit for this long is a
# point over and a new one: two rallies.
SPLIT_GAP_S = 8.0
# Two rallies whose hits follow on within this are one point that the
# movement split (everyone still for a moment).
MERGE_GAP_S = 3.0
TAIL_S = 1.0
# A first hit from this far behind the net is a serve.
SERVE_BACK_M = 6.0


@dataclass
class BallInput:
    T: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    players: list[PlayerTrack]
    vp: tuple[float, float]
    height: int
    model: HitModel


def ball_input(directory, calibration, identity, height: int) -> BallInput | None:
    """What the hits are read from, when the run followed the ball and a
    hit model is there; None otherwise."""
    from app.ml.artifacts import read_ball

    model = load_model()
    if model is None or directory is None or not height:
        return None
    found = read_ball(directory)
    if not found:
        return None
    T, X, Y = ball_points(found, height)
    if len(T) < 3:
        return None
    players = [PlayerTrack.from_player(p) for p in identity.players if p.observations]
    return BallInput(T, X, Y, players, vanishing_point(calibration), height, model)


def refine_rallies(rallies: list, ball: BallInput, net_distance=None) -> tuple[list, list[Hit]]:
    """The rallies from movement made to start at the serve and end at the
    last hit, split and merged where the hits say so; and the hits in them.
    Rallies where the ball was not seen hit stay as movement found them.
    `net_distance(player id, t)`: metres from the net, to tell a serve."""
    from app.ml.rallies import Rally

    zones: list[list[float]] = []
    for r in rallies:
        a, b = r.start_s - EXTEND_S, r.end_s + EXTEND_S
        if zones and a <= zones[-1][1]:
            zones[-1][1] = max(zones[-1][1], b)
        else:
            zones.append([a, b])
    found = find_hits(ball.T, ball.X, ball.Y, ball.players,
                      [{"start_s": a + RALLY_LEAD_S, "end_s": b} for a, b in zones],
                      ball.vp, ball.height, ball.model)
    hits = sorted({h.t: h for h in found}.values(), key=lambda h: h.t)

    spans: list[list] = []         # [start, end, hits or None, source rally]
    used = 0
    for r in rallies:
        inside = []
        while used < len(hits) and hits[used].t < r.start_s - EXTEND_S:
            used += 1
        k = used
        while k < len(hits) and hits[k].t <= r.end_s:
            inside.append(hits[k])
            k += 1
        if len(inside) < 2:
            spans.append([r.start_s, r.end_s, None, r])
            continue
        used = k
        parts = [[inside[0]]]
        for h in inside[1:]:
            if h.t - parts[-1][-1].t > SPLIT_GAP_S:
                parts.append([h])
            else:
                parts[-1].append(h)
        for part in parts:
            if len(part) >= 2:
                spans.append([part[0].t, part[-1].t + TAIL_S, part, r])
    spans.sort(key=lambda x: x[0])
    merged: list[list] = []
    for span in spans:
        last = merged[-1] if merged else None
        if (last is not None and last[2] is not None and span[2] is not None
                and span[0] - last[2][-1].t <= MERGE_GAP_S):
            last[1] = max(last[1], span[1])
            last[2] = last[2] + span[2]
        else:
            merged.append(span)

    out, kept = [], []
    for i, (start, end, part, source) in enumerate(merged):
        serve = source.serve
        if part is not None:
            kept += part
            first = part[0]
            if net_distance is not None:
                d = net_distance(first.player, first.t)
                serve = serve or (d is not None and d >= SERVE_BACK_M)
        out.append(Rally(index=i, start_s=float(start), end_s=float(end),
                         players_tracked=source.players_tracked, serve=bool(serve)))
    return out, kept


def hit_zone(distance_m: float) -> str:
    from app.ml.metrics import ZONE_MID_M, ZONE_NET_M

    if distance_m <= ZONE_NET_M:
        return "net"
    return "mid" if distance_m <= ZONE_MID_M else "back"


def hit_statistics(result: dict, hits: list[Hit], net_distance, position, measured: dict | None) -> None:
    """Into the result: per rally its hits [time, player]; per player how
    many, from where (net, mid, back: as zone_pct), the court positions they
    hit from, and how many rallies their hit closed."""
    for rally in result["rallies"]:
        rally["hits"] = [[h.t, h.player] for h in hits
                         if rally["start_s"] - RALLY_LEAD_S <= h.t <= rally["end_s"]]
    closing: dict[int, int] = {}
    for rally in result["rallies"]:
        if rally["hits"]:
            closing[rally["hits"][-1][1]] = closing.get(rally["hits"][-1][1], 0) + 1
    for player_id, stats in result["per_player"].items():
        pid = int(player_id)
        mine = [h for h in hits if h.player == pid]
        zones = {"net": 0, "mid": 0, "back": 0}
        spots = []
        for h in mine:
            d = net_distance(pid, h.t)
            if d is not None:
                zones[hit_zone(d)] += 1
            xy = position(pid, h.t)
            if xy is not None:
                spots.append([round(xy[0], 1), round(xy[1], 1)])
        stats["hits"] = len(mine)
        stats["hit_zones"] = zones
        stats["hit_spots"] = spots
        stats["closing_hits"] = closing.get(pid, 0)
    if measured:
        result["data_quality"].setdefault("ball", {})["hits_measured"] = measured
