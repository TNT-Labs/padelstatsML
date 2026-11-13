"""The hits found, read as a serve, a hit in play or a hit between points:
the points from them.

The rallies from the players' movement join points: between two points the
players walk to their places for 7-13 s, lively enough to look in play, and
the ball is hit there too — passed back to the server, bounced before the
serve — which the hit model takes for hits (it learnt from the marked hits,
a second around each: pauses it barely saw). On three sets of 22/10 with
every point marked ("Segna i punti", training/points), 23 of 89 points were
joined to the one before or after.

So each hit found is read in its context by two small models
(phase_model.json, training/points/train_phases.py), from what the players
do around it and the hits around it (phase_features):

1. pause: the hit is between points;
2. serve: of the others, the hit opens a point.

The points are then the runs of hits that are not pauses, a new one at each
serve or after a gap of more than `gap_s`; a run of one hit is a point only
when a serve is sure (`lone_serve`): an ace, or a return that never came.
Measured on each set with models and thresholds chosen on the other two
(and hits found by hit models that never saw it): 69 of 89 points right one
to one against 56 before, joined to another 11 against 24, rallies that are
no point 14 against 11, lost 7 and split 2 as before. The hits read as
between points are left out of the hits' statistics too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

PHASE_MODEL_PATH = Path(__file__).with_name("phase_model.json")
N_PHASE_FEATURES = 16
# Players' speeds read this long before and after the hit (s), the hitter's
# own just before it, and how far from the net "at the back" is.
SPEED_BEFORE = (2.5, 0.3)
SPEED_AFTER = (0.3, 2.5)
HITTER_BEFORE = (1.5, 0.2)
BACK_M = 6.5
COUNT_S = 6.0
CAP_GAP_S = 15.0
COURT_HALF_WIDTH_M = 5.0
TAIL_S = 1.0


@dataclass
class PhaseModel:
    pause: object                       # ball_hits.Trees
    serve: object
    serve_threshold: float = 0.6
    pause_threshold: float = 0.6
    gap_s: float = 8.0
    lone_serve: float = 0.9
    measured: dict = field(default_factory=dict)


@lru_cache(maxsize=2)
def load_phase_model(path: str = str(PHASE_MODEL_PATH)) -> PhaseModel | None:
    from app.ml.ball_hits import Trees

    file = Path(path)
    if not file.exists():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    if int(data.get("features", 0)) != N_PHASE_FEATURES:
        return None                     # made for other features: not to be read with these
    seg = data.get("segment", {})
    return PhaseModel(Trees(data["pause"]), Trees(data["serve"]),
                      float(seg.get("serve_threshold", 0.6)), float(seg.get("pause_threshold", 0.6)),
                      float(seg.get("gap_s", 8.0)), float(seg.get("lone_serve", 0.9)),
                      data.get("measured", {}))


def _speeds(paths: dict, a: float, b: float) -> tuple[float, float]:
    """Mean and highest of the players' mean speeds between a and b."""
    means = []
    for path in paths.values():
        if not len(path.times):
            continue
        inside = (path.times >= a) & (path.times <= b)
        if inside.any():
            means.append(float(np.mean(path.speeds[inside])))
    return (float(np.mean(means)), float(np.max(means))) if means else (0.0, 0.0)


def phase_features(hits: list, rallies: list, where) -> np.ndarray:
    """One row per hit (in time order): the time to the hits around it, the
    hitter's distance from the net, side and speed, how fast the players
    move before and after, how many stand at the back, whether the
    movement said play and since when, and how near a serve formation the
    movement saw. `rallies`: the rallies from movement; `where`: the
    players' filtered paths (.paths, .net_distance, .position)."""
    paths = where.paths
    T = np.array([h.t for h in hits], dtype=np.float64)
    serves = [r.start_s for r in rallies if r.serve]
    rows = []
    for i, h in enumerate(hits):
        t = h.t
        prev = t - T[i - 1] if i > 0 else CAP_GAP_S
        nxt = T[i + 1] - t if i + 1 < len(T) else CAP_GAP_S
        d = where.net_distance(h.player, t)
        b_mean, b_max = _speeds(paths, t - SPEED_BEFORE[0], t - SPEED_BEFORE[1])
        a_mean, a_max = _speeds(paths, t + SPEED_AFTER[0], t + SPEED_AFTER[1])
        back = sum(1 for pid in paths if (where.net_distance(pid, t) or 0.0) > BACK_M)
        inside = next((r for r in rallies if r.start_s - 1.0 <= t <= r.end_s + 1.0), None)
        since = t - inside.start_s if inside is not None else -5.0
        to_serve = min((t - s for s in serves), key=abs, default=20.0)
        hitter = paths.get(h.player)
        hitter_speed = 0.0
        if hitter is not None and len(hitter.times):
            m = (hitter.times >= t - HITTER_BEFORE[0]) & (hitter.times <= t - HITTER_BEFORE[1])
            hitter_speed = float(np.mean(hitter.speeds[m])) if m.any() else 0.0
        pos = where.position(h.player, t)
        rows.append([
            min(prev, CAP_GAP_S), min(nxt, CAP_GAP_S), d if d is not None else -1.0,
            b_mean, b_max, a_mean, a_max, float(back), float(h.far), float(inside is not None),
            min(since, 30.0), float(((T >= t - COUNT_S) & (T < t)).sum()),
            float(((T > t) & (T <= t + COUNT_S)).sum()), max(-20.0, min(20.0, to_serve)),
            hitter_speed, abs(pos[0] - COURT_HALF_WIDTH_M) if pos is not None else -1.0,
        ])
    return np.asarray(rows, dtype=np.float64).reshape(-1, N_PHASE_FEATURES)


def segment(hits: list, p_pause, p_serve, serve_threshold: float, pause_threshold: float,
            gap_s: float, lone_serve: float) -> list[tuple[float, float, list, bool]]:
    """The points from the hits read: (first hit, last hit, hits, opened by
    a serve) each, in time order."""
    out: list[tuple[float, float, list, bool]] = []
    run: list = []
    serve_p = 0.0

    def close() -> None:
        if run and (len(run) >= 2 or serve_p >= lone_serve):
            out.append((run[0].t, run[-1].t, list(run), serve_p >= serve_threshold))

    for h, pp, ps in zip(hits, p_pause, p_serve):
        if pp >= pause_threshold:
            close()
            run, serve_p = [], 0.0
            continue
        if not run or ps >= serve_threshold or h.t - run[-1].t > gap_s:
            close()
            run, serve_p = [h], float(ps)
        else:
            run.append(h)
    close()
    return out


def read_points(hits: list, rallies: list, where, model: PhaseModel) -> list[tuple[float, float, list, bool]]:
    """segment of the hits with the model's two readings and settings."""
    if not hits:
        return []
    F = phase_features(hits, rallies, where)
    return segment(hits, model.pause.proba(F), model.serve.proba(F), model.serve_threshold,
                   model.pause_threshold, model.gap_s, model.lone_serve)
