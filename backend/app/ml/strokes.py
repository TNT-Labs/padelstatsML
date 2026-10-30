"""High strokes from the players' posture: smash and bandeja, and lob.

What the posture can tell at 5 samples a second. A stroke lasts about 0.3
s, one or two samples: the swing of a forehand or a backhand falls between
them (the wrist moved as much between points as in a rally). A raised
racket does not: the wrist above the shoulders by HIGH_WRIST_TORSO of the
torso's length marks a candidate.

Tuned on the third set of 22/10, 58 candidates checked one by one in the
video (smash 21, bandeja 13, lob 8, other strokes 7, nothing 9):

- smash and bandeja: the racket wrist up, within OVERHEAD_MAX_NET_M of the
  net, *and the free arm raised too* (FREE_ARM_MIN), as it is to point at
  the ball and balance the swing: 29 of the 30 so called were, 29 of the
  34 there were; nothing and other strokes had the free arm low, every
  one. The result holds from -0.2 to -0.35 and from 6 to 7 m: a gesture,
  not thresholds fitted to the set.
- lob: the racket wrist up further back, the player nearly still
  (LOB_MAX_SPEED_MS): 7 of the 8 so called, on few cases.
- anything else is left out.

The racket hand is known only once the players are named with it, after
the analysis: each candidate keeps what both wrists did, and classify()
decides with the hand when the statistics are read.
"""
from __future__ import annotations

import numpy as np

from app.ml.identity import PlayerTrack
from app.ml.smoothing import SmoothedPath

LEFT_WRIST, RIGHT_WRIST = 9, 10
SHOULDERS, HIPS = (5, 6), (11, 12)
MIN_VISIBLE = 0.5
HIGH_WRIST_TORSO = 0.4
# Samples this close belong to the same stroke.
SAME_STROKE_S = 0.8
OVERHEAD_MAX_NET_M = 6.5
# The free arm's height, torsos above the shoulders, at least.
FREE_ARM_MIN = -0.3
LOB_MAX_SPEED_MS = 2.0
SPEED_WINDOW_S = 0.6
# Strokes count around a rally, not while players fool about between points.
RALLY_MARGIN_S = 0.5
NET_Y_M = 10.0


def detect_strokes(players: list[PlayerTrack], paths: dict[int, SmoothedPath], rallies) -> dict[int, list[dict]]:
    """Per player id, the candidates: {"t", "net_m", "speed", "up": wrists
    raised ("L", "R", "LR"), "high": {"L", "R"} each wrist at its highest
    over the stroke, torsos above the shoulders (None unseen)}."""
    out: dict[int, list[dict]] = {}
    for player in players:
        path = paths.get(player.player_id)
        strokes: list[dict] = []
        current = None
        for obs in player.observations:
            heights = _wrist_heights(obs.keypoints)
            up = {side for side, h in heights.items() if h is not None and h > HIGH_WRIST_TORSO}
            if not up:
                continue
            t = float(obs.timestamp_s)
            if current is None or t - current["last"] > SAME_STROKE_S:
                if not _in_rally(t, rallies):
                    current = None
                    continue
                current = {"t": round(t, 2), "last": t, **_where(path, t)}
                strokes.append(current)
            current["last"] = t
            current["up"] = "".join(sorted(set(current.get("up", "")) | up))
            # Each wrist at its highest over the stroke: the free arm often
            # rises first, to point at the ball, the racket the sample after.
            high = current.setdefault("high", {"L": None, "R": None})
            for side, h in heights.items():
                if h is not None and (high[side] is None or h > high[side]):
                    high[side] = round(h, 2)
        for stroke in strokes:
            del stroke["last"]
        out[player.player_id] = strokes
    return out


def classify(stroke: dict, hand: str | None) -> str | None:
    """"overhead" (smash, bandeja), "lob" or None for a candidate, with the
    racket in `hand` ("L"/"R"; either when unknown)."""
    sides = [hand] if hand in ("L", "R") else ["R", "L"]
    kinds = []
    for side in sides:
        if side not in stroke["up"]:
            continue
        if stroke["net_m"] <= OVERHEAD_MAX_NET_M:
            free = stroke["high"]["R" if side == "L" else "L"]
            kinds.append("overhead" if free is not None and free >= FREE_ARM_MIN else None)
        else:
            kinds.append("lob" if stroke["speed"] <= LOB_MAX_SPEED_MS else None)
    return next((k for k in kinds if k), None)


def for_hand(strokes: list[dict], hand: str | None) -> list[dict]:
    """The strokes kept with the racket in `hand`: [{"t", "kind"}]. Results
    stored before the free arm was looked at carry their kind and the
    wrists raised: kept as they were, the racket hand's only."""
    out = []
    for stroke in strokes:
        if "net_m" not in stroke:
            if hand not in ("L", "R") or hand in stroke.get("wrists", ""):
                out.append({"t": stroke["t"], "kind": stroke["kind"]})
            continue
        kind = classify(stroke, hand)
        if kind:
            out.append({"t": stroke["t"], "kind": kind})
    return out


def _wrist_heights(points) -> dict[str, float | None]:
    """Each wrist's height above the shoulders, in torso lengths; None when
    it, the shoulders or the hips are not seen."""
    none = {"L": None, "R": None}
    if points is None:
        return none
    visible = points[:, 2] >= MIN_VISIBLE
    if not all(visible[i] for i in SHOULDERS + HIPS):
        return none
    shoulders = float(np.mean(points[list(SHOULDERS), 1]))
    torso = float(np.mean(points[list(HIPS), 1])) - shoulders
    if torso <= 4.0:                      # lying, bent double, or a bad pose
        return none
    return {side: (shoulders - float(points[i, 1])) / torso if visible[i] else None
            for side, i in (("L", LEFT_WRIST), ("R", RIGHT_WRIST))}


def _in_rally(t: float, rallies) -> bool:
    return any(r.start_s - RALLY_MARGIN_S <= t <= r.end_s + RALLY_MARGIN_S for r in rallies)


def _where(path: SmoothedPath | None, t: float) -> dict:
    """Distance from the net and top speed around `t`."""
    if path is None or not len(path.times):
        return {"net_m": 0.0, "speed": 0.0}
    i = int(np.argmin(np.abs(path.times - t)))
    near = np.abs(path.times - t) <= SPEED_WINDOW_S
    return {"net_m": round(abs(float(path.positions[i, 1]) - NET_Y_M), 2),
            "speed": round(float(path.speeds[near].max()), 2)}
