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


# ── Forehand and backhand, from the posture between the samples ─────────────
#
# At 15 poses a second the swing shows: the racket wrist, relative to the
# hips, peaks in speed as the ball is hit. Which side of the body the wrist
# comes from — the backswing, the sample before the peak — says forehand or
# backhand, once the racket hand is known: a right-hander's forehand comes
# from his right. The body's right is the picture's right for a player seen
# from behind (the near pair, mostly) and its left for one facing the
# camera: the shoulders tell which, the half of the court when they do not.
#
# A first version, to be checked in the video (StrokesCard, Verifica):
# SWING_MIN_TORSO_S is a guess from the sizes of a swing (the wrist moves
# about a metre in a tenth of a second, a torso is half a metre), not yet a
# measurement.

SWING_MIN_TORSO_S = 6.0
# Two swings of one player closer than this are one.
SAME_SWING_S = 0.7
# Poses further apart than this are not one movement.
MAX_STEP_S = 0.15
VOLLEY_MAX_NET_M = 3.5
# The shoulders tell which way the player faces when this far apart.
FACING_MIN_TORSO = 0.15
# A pose between the samples belongs to the player its track was, at the
# sample within this time.
OWNER_WINDOW_S = 0.3


def pose_series(players: list[PlayerTrack], tracklets, dense) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per player id: (times, keypoints n x 17 x 3), the samples' poses and
    those between them, in time order. A pose between the samples is the
    player's whose observation of the same track is nearest in time."""
    from app.ml.tracking import observation_key

    owner = {observation_key(o): p.player_id for p in players for o in p.observations}
    by_track = {}
    for tracklet in tracklets:
        times = np.array([o.timestamp_s for o in tracklet.observations], dtype=np.float64)
        by_track[tracklet.id] = (times, [observation_key(o) for o in tracklet.observations])
    series: dict[int, list[tuple[float, np.ndarray]]] = {p.player_id: [] for p in players}
    for player in players:
        series[player.player_id] += [(o.timestamp_s, o.keypoints) for o in player.observations
                                     if o.keypoints is not None]
    for t, track_id, points in dense or ():
        track = by_track.get(track_id)
        if track is None or not len(track[0]):
            continue
        i = int(np.argmin(np.abs(track[0] - t)))
        if abs(track[0][i] - t) > OWNER_WINDOW_S:
            continue
        pid = owner.get(track[1][i])
        if pid is not None:
            series[pid].append((float(t), points))
    out = {}
    for pid, items in series.items():
        items.sort(key=lambda x: x[0])
        out[pid] = (np.array([t for t, _ in items], dtype=np.float64),
                    np.array([k for _, k in items], dtype=np.float32).reshape(-1, 17, 3))
    return out


def detect_swings(series: dict[int, tuple[np.ndarray, np.ndarray]], paths: dict[int, SmoothedPath],
                  rallies) -> dict[int, list[dict]]:
    """Per player id, the swings found: {"t", "net_m", "L"/"R": {"v": the
    wrist's top speed (torsos a second), "side": where it came from, torsos
    to the player's right}} — both wrists, the racket hand chooses later."""
    out: dict[int, list[dict]] = {}
    for pid, (times, points) in series.items():
        path = paths.get(pid)
        body = [_body(points[i], _court_y(path, times[i])) for i in range(len(times))]
        swings: list[dict] = []
        for i in range(1, len(times)):
            a, b = body[i - 1], body[i]
            dt = times[i] - times[i - 1]
            if a is None or b is None or not 0 < dt <= MAX_STEP_S or a["over"] or b["over"]:
                continue
            speeds = {side: float(np.hypot(*(b[side] - a[side]))) / dt
                      for side in ("L", "R") if a[side] is not None and b[side] is not None}
            if not speeds or max(speeds.values()) < SWING_MIN_TORSO_S:
                continue
            t = float(times[i])
            if not _in_rally(t, rallies):
                continue
            entry = {side: {"v": round(v, 1), "side": round(float(a[side][0]), 2)} for side, v in speeds.items()}
            last = swings[-1] if swings else None
            if last is not None and t - last["t"] <= SAME_SWING_S:
                # The same swing: keep each wrist's fastest moment.
                for side, value in entry.items():
                    if side not in last or value["v"] > last[side]["v"]:
                        last[side] = value
                continue
            swings.append({"t": round(t, 2), **_where(path, t), **entry})
        for swing in swings:
            swing.pop("speed", None)
        out[pid] = swings
    return out


def ground_for_hand(swings: list[dict], hand: str | None) -> list[dict]:
    """[{"t", "kind": "forehand" | "backhand", "volley"}] with the racket in
    `hand`; nothing without it, the side of the body meaning nothing then."""
    if hand not in ("L", "R"):
        return []
    out = []
    for swing in swings:
        wrist = swing.get(hand)
        if wrist is None or wrist["v"] < SWING_MIN_TORSO_S:
            continue
        on_racket_side = wrist["side"] > 0 if hand == "R" else wrist["side"] < 0
        out.append({"t": swing["t"], "kind": "forehand" if on_racket_side else "backhand",
                    "volley": swing["net_m"] <= VOLLEY_MAX_NET_M})
    return out


def _body(points, court_y: float | None) -> dict | None:
    """The wrists relative to the hips' centre, in torsos, x to the player's
    right and y up; "over": a wrist above the shoulders (a high stroke)."""
    heights = _wrist_heights(points)
    if heights["L"] is None and heights["R"] is None:
        visible = points[:, 2] >= MIN_VISIBLE
        if not all(visible[i] for i in SHOULDERS + HIPS):
            return None
    shoulders = points[list(SHOULDERS), :2].mean(axis=0)
    hips = points[list(HIPS), :2].mean(axis=0)
    torso = float(hips[1] - shoulders[1])
    if torso <= 4.0:
        return None
    across = float(points[6, 0] - points[5, 0])          # right shoulder minus left, in the picture
    if abs(across) >= FACING_MIN_TORSO * torso:
        right = 1.0 if across > 0 else -1.0               # > 0: seen from behind
    else:
        right = 1.0 if court_y is None or court_y < NET_Y_M else -1.0
    visible = points[:, 2] >= MIN_VISIBLE
    out = {"over": any(h is not None and h > HIGH_WRIST_TORSO for h in heights.values())}
    for side, i in (("L", LEFT_WRIST), ("R", RIGHT_WRIST)):
        out[side] = (np.array([(points[i, 0] - hips[0]) * right, hips[1] - points[i, 1]]) / torso
                     if visible[i] else None)
    return out


def _court_y(path: SmoothedPath | None, t: float) -> float | None:
    if path is None or not len(path.times):
        return None
    return float(path.positions[int(np.argmin(np.abs(path.times - t))), 1])
