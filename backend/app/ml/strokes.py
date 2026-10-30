"""High strokes from the players' posture: smash and bandeja, and lob.

What the posture can tell at 5 samples a second. A stroke lasts about 0.3
s, one or two samples: the swing of a forehand or a backhand falls between
them (the wrist moved as much between points as in a rally). A raised
racket does not: the wrist of the racket hand above the shoulders by
HIGH_WRIST_TORSO of the torso's length is seen, and checked on a real set
against the video, it was a stroke 22 times out of 24.

Two strokes end with the racket up: the smash or bandeja, hit up there,
and the lob, hit from below and finishing high. Where the player stands
tells them apart: all eleven smashes and bandejas checked were within
6.2 m of the net, six lobs out of eight beyond 7.3 m. Within
OVERHEAD_MAX_NET_M, a smash or bandeja; beyond, a lob — about two in
three right on that set, said so on screen.

Each stroke keeps which wrist was up ("L", "R"): the racket hand, once the
players are named with it, keeps its own and leaves out the free arm
raised to point at the ball.
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
# Strokes count around a rally, not while players fool about between points.
RALLY_MARGIN_S = 0.5
NET_Y_M = 10.0


def detect_strokes(players: list[PlayerTrack], paths: dict[int, SmoothedPath], rallies) -> dict[int, list[dict]]:
    """Per player id: [{"t", "kind": "overhead" | "lob", "wrists": "L"/"R"/"LR"}]."""
    out: dict[int, list[dict]] = {}
    for player in players:
        path = paths.get(player.player_id)
        strokes: list[dict] = []
        current = None
        for obs in player.observations:
            up = _raised_wrists(obs.keypoints)
            if not up:
                continue
            t = float(obs.timestamp_s)
            if current is not None and t - current["last"] <= SAME_STROKE_S:
                current["last"] = t
                current["wrists"] |= up
                continue
            if not _in_rally(t, rallies):
                current = None
                continue
            current = {"t": t, "last": t, "wrists": set(up)}
            strokes.append(current)
        out[player.player_id] = [
            {"t": round(s["t"], 2), "kind": _kind(path, s["t"]), "wrists": "".join(sorted(s["wrists"]))}
            for s in strokes
        ]
    return out


def for_hand(strokes: list[dict], hand: str | None) -> list[dict]:
    """The strokes of the racket hand, `hand` "L"/"R"; all without it."""
    if hand not in ("L", "R"):
        return list(strokes)
    return [s for s in strokes if hand in s.get("wrists", "")]


def _raised_wrists(points) -> set[str]:
    if points is None:
        return set()
    visible = points[:, 2] >= MIN_VISIBLE
    if not all(visible[i] for i in SHOULDERS + HIPS):
        return set()
    shoulders = float(np.mean(points[list(SHOULDERS), 1]))
    torso = float(np.mean(points[list(HIPS), 1])) - shoulders
    if torso <= 4.0:                      # lying, bent double, or a bad pose
        return set()
    line = shoulders - HIGH_WRIST_TORSO * torso
    return {side for side, i in (("L", LEFT_WRIST), ("R", RIGHT_WRIST))
            if visible[i] and points[i, 1] < line}


def _in_rally(t: float, rallies) -> bool:
    return any(r.start_s - RALLY_MARGIN_S <= t <= r.end_s + RALLY_MARGIN_S for r in rallies)


def _kind(path: SmoothedPath | None, t: float) -> str:
    if path is None or not len(path.times):
        return "overhead"
    i = int(np.argmin(np.abs(path.times - t)))
    return "overhead" if abs(float(path.positions[i, 1]) - NET_Y_M) <= OVERHEAD_MAX_NET_M else "lob"
