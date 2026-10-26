"""How far off the analysis is, measured against what a person sees.

Every number the system reports is in metres, and until now nothing said
how many metres it could be wrong by: the numbers always look plausible.
This compares the analysis with points a person marks by hand:

* Calibration. Court landmarks whose position is known exactly — where a
  service line meets a side line, the T, the foot of the net posts — are
  marked on the keyframe. Through the calibration each lands at some point
  of the court; the distance from where it really is, is the calibration's
  error at that spot. The four corners cannot check themselves, so these
  are other points, near and far.
* Players. In a few frames chosen across the match, the person marks the
  spot on the ground between each player's feet — where it is, also when
  the net hides it. Each mark is paired with the nearest player the
  analysis placed there; the distance between the two, through the same
  calibration, is the error of detection and foot point, on the raw
  detections and on the filtered paths the metrics use. Marks no player
  was placed near are players missed; players placed where nobody was
  marked are extra.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.ml.court import (
    COURT_CORNERS_M,
    DERIVED_MAX_OFF_FRAME,
    NET_Y_M,
    SERVICE_LINE_OFFSET_M,
    CalibrationError,
    CourtCalibration,
    build_calibration,
    distort_points,
    fit_corners,
    undistort_points,
)
from app.ml.identity import PlayerTrack
from app.ml.rallies import Rally
from app.ml.smoothing import MAX_BRIDGE_S, SmoothedPath

_NEAR_SERVICE = NET_Y_M - SERVICE_LINE_OFFSET_M
_FAR_SERVICE = NET_Y_M + SERVICE_LINE_OFFSET_M


@dataclass(frozen=True)
class Landmark:
    key: str
    label: str
    x_m: float
    y_m: float


# Left and right as seen from the camera; near and far likewise.
LANDMARKS: tuple[Landmark, ...] = (
    Landmark("servizio_vicino_sx", "Linea di servizio × laterale sinistra, metà vicina", 0.0, _NEAR_SERVICE),
    Landmark("servizio_vicino_t", "«T» centrale, metà vicina", 5.0, _NEAR_SERVICE),
    Landmark("servizio_vicino_dx", "Linea di servizio × laterale destra, metà vicina", 10.0, _NEAR_SERVICE),
    Landmark("rete_sx", "Base del palo della rete, a sinistra", 0.0, NET_Y_M),
    Landmark("rete_dx", "Base del palo della rete, a destra", 10.0, NET_Y_M),
    Landmark("servizio_lontano_sx", "Linea di servizio × laterale sinistra, metà lontana", 0.0, _FAR_SERVICE),
    Landmark("servizio_lontano_t", "«T» centrale, metà lontana", 5.0, _FAR_SERVICE),
    Landmark("servizio_lontano_dx", "Linea di servizio × laterale destra, metà lontana", 10.0, _FAR_SERVICE),
)
_BY_KEY = {lm.key: lm for lm in LANDMARKS}

# Frames offered for marking players, and how many of them inside rallies:
# players running are the hard case, players waiting the easy one.
FRAMES = 8
# A mark this far from every player placed by the analysis is a player
# the analysis missed, not an error of a few metres.
MATCH_MAX_M = 2.5


def _half(y_m: float) -> str:
    if abs(y_m - NET_Y_M) < 1e-6:
        return "rete"
    return "vicina" if y_m < NET_Y_M else "lontana"


def _summary(errors: list[float]) -> dict:
    if not errors:
        return {"count": 0}
    values = np.asarray(errors, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_m": round(float(values.mean()), 2),
        "median_m": round(float(np.median(values)), 2),
        "p90_m": round(float(np.percentile(values, 90)), 2),
        "max_m": round(float(values.max()), 2),
    }


def calibration_errors(calib: CourtCalibration, marks: dict[str, tuple[float, float]]) -> dict:
    """Error of the calibration at each marked landmark, in metres."""
    items = []
    for key, (u, v) in marks.items():
        landmark = _BY_KEY.get(key)
        if landmark is None:
            continue
        x, y = calib.floor_points_to_court(np.array([[u, v]], dtype=np.float32))[0]
        error = float(np.hypot(x - landmark.x_m, y - landmark.y_m))
        items.append({
            "key": key, "label": landmark.label, "half": _half(landmark.y_m),
            # Kept, so the calibration can be corrected from them later.
            "px": [round(float(u), 1), round(float(v), 1)],
            "measured_m": [round(float(x), 2), round(float(y), 2)],
            "error_m": round(error, 2),
        })
    by_half = {
        half: _summary([i["error_m"] for i in items if i["half"] == half])
        for half in ("vicina", "rete", "lontana")
    }
    return {"items": items, "all": _summary([i["error_m"] for i in items]), "by_half": by_half}


def choose_frames(players: list[PlayerTrack], rallies: list[Rally], count: int = FRAMES) -> list[dict]:
    """Frames to mark, spread over the match: as many as possible with at
    least three players placed, half of them during rallies."""
    seen: dict[int, list[float]] = {}
    for player in players:
        for obs in player.observations:
            seen.setdefault(obs.frame_index, []).append(obs.timestamp_s)
    crowded = sorted((index, times[0]) for index, times in seen.items() if len(times) >= 3)
    if not crowded:
        crowded = sorted((index, times[0]) for index, times in seen.items())
    if not crowded:
        return []

    def in_rally(t: float) -> bool:
        return any(r.contains(t) for r in rallies)

    playing = [c for c in crowded if in_rally(c[1])]
    waiting = [c for c in crowded if not in_rally(c[1])]
    wanted_playing = min(len(playing), (count + 1) // 2)
    picks = _spread(playing, wanted_playing) + _spread(waiting, count - wanted_playing)
    if len(picks) < count:
        rest = [c for c in crowded if c not in picks]
        picks += _spread(rest, count - len(picks))
    return [{"frame_index": int(i), "t": round(float(t), 3)} for i, t in sorted(picks)]


def _spread(items: list, count: int) -> list:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    positions = np.linspace(0, len(items) - 1, count + 2)[1:-1]
    return [items[int(round(p))] for p in positions]


def player_errors(
    calib: CourtCalibration,
    players: list[PlayerTrack],
    paths: dict[int, SmoothedPath],
    marked: list[tuple[int, list[tuple[float, float]]]],
) -> dict:
    """Errors of the players' positions against the marks, per frame
    [(frame_index, [(u, v), ...])]."""
    by_frame: dict[int, list[tuple[int, tuple[float, float], float]]] = {}
    for player in players:
        for obs in player.observations:
            by_frame.setdefault(obs.frame_index, []).append(
                (player.player_id, obs.foot_court, obs.timestamp_s))

    raw_errors: list[tuple[float, str]] = []
    smoothed_errors: list[tuple[float, str]] = []
    marks_total = missed = extra = 0
    for frame_index, points in marked:
        placed = by_frame.get(frame_index, [])
        marks_total += len(points)
        if not points:
            extra += len(placed)
            continue
        truth = calib.floor_points_to_court(np.array(points, dtype=np.float32).reshape(-1, 2))
        if not placed:
            missed += len(points)
            continue
        positions = np.array([p[1] for p in placed], dtype=np.float64)
        cost = np.linalg.norm(truth[:, None, :] - positions[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        matched = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= MATCH_MAX_M]
        missed += len(points) - len(matched)
        extra += len(placed) - len(matched)
        for r, c in matched:
            half = "vicina" if truth[r][1] < NET_Y_M else "lontana"
            raw_errors.append((float(cost[r, c]), half))
            player_id, _, t = placed[c]
            smoothed = _path_at(paths.get(player_id), t)
            if smoothed is not None:
                smoothed_errors.append((float(np.linalg.norm(truth[r] - smoothed)), half))

    def report(errors):
        return {
            "all": _summary([e for e, _ in errors]),
            "by_half": {h: _summary([e for e, half in errors if half == h]) for h in ("vicina", "lontana")},
        }

    return {
        "frames": len(marked),
        "marks": marks_total,
        "matched": len(raw_errors),
        "missed": missed,
        "extra": extra,
        "raw": report(raw_errors),
        "smoothed": report(smoothed_errors),
    }


def _path_at(path: SmoothedPath | None, t: float) -> np.ndarray | None:
    """The filtered position at time t, between two samples of one path."""
    if path is None or len(path.times) == 0:
        return None
    i = int(np.searchsorted(path.times, t))
    if i < len(path.times) and abs(path.times[i] - t) < 1e-3:
        return path.positions[i]
    if i == 0 or i >= len(path.times) or path.segment[i] != path.segment[i - 1]:
        return None
    t0, t1 = path.times[i - 1], path.times[i]
    if t1 - t0 > MAX_BRIDGE_S:
        return None
    f = (t - t0) / (t1 - t0)
    return path.positions[i - 1] + f * (path.positions[i] - path.positions[i - 1])


def landmarks_payload() -> list[dict]:
    return [
        {"key": lm.key, "label": lm.label, "x_m": lm.x_m, "y_m": lm.y_m, "half": _half(lm.y_m)}
        for lm in LANDMARKS
    ]



# ── Correcting the calibration from the landmarks ────────────────────────────

# The four corners the calibration was made from count in the fit too — a
# corner out of the picture was placed by eye, so less than a landmark.
CORNER_WEIGHT = 0.25
# Fewer landmarks than this cannot be checked one against the others.
REFINE_MIN_LANDMARKS = 4
# Lens corrections tried, and how much better than none one must do — in
# metres of expected error — to be kept.
LENS_STEPS = tuple(round(0.02 * i, 2) for i in range(26))       # 0 … 0.5
LENS_MIN_GAIN_M = 0.03


def refine_calibration(calib: CourtCalibration, marks: dict[str, tuple[float, float]]) -> dict:
    """A calibration fitted on the corners and the marked landmarks — lens
    correction included when it helps — and how wrong it is expected to be.

    The error is measured honestly: each landmark is left out in turn, the
    calibration fitted on the others, and the distance at the one left out
    is its error — a fit is always right on the points it was made from.
    The lens correction is chosen the same way, among LENS_STEPS values: a
    wide-angle lens bends the lines, and the landmarks along them show it."""
    known = [(key, marks[key]) for key in (lm.key for lm in LANDMARKS) if key in marks]
    if len(known) < REFINE_MIN_LANDMARKS:
        raise CalibrationError(
            f"Servono almeno {REFINE_MIN_LANDMARKS} punti del campo segnati per correggere la calibrazione."
        )
    corners_px = np.asarray(calib.corners_px, dtype=np.float64)
    lm_px = np.array([p for _, p in known], dtype=np.float64)
    lm_m = np.array([(_BY_KEY[k].x_m, _BY_KEY[k].y_m) for k, _ in known], dtype=np.float64)
    frame = calib.frame_size

    def fit(k1: float, skip: int | None) -> CourtCalibration:
        keep = [i for i in range(len(known)) if i != skip]
        straight = undistort_points(np.vstack([corners_px, lm_px[keep]]), k1, frame)
        corners = fit_corners(
            straight,
            np.vstack([COURT_CORNERS_M, lm_m[keep]]),
            np.r_[np.full(4, CORNER_WEIGHT), np.ones(len(keep))],
        )
        return build_calibration(distort_points(corners, k1, frame), frame, calib.source, k1=k1,
                                 max_off_frame=DERIVED_MAX_OFF_FRAME)

    def error_at(c: CourtCalibration, i: int) -> float:
        court = c.floor_points_to_court(lm_px[i:i + 1].astype(np.float32))[0]
        return float(np.linalg.norm(court - lm_m[i]))

    before = [error_at(calib, i) for i in range(len(known))]
    best: tuple[float, float, list[float]] | None = None
    for k1 in LENS_STEPS:
        try:
            errors = [error_at(fit(k1, i), i) for i in range(len(known))]
        except CalibrationError:
            continue
        mean = float(np.mean(errors))
        # A lens correction must earn its place over none.
        if best is None or mean < best[0] - (LENS_MIN_GAIN_M if best[1] == 0.0 else 0.0):
            best = (mean, k1, errors)
    if best is None:
        raise CalibrationError("Con questi punti non si ottiene una calibrazione valida.")
    _, k1, after = best
    proposal = fit(k1, None)
    return {
        "corners_px": [[round(float(x), 2), round(float(y), 2)] for x, y in proposal.corners_px],
        "k1": round(k1, 3),
        "before": _summary(before),
        "after": _summary(after),
        "landmarks": len(known),
    }
