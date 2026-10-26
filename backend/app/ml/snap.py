"""Snap a calibration onto the court's painted lines.

The corners are placed by hand, a couple of them often out of the picture,
and every metre the system reports depends on them. The painted lines —
baselines, side lines, service lines, the centre line — are in the picture
by the hundred pixels, and say exactly where the court is.

1. Line evidence: thin bright structures (white top-hat of the darkest
   colour channel: a white line is bright in all three, the blue or green
   turf is not), softened a little. Glass, adverts and wide bright areas
   are not thin and are filtered out.
2. The court model is sampled every 25 cm along the painted lines and
   projected; from every sample the nearest line in the picture is sought
   across the line (along its normal), within a radius.
   Before that, a coarse alignment moves the four corners together to put
   as much of the model as possible on the evidence blurred wide: it brings
   a model tens of pixels off close enough for the search to pick the
   right line.
3. The calibration is fitted again on those correspondences (weighted by
   how clearly the line was seen), and again with a smaller radius — an
   iterative closest point on lines. The lens correction is tried a little
   either side of the current one.
4. Before and after are compared on how much of the model lies on a line
   (the net's tape included, so that a model put on the tape scores less),
   and on how far, in centimetres on the court, the lines found lie from it.
   A snap moving the court by more than MAX_SHIFT_M is refused.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from app.ml.court import (
    COURT_CORNERS_M,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    DERIVED_MAX_OFF_FRAME,
    MAX_K1,
    NET_Y_M,
    SERVICE_LINE_OFFSET_M,
    CalibrationError,
    CourtCalibration,
    build_calibration,
    distort_points,
    fit_corners,
    undistort_points,
)

_NEAR_SERVICE = NET_Y_M - SERVICE_LINE_OFFSET_M
_FAR_SERVICE = NET_Y_M + SERVICE_LINE_OFFSET_M

# Painted lines only: the net is not a line on the floor.
# (start, end, axis): axis 0 = the line runs along X (Y fixed), 1 = along Y.
PAINTED_LINES: tuple[tuple[tuple[float, float], tuple[float, float], int], ...] = (
    ((0.0, 0.0), (COURT_WIDTH_M, 0.0), 0),
    ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M), 0),
    ((0.0, 0.0), (0.0, COURT_LENGTH_M), 1),
    ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M), 1),
    ((0.0, _NEAR_SERVICE), (COURT_WIDTH_M, _NEAR_SERVICE), 0),
    ((0.0, _FAR_SERVICE), (COURT_WIDTH_M, _FAR_SERVICE), 0),
    ((COURT_WIDTH_M / 2, _NEAR_SERVICE), (COURT_WIDTH_M / 2, _FAR_SERVICE), 1),
)
SAMPLE_STEP_M = 0.25
RADII_PX = (30, 24, 18, 14, 10, 8, 6, 6, 5, 5, 4, 4, 4, 4, 4, 4)
# A line counts as seen at a sample when the evidence there is at least this
# (evidence is scaled to 1 at the 99th percentile of the picture).
MIN_EVIDENCE = 0.25
MIN_MATCHES = 40
MIN_LINES = 3
# Height of the net's white tape. It is the most line-like thing in the
# picture, and with a low camera it lies a few pixels from the far lines:
# scored as a line of its own, it explains itself — a model that put the far
# baseline on it leaves its own tape on nothing, and scores less.
NET_TAPE_M = 0.9
# Lines found closer than this to the predicted tape are taken for it.
TAPE_BAND_M = 0.2
# The corners of the coarse alignment take part in every fit with this share
# of the total weight of the lines found: nothing where the lines decide,
# enough to hold the model where they cannot (a far half hidden by the net).
# Those of the coarse alignment, not the hand-placed ones: these can be tens
# of pixels off, and would pull the fit back towards the error.
ANCHOR_SHARE = 0.02
# A snap moving the positions on the court by more than this, on average,
# is a different court, not a refined one.
MAX_SHIFT_M = 1.5
# Search radius of the reported distance of the lines, before and after.
RESIDUAL_RADIUS_PX = 12
# Blur of the line evidence for the coarse alignment, wide to narrow.
COARSE_SIGMAS_PX = (20.0, 10.0, 5.0)
COARSE_DRIFT_COST = 0.2
# Lens corrections tried around the current one.
LENS_SPAN = 0.06
LENS_STEP = 0.02


@dataclass(frozen=True)
class _Sample:
    court: np.ndarray        # (n, 2) metres
    line: np.ndarray         # (n,) index into PAINTED_LINES


def _samples() -> _Sample:
    points, lines = [], []
    for index, (a, b, _) in enumerate(PAINTED_LINES):
        a, b = np.asarray(a), np.asarray(b)
        n = max(int(np.linalg.norm(b - a) / SAMPLE_STEP_M), 1)
        for f in np.linspace(0.0, 1.0, n + 1):
            points.append(a + (b - a) * f)
            lines.append(index)
    return _Sample(np.asarray(points, dtype=np.float64), np.asarray(lines))


def line_evidence(image: np.ndarray) -> np.ndarray:
    """Per pixel, how much it looks like a painted line (0..~1)."""
    darkest = image.min(axis=2) if image.ndim == 3 else image
    h = image.shape[0]
    size = max(9, int(round(h * 0.025)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    tophat = cv2.morphologyEx(darkest, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    tophat = cv2.GaussianBlur(tophat, (0, 0), 1.5)
    scale = float(np.percentile(tophat, 99)) or 1.0
    return np.clip(tophat / scale, 0.0, 1.5)


def _tape_row(calib: CourtCalibration):
    """(columns, rows) of the net's tape as the calibration predicts it, and
    the picture's scale across the net in pixels per metre."""
    px = calib.court_to_pixels(_TAPE_M.astype(np.float32)).astype(np.float64)
    if not np.isfinite(px).all():
        return None
    tape = _with_tape(np.zeros((0, 2)), px)
    per_metre = float(np.linalg.norm(px[-1] - px[0])) / COURT_WIDTH_M
    order = np.argsort(tape[:, 0])
    return tape[order, 0], tape[order, 1], per_metre


def _matches(calib: CourtCalibration, evidence: np.ndarray, samples: _Sample, radius: int):
    """For each visible sample, the line found across it: (pixel, metres,
    weight, line index). A line found on the net's tape is the tape, not a
    court line, and is left out."""
    h, w = evidence.shape
    pixels = calib.court_to_pixels(samples.court.astype(np.float32)).astype(np.float64)
    # The line's direction in the picture, from a step along it.
    step = np.zeros_like(samples.court)
    for index, (a, b, _) in enumerate(PAINTED_LINES):
        d = (np.asarray(b) - np.asarray(a)) / np.linalg.norm(np.asarray(b) - np.asarray(a))
        step[samples.line == index] = d * 0.05
    ahead = calib.court_to_pixels((samples.court + step).astype(np.float32)).astype(np.float64)
    tangent = ahead - pixels
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    ok = (norm[:, 0] > 1e-6) & np.isfinite(pixels).all(axis=1)
    normal = np.zeros_like(tangent)
    normal[ok] = np.c_[-tangent[ok, 1], tangent[ok, 0]] / norm[ok]
    inside = ok & (pixels[:, 0] >= radius) & (pixels[:, 0] < w - radius) \
        & (pixels[:, 1] >= radius) & (pixels[:, 1] < h - radius)
    idx = np.flatnonzero(inside)
    if idx.size == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=int)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    xs = (pixels[idx, 0, None] + offsets[None, :] * normal[idx, 0, None]).astype(np.float32)
    ys = (pixels[idx, 1, None] + offsets[None, :] * normal[idx, 1, None]).astype(np.float32)
    profile = cv2.remap(evidence, xs, ys, cv2.INTER_LINEAR)
    best = profile.argmax(axis=1)
    strength = profile[np.arange(len(idx)), best]
    # A peak at the edge of the search is a line beyond it, not this one.
    keep = (strength >= MIN_EVIDENCE) & (best > 0) & (best < len(offsets) - 1)
    idx, best, strength, profile = idx[keep], best[keep], strength[keep], profile[keep]
    # Sub-pixel: the top of the parabola through the peak and its neighbours.
    rows = np.arange(len(idx))
    left, centre, right = profile[rows, best - 1], profile[rows, best], profile[rows, best + 1]
    curvature = left - 2.0 * centre + right
    shift = np.where(curvature < -1e-6, 0.5 * (left - right) / np.where(curvature < -1e-6, curvature, -1.0), 0.0)
    found = pixels[idx] + (offsets[best] + np.clip(shift, -0.5, 0.5))[:, None] * normal[idx]
    tape = _tape_row(calib)
    if tape is not None:
        cols, rows, per_metre = tape
        near_tape = np.abs(found[:, 1] - np.interp(found[:, 0], cols, rows)) < max(TAPE_BAND_M * per_metre, 4.0)
        on_net = (found[:, 0] >= cols[0]) & (found[:, 0] <= cols[-1]) & near_tape
        found, idx, strength = found[~on_net], idx[~on_net], strength[~on_net]
    return found, samples.court[idx], strength.astype(np.float64), samples.line[idx]


def _residual_m(calib: CourtCalibration, found_px: np.ndarray, line: np.ndarray) -> np.ndarray:
    """Distance on the court, across the line, between each line found in
    the picture and the model line it belongs to."""
    court = calib.floor_points_to_court(found_px.astype(np.float32)).astype(np.float64)
    out = np.empty(len(found_px))
    for index, (a, _, axis) in enumerate(PAINTED_LINES):
        mask = line == index
        # A line along X has a fixed Y: the error is in Y; and vice versa.
        out[mask] = np.abs(court[mask, 1 - axis] - a[1 - axis])
    return out


def _project(corners: np.ndarray, k1: float, frame_size, court: np.ndarray) -> np.ndarray | None:
    """Court metres to picture pixels for corners `corners` (picture) and
    lens `k1`, without the checks of build_calibration: fast enough to be
    evaluated thousands of times."""
    straight = undistort_points(corners, k1, frame_size).astype(np.float32)
    try:
        h = cv2.getPerspectiveTransform(COURT_CORNERS_M, straight)
    except cv2.error:
        return None
    if h is None or not np.isfinite(h).all():
        return None
    px = cv2.perspectiveTransform(court.astype(np.float32).reshape(-1, 1, 2), h).reshape(-1, 2)
    if not np.isfinite(px).all():
        return None
    return distort_points(px, k1, frame_size)


def _with_tape(px: np.ndarray | None, tape_foot: np.ndarray | None) -> np.ndarray | None:
    """The model's samples plus the net's tape: its foot raised by
    NET_TAPE_M at the picture's scale across the net (a camera looking
    roughly level sees a metre up about as large as a metre across)."""
    if px is None or tape_foot is None or not np.isfinite(tape_foot).all():
        return px
    per_metre = float(np.linalg.norm(tape_foot[-1] - tape_foot[0])) / COURT_WIDTH_M
    return np.vstack([px, tape_foot - [0.0, NET_TAPE_M * per_metre]])


_TAPE_M = np.c_[np.linspace(0.0, COURT_WIDTH_M, 41), np.full(41, NET_Y_M)]


def _alignment(evidence: np.ndarray, px: np.ndarray | None) -> float:
    """How much line lies under the projected model: the mean evidence at
    its samples (0 outside the picture)."""
    if px is None:
        return 0.0
    values = cv2.remap(evidence, px[:, 0].astype(np.float32).reshape(1, -1),
                       px[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return float(np.minimum(values, 1.0).mean())


def _coarse(calib: CourtCalibration, evidence: np.ndarray, samples: _Sample) -> np.ndarray:
    """Move the four corners together to put as much model on line as
    possible, on the evidence blurred wide and then less: wide blur pulls
    the model from tens of pixels away as a whole, before any single line
    can capture the wrong model line."""
    corners = np.asarray(calib.corners_px, dtype=np.float64)
    for sigma in COARSE_SIGMAS_PX:
        blurred = cv2.GaussianBlur(evidence, (0, 0), sigma)
        blurred = blurred / max(float(blurred.max()), 1e-9)

        diagonal = float(np.hypot(*calib.frame_size))

        def cost(delta, start=corners, blurred=blurred):
            moved = start + delta.reshape(4, 2)
            # A small price on moving away from the start: among equally good
            # fits, the nearest — and no wandering where nothing is seen.
            drift = float(np.abs(moved - np.asarray(calib.corners_px)).mean()) / diagonal
            projected = _project(moved, calib.k1, calib.frame_size, np.vstack([samples.court, _TAPE_M]))
            if projected is None:
                return 1.0
            model = _with_tape(projected[: len(samples.court)], projected[len(samples.court):])
            return -_alignment(blurred, model) + COARSE_DRIFT_COST * drift

        result = minimize(cost, np.zeros(8), method="Powell",
                          options={"xtol": 0.5, "ftol": 1e-4, "maxfev": 3000})
        corners = corners + result.x.reshape(4, 2)
    return corners


def snap_to_lines(calib: CourtCalibration, image: np.ndarray) -> dict:
    """A calibration snapped onto the painted lines of `image` (the frame it
    was made on), with how much of the model lies on the lines and how far,
    in centimetres on the court, the lines are from it — before and after."""
    if image is None or image.size == 0:
        raise CalibrationError("Fotogramma non disponibile.")
    if (image.shape[1], image.shape[0]) != tuple(calib.frame_size):
        image = cv2.resize(image, tuple(calib.frame_size), interpolation=cv2.INTER_AREA)
    evidence = line_evidence(image)
    samples = _samples()

    def aligned(c: CourtCalibration) -> float:
        px = c.court_to_pixels(np.vstack([samples.court, _TAPE_M]).astype(np.float32)).astype(np.float64)
        return _alignment(evidence, _with_tape(px[: len(samples.court)], px[len(samples.court):]))

    def residual_cm(c: CourtCalibration) -> float | None:
        found, _, _, line = _matches(c, evidence, samples, RESIDUAL_RADIUS_PX)
        if len(found) < MIN_MATCHES:
            return None
        return round(float(np.median(_residual_m(c, found, line))) * 100, 1)

    try:
        start = build_calibration(_coarse(calib, evidence, samples), calib.frame_size, calib.source, k1=calib.k1,
                                  max_off_frame=DERIVED_MAX_OFF_FRAME)
    except CalibrationError:
        start = calib

    best: tuple[float, CourtCalibration] | None = None
    lenses = np.arange(max(0.0, calib.k1 - LENS_SPAN), min(MAX_K1, calib.k1 + LENS_SPAN) + 1e-9, LENS_STEP)
    for k1 in sorted(set(np.round(np.r_[lenses, calib.k1], 3))):
        current = start
        try:
            for radius in RADII_PX:
                px, metres, weight, lines = _matches(current, evidence, samples, radius)
                if len(px) < MIN_MATCHES or len(np.unique(lines)) < MIN_LINES:
                    raise CalibrationError("linee insufficienti")
                anchors = undistort_points(np.asarray(start.corners_px, np.float64), k1, calib.frame_size)
                straight = undistort_points(px, k1, calib.frame_size)
                corners = fit_corners(
                    np.vstack([straight, anchors]),
                    np.vstack([metres, COURT_CORNERS_M]),
                    np.r_[weight, np.full(4, ANCHOR_SHARE * weight.sum() / 4.0)],
                )
                current = build_calibration(distort_points(corners, k1, calib.frame_size),
                                            calib.frame_size, calib.source, k1=float(k1),
                                            max_off_frame=DERIVED_MAX_OFF_FRAME)
        except CalibrationError:
            continue
        score = aligned(current)
        if best is None or score > best[0]:
            best = (score, current)
    if best is None:
        raise CalibrationError(
            "Non trovo abbastanza linee del campo nel fotogramma: aggancio non possibile. "
            "Avvicina prima gli angoli alle linee a mano."
        )
    snapped = best[1]
    before, after = aligned(calib), best[0]
    shift = _court_shift(calib, snapped)
    if shift > MAX_SHIFT_M:
        raise CalibrationError(
            f"L'aggancio sposterebbe il campo di {shift:.1f} m in media: probabilmente ha preso linee "
            "sbagliate. Avvicina prima gli angoli a mano e riprova."
        )
    return {
        "corners_px": [[round(float(x), 2), round(float(y), 2)] for x, y in snapped.corners_px],
        "k1": round(float(snapped.k1), 3),
        # Share of the model that lies on a painted line.
        "aligned_before": round(before, 3),
        "aligned_after": round(after, 3),
        # Median distance, across the lines, of the lines found from the model.
        "before_cm": residual_cm(calib),
        "after_cm": residual_cm(snapped),
        # Judged on the alignment, which does not depend on which line each
        # sample was paired with: a snap onto the wrong lines lowers it.
        "improved": bool(after > before * 1.05 + 0.01),
        "shift_m": round(shift, 2),
    }


def _court_shift(a: CourtCalibration, b: CourtCalibration) -> float:
    """Mean distance, on the visible court, between where the two
    calibrations place the same pixel."""
    xs, ys = np.meshgrid(np.linspace(0.5, COURT_WIDTH_M - 0.5, 7), np.linspace(0.5, COURT_LENGTH_M - 0.5, 15))
    court = np.c_[xs.ravel(), ys.ravel()].astype(np.float32)
    px = a.court_to_pixels(court)
    w, h = a.frame_size
    visible = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    if not visible.any():
        return 0.0
    moved = b.floor_points_to_court(px[visible])
    return float(np.linalg.norm(moved - court[visible], axis=1).mean())
