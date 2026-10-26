"""Where a player stands when the box does not reach the feet.

The foot point is the middle of the box's bottom edge. Behind the net that
fails in a systematic way: with the camera a couple of metres up, the net's
white tape crosses the far players at the knees, the detector closes their
box on it, and the bottom edge — 0.9 m above the floor — projected onto the
floor lands metres behind the player, often off the court. The far half of
every match ends up displaced, and the smoothing cannot repair a bias.

What is left of the box still says how far the player is. With an upright
camera, rows below the floor's horizon shrink in proportion to distance, so
for a person standing on the floor, D being the foot row's distance below
the horizon:

    height:  (foot row - top row) = rho * D     rho = player / camera height
    width:   box width            = omega * D   omega = player width / camera height

rho and omega are measured on the match itself — medians over the boxes in
the near half, feet in sight — so they carry the ready stance, the racket
and the detector's margins with them. For a box cut by the tape both give D:
the head through rho (well conditioned with the camera well above the
players, rho small), the width through omega (rougher, but independent of
the camera height). The two are averaged, weighted by how much each varies
in this match. The foot is kept between the tape and the foot of the net:
the player is beyond the net.

Everything is computed on the straightened picture, where the homography
and the horizon live; the result is returned in picture pixels.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from app.ml.court import (
    COURT_WIDTH_M,
    NET_HEIGHT_CENTER_M,
    NET_HEIGHT_POST_M,
    NET_Y_M,
    CourtCalibration,
    distort_points,
    undistort_points,
)

# Recorded in the artifacts' meta.json by runs whose foot points went through
# this model.
FOOT_MODEL_VERSION = 1

TAPE_HEIGHT_M = (NET_HEIGHT_CENTER_M + NET_HEIGHT_POST_M) / 2.0
# The ratios are trusted from this many boxes on: before, boxes are left as
# they are.
MIN_SAMPLES = 25
# The most recent boxes only: the cost stays flat over a two-hour match.
MAX_SAMPLES = 3000
# The medians are taken again after this many new boxes, not after each one.
REFRESH_EVERY = 50
# A box is cut by the net when its bottom lies on the tape: at most this
# share of the band between tape and net foot (in rows) below it, and at most
# TAPE_TOLERANCE_ABOVE above it, at least TAPE_TOLERANCE_PX either way. Above
# the tape further than that, the feet are in sight: the box is whole.
TAPE_TOLERANCE = 0.35
TAPE_TOLERANCE_ABOVE = 0.1
TAPE_TOLERANCE_PX = 3.0
# Boxes that teach the ratios: bottom at least this many tape bands below the
# net foot (the near half, feet in sight) and this many rows below the horizon.
LEARN_BELOW_NET = 1.0
LEARN_MIN_ROWS = 20.0
# The correction must lower the foot by at least this share of the box. Kept
# at nothing: a box ending on the tape is cut far more often than not, and a
# whole one is only moved by the cues' noise.
MIN_HIDDEN = 0.0
# A box closed by the tape has the feet behind it: at least this much of the
# player, in metres at the net, is hidden below the box — the tape's own
# height. When the cues put the feet higher (a player side-on, narrow), this
# is the estimate: still beyond the box, never on it.
TAPE_HIDES_M = 0.1
# Camera near a player's height: rho near 1, the head on the horizon wherever
# the player stands, and the top says nothing about the distance.
MAX_RHO = 0.95
# Floor of the relative spread of a ratio: a handful of identical boxes must
# not make one cue absolute.
MIN_SPREAD = 0.03


class FootModel:
    """Foot points for one calibration, the ratios learned from the boxes seen."""

    def __init__(self, calibration: CourtCalibration) -> None:
        self.calibration = calibration
        self._heights: deque[float] = deque(maxlen=MAX_SAMPLES)
        self._widths: deque[float] = deque(maxlen=MAX_SAMPLES)
        # (rho, its relative spread, omega, its relative spread)
        self._ratios: tuple[float, float, float, float] | None = None
        self._new = 0
        self.corrected = 0
        # The foot of the net, as a straight line in the straightened picture.
        base = _straight_project(calibration, np.array([[0.0, NET_Y_M], [COURT_WIDTH_M, NET_Y_M]]))
        self._base = base if base is not None and abs(base[1, 0] - base[0, 0]) > 1e-6 else None

    @property
    def rho(self) -> float | None:
        """Box height over the distance below the horizon (player height
        over camera height), as seen in this match."""
        ratios = self._current()
        return ratios[0] if ratios else None

    def fit(self, bboxes) -> "FootModel":
        """Learn the ratios from a whole match at once (stored runs)."""
        for bbox in bboxes:
            self.learn(bbox)
        return self

    def learn(self, bbox) -> None:
        geometry = self._geometry(bbox, self._ratios[0] if self._ratios else None)
        if geometry is None:
            return
        top, bottom, width, horizon, base, band = geometry
        _, frame_h = self.calibration.frame_size
        depth = bottom - horizon
        if bbox[3] >= frame_h - 2 or depth < LEARN_MIN_ROWS or bottom < base + LEARN_BELOW_NET * band:
            return
        self._heights.append((bottom - top) / depth)
        self._widths.append(width / depth)
        self._new += 1

    def foot(self, bbox) -> tuple[float, float]:
        """The foot point of `bbox`, in picture pixels."""
        x1, _, x2, y2 = bbox
        raw = ((x1 + x2) / 2.0, y2)
        ratios = self._current()
        if ratios is None:
            return raw
        rho, rho_spread, omega, omega_spread = ratios
        geometry = self._geometry(bbox, rho)
        if geometry is None:
            return raw
        top, bottom, width, horizon, base, band = geometry
        if not self._cut(bottom, base, band):
            return raw

        # Each cue's estimate of log D and its variance (relative errors).
        estimates = []
        if 0.0 < rho < MAX_RHO and top > horizon - 1e3:
            depth = (top - horizon) / (1.0 - rho)
            if depth > 0:
                estimates.append((np.log(depth), (rho * rho_spread / (1.0 - rho)) ** 2))
        if omega > 0 and width > 0:
            estimates.append((np.log(width / omega), omega_spread ** 2))
        lowest = bottom + band * TAPE_HIDES_M / TAPE_HEIGHT_M
        row = lowest
        if estimates:
            weights = np.array([1.0 / v for _, v in estimates])
            depth = float(np.exp(np.dot(weights, [e for e, _ in estimates]) / weights.sum()))
            if np.isfinite(depth):
                row = max(horizon + depth, lowest)
        row = min(row, base)
        if not np.isfinite(row) or row - bottom <= MIN_HIDDEN * (bottom - top):
            return raw
        column = float(undistort_points(np.array([raw]), self.calibration.k1, self.calibration.frame_size)[0, 0])
        x, y = distort_points(np.array([[column, row]]), self.calibration.k1, self.calibration.frame_size)[0]
        self.corrected += 1
        return float(x), float(y)

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _cut(bottom: float, base: float, band: float) -> bool:
        """Whether a box ending at row `bottom` ends on the net's tape."""
        offset = bottom - (base - band)          # below the tape: positive
        return (offset <= max(TAPE_TOLERANCE * band, TAPE_TOLERANCE_PX)
                and -offset <= max(TAPE_TOLERANCE_ABOVE * band, TAPE_TOLERANCE_PX))

    def _current(self) -> tuple[float, float, float, float] | None:
        if len(self._heights) >= MIN_SAMPLES and (self._ratios is None or self._new >= REFRESH_EVERY):
            self._ratios = (*_centre_and_spread(self._heights), *_centre_and_spread(self._widths))
            self._new = 0
        return self._ratios

    def _geometry(self, bbox, rho: float | None):
        """(top, bottom, width, horizon, net foot, tape band) on the box's
        column of the straightened picture; None where undefined."""
        if self._base is None:
            return None
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        (_, top), (bx, bottom), (left, _), (right, _) = undistort_points(
            np.array([[cx, y1], [cx, y2], [x1, y2], [x2, y2]]), self.calibration.k1, self.calibration.frame_size)
        horizon = self.calibration.horizon_y(bx)
        if horizon is None or bottom <= horizon:
            return None
        (ax, ay), (bx2, by) = self._base
        base = ay + (by - ay) * (bx - ax) / (bx2 - ax)
        if base <= horizon:
            return None
        # The tape stands TAPE_HEIGHT_M over the net foot: in rows, its share
        # of the camera height times the net foot's distance below the
        # horizon. Camera height in player heights is 1 / rho; a player is
        # taken as 1.75 m. Before rho is known, a camera at 3 m.
        camera_m = 1.75 / rho if rho else 3.0
        band = (base - horizon) * TAPE_HEIGHT_M / camera_m
        return float(top), float(bottom), float(right - left), float(horizon), float(base), float(band)


def _centre_and_spread(values) -> tuple[float, float]:
    """Median, and the relative spread around it (scaled MAD)."""
    array = np.asarray(values, dtype=np.float64)
    centre = float(np.median(array))
    spread = 1.4826 * float(np.median(np.abs(array - centre))) / centre if centre > 0 else 1.0
    return centre, max(spread, MIN_SPREAD)


def _straight_project(calibration: CourtCalibration, points_m: np.ndarray) -> np.ndarray | None:
    """Court metres to the straightened picture (no lens distortion)."""
    h = np.asarray(calibration.homography_inv, dtype=np.float64)
    pts = np.c_[points_m, np.ones(len(points_m))] @ h.T
    if np.any(np.abs(pts[:, 2]) < 1e-12):
        return None
    return pts[:, :2] / pts[:, 2:3]
