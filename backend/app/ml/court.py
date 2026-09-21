"""Court geometry and homography calibration.

What changed and why
--------------------
The previous implementation tried three automatic strategies (white-line
Hough, HSV colour segmentation, then a hardcoded trapezoid) and silently fell
through to the hardcoded one. On a padel court — glass walls, metal mesh,
few white lines, reflective surfaces — the first two fail most of the time,
so in practice the system ran on invented corners. Every metric downstream is
expressed in metres, so invented corners produce numbers that look precise
and are wrong.

The new contract:
  * A calibration is only usable if a human confirmed it (MANUAL, PRESET, or
    an AUTO suggestion the user explicitly accepted).
  * `suggest_corners` is exactly that — a suggestion to drag into place. It
    never becomes a calibration on its own.
  * `build_calibration` validates the quadrilateral geometrically and refuses
    degenerate input instead of returning something unusable.

Ball height
-----------
The homography maps the *court floor plane* to metres. Projecting an airborne
ball through it places the ball where the camera ray hits the floor, which is
metres away from the truth. This module therefore exposes
`floor_points_to_court` — the name states the precondition. Player feet
satisfy it; a flying ball does not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

# ── Regulation padel court (metres) ──────────────────────────────────────────
COURT_WIDTH_M  = 10.0
COURT_LENGTH_M = 20.0
NET_Y_M        = 10.0
SERVICE_LINE_OFFSET_M = 6.95      # service line distance from the net
NET_HEIGHT_CENTER_M = 0.88
NET_HEIGHT_POST_M   = 0.92

# Court corners in metres, in the order the UI asks the user to place them:
# far-left, far-right, near-right, near-left (clockwise on screen).
COURT_CORNERS_M = np.array(
    [
        [0.0,            COURT_LENGTH_M],   # TL — far baseline, left
        [COURT_WIDTH_M,  COURT_LENGTH_M],   # TR — far baseline, right
        [COURT_WIDTH_M,  0.0],              # BR — near baseline, right
        [0.0,            0.0],              # BL — near baseline, left
    ],
    dtype=np.float32,
)

CORNER_LABELS = ("Fondo sinistra", "Fondo destra", "Vicino destra", "Vicino sinistra")


class CalibrationSource(str, Enum):
    MANUAL = "manual"
    PRESET = "preset"
    AUTO   = "auto"


class CalibrationError(ValueError):
    """Raised when the supplied corners cannot define a usable homography."""


@dataclass(frozen=True)
class CourtCalibration:
    """Homography between image pixels and court metres.

    Court frame: origin at the near-left corner, X across the width (0-10 m),
    Y along the length (0-20 m), net at Y = 10 m.
    """

    homography: np.ndarray        # 3x3, pixels -> metres
    homography_inv: np.ndarray    # 3x3, metres -> pixels
    corners_px: np.ndarray        # 4x2, image space, in COURT_CORNERS_M order
    frame_size: tuple[int, int]   # (w, h)
    source: CalibrationSource
    net_error_m: float | None = None   # deviation of the marked net from Y=10 m

    # ── Projections ──────────────────────────────────────────────────────────

    def floor_points_to_court(self, points_px: np.ndarray) -> np.ndarray:
        """Map Nx2 pixel coordinates **that lie on the court floor** to metres.

        Precondition: the points touch the ground (player feet, ball bounce).
        Airborne points are projected onto the floor along the camera ray and
        the result is meaningless — do not call this for a ball in flight.
        """
        pts = np.asarray(points_px, dtype=np.float32)
        if pts.size == 0:
            return pts.reshape(-1, 2)
        pts = pts.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def court_to_pixels(self, points_m: np.ndarray) -> np.ndarray:
        """Map Nx2 court coordinates back to image pixels (for overlays)."""
        pts = np.asarray(points_m, dtype=np.float32)
        if pts.size == 0:
            return pts.reshape(-1, 2)
        pts = pts.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography_inv).reshape(-1, 2)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def court_polygon_px(self) -> np.ndarray:
        return self.corners_px.astype(np.float32)

    def roi_px(self, margin_m: float = 1.5, height_headroom: float = 0.45) -> tuple[int, int, int, int]:
        """Bounding box of the court plus margin, clamped to the frame.

        The box is extended upward by `height_headroom` of its own height
        because players (and the glass walls behind them) stand *above* the
        floor plane and would otherwise be cut off. Cropping to this ROI
        before detection makes players larger in the letterboxed network
        input, which measurably improves recall on the far baseline.
        """
        w, h = self.frame_size
        extra = self.court_to_pixels(
            np.array(
                [
                    [-margin_m, -margin_m],
                    [COURT_WIDTH_M + margin_m, -margin_m],
                    [COURT_WIDTH_M + margin_m, COURT_LENGTH_M + margin_m],
                    [-margin_m, COURT_LENGTH_M + margin_m],
                ],
                dtype=np.float32,
            )
        )
        pts = np.vstack([self.corners_px, extra])
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 4:
            return 0, 0, w, h

        x1 = int(np.clip(pts[:, 0].min(), 0, w - 1))
        x2 = int(np.clip(pts[:, 0].max(), 1, w))
        y1 = int(np.clip(pts[:, 1].min(), 0, h - 1))
        y2 = int(np.clip(pts[:, 1].max(), 1, h))

        box_h = y2 - y1
        y1 = int(max(0, y1 - box_h * height_headroom))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return 0, 0, w, h
        return x1, y1, x2, y2

    def to_dict(self) -> dict:
        return {
            "corners_px":  self.corners_px.tolist(),
            "frame_size":  list(self.frame_size),
            "source":      self.source.value,
            "net_error_m": self.net_error_m,
            "homography":  self.homography.tolist(),
        }


# ── Construction & validation ────────────────────────────────────────────────

def build_calibration(
    corners_px: np.ndarray | list,
    frame_size: tuple[int, int],
    source: CalibrationSource = CalibrationSource.MANUAL,
    net_px: np.ndarray | list | None = None,
) -> CourtCalibration:
    """Build and validate a calibration from four user-placed court corners.

    `corners_px` must be in COURT_CORNERS_M order (far-left, far-right,
    near-right, near-left). The UI labels each handle, so no automatic
    reordering happens here — guessing the order from pixel sums is exactly
    what used to silently mirror the court on angled camera positions.

    Raises CalibrationError with a user-facing Italian message.
    """
    pts = np.asarray(corners_px, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        raise CalibrationError("Servono esattamente 4 angoli del campo.")
    if not np.isfinite(pts).all():
        raise CalibrationError("Coordinate degli angoli non valide.")

    w, h = int(frame_size[0]), int(frame_size[1])
    if w <= 0 or h <= 0:
        raise CalibrationError("Dimensioni del frame non valide.")

    _validate_quad(pts, w, h)

    # Exact 4-point solve: with four correspondences the homography is unique,
    # so RANSAC would only add a failure mode.
    homography = cv2.getPerspectiveTransform(pts, COURT_CORNERS_M)
    if homography is None or not np.isfinite(homography).all():
        raise CalibrationError("Impossibile calcolare l'omografia dagli angoli indicati.")

    if not _homography_is_stable(homography, pts):
        raise CalibrationError(
            "Gli angoli indicati producono una prospettiva degenere: "
            "controlla che seguano il perimetro del campo senza incrociarsi."
        )

    homography_inv = np.linalg.inv(homography)

    calib = CourtCalibration(
        homography=homography.astype(np.float64),
        homography_inv=homography_inv.astype(np.float64),
        corners_px=pts,
        frame_size=(w, h),
        source=source,
    )

    net_error = _net_error_m(calib, net_px)
    if net_error is not None:
        calib = CourtCalibration(
            homography=calib.homography,
            homography_inv=calib.homography_inv,
            corners_px=calib.corners_px,
            frame_size=calib.frame_size,
            source=calib.source,
            net_error_m=net_error,
        )
    return calib


def calibration_from_dict(data: dict) -> CourtCalibration:
    """Rebuild a calibration from the JSON stored on the match row."""
    source = CalibrationSource(data.get("source", CalibrationSource.MANUAL.value))
    fw, fh = data["frame_size"]
    return build_calibration(
        corners_px=data["corners_px"],
        frame_size=(int(fw), int(fh)),
        source=source,
        net_px=data.get("net_px"),
    )


def _validate_quad(pts: np.ndarray, w: int, h: int) -> None:
    # Distinct points
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.linalg.norm(pts[i] - pts[j])) < 8.0:
                raise CalibrationError("Due angoli coincidono: spostali sui quattro vertici del campo.")

    # Convex, no bow-tie, no three collinear points: every turn must bend the
    # same way as we walk the perimeter.
    crosses = [
        _cross2(pts[(i + 1) % 4] - pts[i], pts[(i + 2) % 4] - pts[(i + 1) % 4])
        for i in range(4)
    ]
    if any(abs(c) < 1e-6 for c in crosses) or (min(crosses) < 0 < max(crosses)):
        raise CalibrationError(
            "Il quadrilatero si incrocia o ha tre angoli allineati. "
            "Posiziona gli angoli seguendo il perimetro del campo."
        )

    # Winding. The four corners are expected in the order the UI asks for
    # them (fondo-sinistra, fondo-destra, vicino-destra, vicino-sinistra),
    # which traverses the court clockwise on screen and gives a positive
    # shoelace area with Y pointing down. A negative area means the corners
    # were entered in reverse: the homography would still solve, and would
    # silently mirror the court left-to-right — swapping the two players of
    # each pair and flipping every heatmap. This is the failure the old
    # sum/difference sorting used to introduce on angled cameras.
    signed_area = _signed_area(pts)
    if signed_area < 0:
        raise CalibrationError(
            "Gli angoli sono in ordine invertito (antiorario): il campo "
            "risulterebbe specchiato. Inseriscili partendo dal fondo a "
            "sinistra e procedendo in senso orario."
        )

    # Meaningful size: a court smaller than 2% of the frame cannot yield
    # position accuracy better than a metre or two.
    if abs(signed_area) < 0.02 * w * h:
        raise CalibrationError(
            "Il campo selezionato è troppo piccolo nel frame: "
            "avvicina la camera o inquadra meglio il campo."
        )


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _signed_area(pts: np.ndarray) -> float:
    """Shoelace area; positive when the points run clockwise on screen."""
    return 0.5 * float(
        np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
    )


def _homography_is_stable(homography: np.ndarray, corners: np.ndarray) -> bool:
    """Reject homographies whose vanishing line crosses the court.

    If the third homogeneous coordinate changes sign inside the quadrilateral,
    points on the court map through infinity and metre coordinates explode.
    """
    centroid = corners.mean(axis=0)
    samples = np.vstack([corners, centroid[None, :], (corners + centroid) / 2.0])
    ws = samples @ homography[2, :2] + homography[2, 2]
    if not np.isfinite(ws).all():
        return False
    if np.any(np.abs(ws) < 1e-9):
        return False
    return bool(np.all(ws > 0) or np.all(ws < 0))


def _net_error_m(calib: CourtCalibration, net_px: np.ndarray | list | None) -> float | None:
    """How far the user-marked net line lands from Y = 10 m, in metres.

    This is the cheapest independent check available: the net is the one
    landmark whose court position is known exactly and that is not one of the
    four points used to fit the homography. An error above ~0.5 m means the
    corners are misplaced.
    """
    if net_px is None:
        return None
    pts = np.asarray(net_px, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 2 or not np.isfinite(pts).all():
        return None
    court = calib.floor_points_to_court(pts)
    return float(np.max(np.abs(court[:, 1] - NET_Y_M)))


# ── Suggestion (never a calibration on its own) ──────────────────────────────

@dataclass(frozen=True)
class CornerSuggestion:
    corners_px: np.ndarray
    method: str        # "lines" | "default"
    confidence: float  # 0.0 for the geometric default

    def to_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.corners_px]


def suggest_corners(frame: np.ndarray) -> CornerSuggestion:
    """Propose four corners for the calibration UI to pre-fill.

    Strategy: find long straight segments, split them into the two pencils
    that a projected rectangle produces, and intersect the extreme lines of
    each pencil. This is far more robust on a padel court than thresholding
    for white pixels, because it uses gradient direction rather than colour.

    When it fails — which it is allowed to do — a generic trapezoid for the
    usual "camera behind the baseline, elevated" setup is returned with
    confidence 0. The user drags four handles either way, so a failed
    suggestion costs seconds, not correctness.
    """
    try:
        found = _suggest_from_lines(frame)
        if found is not None:
            return found
    except cv2.error:
        pass
    return CornerSuggestion(_default_trapezoid(frame), method="default", confidence=0.0)


def _default_trapezoid(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return np.array(
        [
            [w * 0.24, h * 0.16],   # far-left
            [w * 0.76, h * 0.16],   # far-right
            [w * 0.95, h * 0.90],   # near-right
            [w * 0.05, h * 0.90],   # near-left
        ],
        dtype=np.float32,
    )


def _suggest_from_lines(frame: np.ndarray) -> CornerSuggestion | None:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 60, 60)
    edges = cv2.Canny(gray, 45, 140, apertureSize=3)

    min_len = int(min(w, h) * 0.22)
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=70, minLineLength=min_len, maxLineGap=18
    )
    if segments is None or len(segments) < 6:
        return None

    lines = [_segment_to_line(s[0]) for s in segments]
    lines = [ln for ln in lines if ln is not None]
    if len(lines) < 6:
        return None

    # Two families: roughly-horizontal (baselines/service lines) and
    # roughly-vertical (sidelines) in image space. A camera behind the
    # baseline keeps this split valid for any realistic tilt.
    horiz = [ln for ln in lines if abs(math.sin(ln[2])) >= 0.55]
    vert  = [ln for ln in lines if abs(math.sin(ln[2])) < 0.55]
    if len(horiz) < 2 or len(vert) < 2:
        return None

    # Extremes: topmost/bottommost horizontals, leftmost/rightmost verticals,
    # measured at the frame centre so perspective does not flip the ordering.
    cx, cy = w / 2.0, h / 2.0
    horiz.sort(key=lambda ln: _y_at_x(ln, cx))
    vert.sort(key=lambda ln: _x_at_y(ln, cy))
    top, bottom = horiz[0], horiz[-1]
    left, right = vert[0], vert[-1]

    if _y_at_x(bottom, cx) - _y_at_x(top, cx) < h * 0.25:
        return None
    if _x_at_y(right, cy) - _x_at_y(left, cy) < w * 0.25:
        return None

    corners = []
    for a, b in ((top, left), (top, right), (bottom, right), (bottom, left)):
        p = _intersect(a, b)
        if p is None:
            return None
        corners.append(p)
    pts = np.array(corners, dtype=np.float32)

    # Accept only suggestions that are inside (a padded) frame and convex —
    # otherwise the default trapezoid is a better starting point.
    pad = 0.25
    if (pts[:, 0] < -pad * w).any() or (pts[:, 0] > (1 + pad) * w).any():
        return None
    if (pts[:, 1] < -pad * h).any() or (pts[:, 1] > (1 + pad) * h).any():
        return None
    try:
        _validate_quad(pts, w, h)
    except CalibrationError:
        return None

    return CornerSuggestion(pts, method="lines", confidence=0.45)


def _segment_to_line(seg: np.ndarray) -> tuple[float, float, float] | None:
    """Segment -> (a, b, theta) for the line y = a*x + b, plus its angle.

    Returns None for near-vertical segments, which are re-parameterised by
    `_x_at_y` through the stored angle instead.
    """
    x1, y1, x2, y2 = (float(v) for v in seg)
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    theta = math.atan2(dy, dx)
    # Store the line in general form via a point and its direction.
    return (x1, y1, theta)


def _y_at_x(line: tuple[float, float, float], x: float) -> float:
    px, py, theta = line
    dx, dy = math.cos(theta), math.sin(theta)
    if abs(dx) < 1e-6:
        return py
    return py + (x - px) * dy / dx


def _x_at_y(line: tuple[float, float, float], y: float) -> float:
    px, py, theta = line
    dx, dy = math.cos(theta), math.sin(theta)
    if abs(dy) < 1e-6:
        return px
    return px + (y - py) * dx / dy


def _intersect(l1: tuple[float, float, float], l2: tuple[float, float, float]) -> tuple[float, float] | None:
    p1x, p1y, t1 = l1
    p2x, p2y, t2 = l2
    d1 = (math.cos(t1), math.sin(t1))
    d2 = (math.cos(t2), math.sin(t2))
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    t = ((p2x - p1x) * d2[1] - (p2y - p1y) * d2[0]) / denom
    return (p1x + t * d1[0], p1y + t * d1[1])
