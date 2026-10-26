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
    never becomes a calibration on its own. (A video from a camera position
    already calibrated is calibrated by itself — app.ml.camera_match — from
    a calibration a human confirmed.)
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

# How far outside the frame a corner may lie, as a fraction of the frame size
# on each side. A camera close to the court often leaves a corner out of the
# picture: the user places it where the visible lines would meet, which the
# homography handles exactly. Beyond this the corner is a guess, not an
# extrapolation of lines that can be seen.
MAX_OFF_FRAME = 1.0
# Corners the system computes — from other reference points, a refinement,
# a stored calibration — may lie further out: with the camera low and close
# behind the back glass the near corners are a frame width or more outside,
# and only their placement by hand would be a guess. This only catches
# nonsense.
DERIVED_MAX_OFF_FRAME = 6.0

# What the user places the four handles on. "corners": the court's corners.
# "net_service": the foot of the two net posts and the two ends of the near
# service line — points in, or close to, the picture when the corners are
# far out of it. Same clockwise order on screen as the corners.
REFERENCE_POINTS_M = {
    "corners": COURT_CORNERS_M,
    "net_service": np.array(
        [
            [0.0,           NET_Y_M],                          # left net post
            [COURT_WIDTH_M, NET_Y_M],                          # right net post
            [COURT_WIDTH_M, NET_Y_M - SERVICE_LINE_OFFSET_M],  # near service line, right end
            [0.0,           NET_Y_M - SERVICE_LINE_OFFSET_M],  # near service line, left end
        ],
        dtype=np.float32,
    ),
}
REFERENCE_LABELS = {
    "corners": CORNER_LABELS,
    "net_service": ("Palo rete sinistro", "Palo rete destro", "Servizio vicino destra", "Servizio vicino sinistra"),
}


# ── Lens distortion ──────────────────────────────────────────────────────────
#
# A wide-angle lens (a phone's 0.6x, an action camera) bends straight lines
# outwards, most at the edges of the picture — where, with the camera close
# to the court, the corners are. A homography cannot follow a bent line, so
# the picture is first straightened with one radial coefficient:
#
#     p_straight = c + (p - c) * (1 + k1 * r^2),   r = |p - c| / R
#
# c the centre of the frame, R half its diagonal. k1 = 0 is no correction;
# barrel distortion, the wide-angle kind, needs a positive k1. Corners and
# boxes stay in the picture as it is; only the step to metres straightens.

MAX_K1 = 0.5


def _lens_frame(frame_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    w, h = float(frame_size[0]), float(frame_size[1])
    return np.array([w / 2.0, h / 2.0]), 0.5 * float(np.hypot(w, h))


def undistort_points(points_px, k1: float, frame_size: tuple[int, int]) -> np.ndarray:
    """Picture pixels to the pixels of a lens without distortion."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if not k1:
        return pts
    centre, radius = _lens_frame(frame_size)
    offset = pts - centre
    r2 = np.sum((offset / radius) ** 2, axis=1, keepdims=True)
    return centre + offset * (1.0 + k1 * r2)


def distort_points(points_px, k1: float, frame_size: tuple[int, int]) -> np.ndarray:
    """The inverse: where a straight-lens pixel lands in the picture."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if not k1:
        return pts
    centre, radius = _lens_frame(frame_size)
    offset = pts - centre
    ru = np.linalg.norm(offset, axis=1) / radius
    rd = ru.copy()
    for _ in range(30):                 # Newton on rd + k1 rd^3 = ru (monotone for k1 >= 0)
        rd -= (rd + k1 * rd ** 3 - ru) / (1.0 + 3.0 * k1 * rd ** 2)
    scale = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
    return centre + offset * scale[:, None]


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
    # Lens distortion (see above): the homography maps straightened pixels.
    k1: float = 0.0

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
        if self.k1:
            pts = undistort_points(pts, self.k1, self.frame_size).astype(np.float32)
        pts = pts.reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def court_to_pixels(self, points_m: np.ndarray) -> np.ndarray:
        """Map Nx2 court coordinates back to image pixels (for overlays)."""
        pts = np.asarray(points_m, dtype=np.float32)
        if pts.size == 0:
            return pts.reshape(-1, 2)
        pts = pts.reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.homography_inv).reshape(-1, 2)
        if self.k1:
            out = distort_points(out, self.k1, self.frame_size).astype(np.float32)
        return out

    # ── Helpers ──────────────────────────────────────────────────────────────

    def horizon_y(self, x: float) -> float | None:
        """Image row of the floor's horizon at column x: where the floor
        plane runs to infinity. None when the horizon is vertical in the
        picture (no real camera position does that)."""
        h31, h32, h33 = (float(v) for v in self.homography[2])
        if abs(h32) < 1e-12:
            return None
        return -(h31 * x + h33) / h32

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

    def far_roi_px(self, margin_m: float = 1.0, player_height_m: float = 2.3) -> tuple[int, int, int, int] | None:
        """The far half of the court with room above it for the players,
        for a second, closer look of the detector (see pipeline): there the
        players are a few dozen pixels tall, and the whole-court crop shrinks
        them further before the network sees them.

        None when the far half is not much smaller than the whole court in
        the picture: cropping it would enlarge nobody."""
        w, h = self.frame_size
        floor = self.court_to_pixels(np.array([
            [-margin_m, NET_Y_M - margin_m],
            [COURT_WIDTH_M + margin_m, NET_Y_M - margin_m],
            [COURT_WIDTH_M + margin_m, COURT_LENGTH_M + margin_m],
            [-margin_m, COURT_LENGTH_M + margin_m],
        ], dtype=np.float32))
        if not np.isfinite(floor).all():
            return None
        # Players stand above the floor: about their height in the
        # picture's scale at the far baseline.
        baseline = self.court_to_pixels(np.array([[0.0, COURT_LENGTH_M], [COURT_WIDTH_M, COURT_LENGTH_M]], np.float32))
        per_metre = float(np.linalg.norm(baseline[1] - baseline[0])) / COURT_WIDTH_M
        x1 = int(np.clip(floor[:, 0].min(), 0, w - 1))
        x2 = int(np.clip(floor[:, 0].max(), 1, w))
        y2 = int(np.clip(floor[:, 1].max(), 1, h))
        y1 = int(np.clip(floor[:, 1].min() - player_height_m * per_metre, 0, h - 1))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        rx1, ry1, rx2, ry2 = self.roi_px()
        # The network sees the longer side at a fixed size: the gain is how
        # much shorter the far crop's longer side is than the whole crop's.
        if max(x2 - x1, y2 - y1) > 0.75 * max(rx2 - rx1, ry2 - ry1):
            return None
        return x1, y1, x2, y2

    def far_player_input_px(self, input_size: int, player_height_m: float = 1.75) -> float:
        """How tall a player on the far baseline reaches the detector when
        it looks at the whole court (roi_px) at `input_size`: the share of
        their height the network has to find them with."""
        baseline = self.court_to_pixels(np.array([[0.0, COURT_LENGTH_M], [COURT_WIDTH_M, COURT_LENGTH_M]], np.float32))
        if not np.isfinite(baseline).all():
            return float("inf")
        per_metre = float(np.linalg.norm(baseline[1] - baseline[0])) / COURT_WIDTH_M
        x1, y1, x2, y2 = self.roi_px()
        scale = input_size / max(x2 - x1, y2 - y1, 1)
        return player_height_m * per_metre * scale

    def to_dict(self) -> dict:
        return {
            "corners_px":  self.corners_px.tolist(),
            "frame_size":  list(self.frame_size),
            "source":      self.source.value,
            "net_error_m": self.net_error_m,
            "homography":  self.homography.tolist(),
            "k1":          self.k1,
        }


# ── Camera height ────────────────────────────────────────────────────────────

# Typical height of a player's box, racket and ready stance included.
PLAYER_BOX_HEIGHT_M = 1.75
# Below this the net hides the feet of the far pair: the detector closes their
# box on the net tape, the foot point lands metres behind the player, and
# their positions are unreliable. The recommended mount is 3-4 m.
LOW_CAMERA_M = 2.3
_CAMERA_MIN_SAMPLES = 50


def estimate_camera_height(calib: CourtCalibration, boxes: list[tuple[float, float, float, float]]) -> float | None:
    """The camera's height above the floor, in metres, from the players' boxes.

    With an upright camera, a person standing on the floor spans in the
    picture the same share of the distance between their feet and the
    floor's horizon as their height is of the camera's: box height /
    (foot row - horizon row) = person height / camera height. The horizon
    comes from the calibration; the median over the match's boxes takes
    care of jumps, crouches and wrong boxes. Boxes cut by the bottom of the
    frame are left out. None with too few usable boxes."""
    _, frame_h = calib.frame_size
    ratios = []
    for x1, y1, x2, y2 in boxes:
        if y2 >= frame_h - 2:
            continue
        horizon = calib.horizon_y((x1 + x2) / 2.0)
        if horizon is None or y2 - horizon < 20:
            continue
        ratios.append((y2 - y1) / (y2 - horizon))
    if len(ratios) < _CAMERA_MIN_SAMPLES:
        return None
    ratio = float(np.median(ratios))
    return PLAYER_BOX_HEIGHT_M / ratio if ratio > 0 else None


# ── Construction & validation ────────────────────────────────────────────────

def build_calibration(
    corners_px: np.ndarray | list,
    frame_size: tuple[int, int],
    source: CalibrationSource = CalibrationSource.MANUAL,
    net_px: np.ndarray | list | None = None,
    k1: float = 0.0,
    max_off_frame: float = MAX_OFF_FRAME,
) -> CourtCalibration:
    """Build and validate a calibration from four user-placed court corners,
    seen through a lens of distortion `k1` (0: none). Corners computed by
    the system rather than placed by hand pass DERIVED_MAX_OFF_FRAME.

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

    k1 = float(k1 or 0.0)
    if not 0.0 <= k1 <= MAX_K1:
        raise CalibrationError("Correzione dell'obiettivo fuori dai limiti.")
    _validate_bounds(pts, w, h, max_off_frame)
    # Shape and perspective are judged on the straightened picture, where the
    # court is a true quadrilateral.
    straight = undistort_points(pts, k1, (w, h)).astype(np.float32)
    _validate_quad(straight, w, h)

    # Exact 4-point solve: with four correspondences the homography is unique,
    # so RANSAC would only add a failure mode.
    homography = cv2.getPerspectiveTransform(straight, COURT_CORNERS_M)
    if homography is None or not np.isfinite(homography).all():
        raise CalibrationError("Impossibile calcolare l'omografia dagli angoli indicati.")

    if not _homography_is_stable(homography, straight):
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
        k1=k1,
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
            k1=k1,
        )
    return calib


def build_from_reference(
    points_px: np.ndarray | list,
    reference: str,
    frame_size: tuple[int, int],
    source: CalibrationSource = CalibrationSource.MANUAL,
    net_px: np.ndarray | list | None = None,
    k1: float = 0.0,
) -> CourtCalibration:
    """A calibration from four handles placed on `reference` points
    (REFERENCE_POINTS_M, in their order): the handles are validated as the
    corners would be, and the corners are computed through them."""
    if reference not in REFERENCE_POINTS_M:
        raise CalibrationError("Riferimento della calibrazione sconosciuto.")
    if reference == "corners":
        return build_calibration(points_px, frame_size, source, net_px, k1)
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise CalibrationError("Servono esattamente 4 punti di riferimento.")
    w, h = int(frame_size[0]), int(frame_size[1])
    k1 = float(k1 or 0.0)
    if w <= 0 or h <= 0 or not 0.0 <= k1 <= MAX_K1:
        raise CalibrationError("Dimensioni del frame o correzione dell'obiettivo non valide.")
    _validate_bounds(pts, w, h, MAX_OFF_FRAME)
    straight = undistort_points(pts, k1, (w, h)).astype(np.float32)
    _validate_quad(straight, w, h, min_area=0.005)
    to_picture = cv2.getPerspectiveTransform(REFERENCE_POINTS_M[reference], straight)
    corners = cv2.perspectiveTransform(COURT_CORNERS_M.reshape(-1, 1, 2), to_picture).reshape(-1, 2)
    if not np.isfinite(corners).all():
        raise CalibrationError("I punti indicati non definiscono una prospettiva valida.")
    return build_calibration(distort_points(corners, k1, (w, h)), (w, h), source, net_px, k1,
                             max_off_frame=DERIVED_MAX_OFF_FRAME)


def reference_points(calib: CourtCalibration, reference: str) -> np.ndarray:
    """Where the handles of `reference` lie in the picture."""
    return calib.court_to_pixels(REFERENCE_POINTS_M[reference]).astype(np.float64)


def fit_corners(
    points_px: np.ndarray | list,
    points_m: np.ndarray | list,
    weights: np.ndarray | list | None = None,
) -> np.ndarray:
    """The four court corners in pixels of the homography that best fits
    several (pixel, metre) pairs: weighted least squares (normalised DLT).

    Four corners fix a homography exactly but give no protection against a
    misplaced one — and a corner out of the picture is placed by eye. More
    points, landmarks on the visible lines, average the errors out. The
    result is expressed as corners again, so every calibration keeps one
    shape: the 4-point homography through them is the fitted one."""
    px = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    m = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    if len(px) < 4 or len(px) != len(m):
        raise CalibrationError("Servono almeno 4 punti per calcolare la calibrazione.")
    w = np.ones(len(px)) if weights is None else np.asarray(weights, dtype=np.float64)

    def normaliser(points):
        centre = points.mean(axis=0)
        scale = np.sqrt(2.0) / max(float(np.mean(np.linalg.norm(points - centre, axis=1))), 1e-9)
        return np.array([[scale, 0, -scale * centre[0]], [0, scale, -scale * centre[1]], [0, 0, 1.0]])

    tm, tp = normaliser(m), normaliser(px)
    hm = np.c_[m, np.ones(len(m))] @ tm.T
    hp = np.c_[px, np.ones(len(px))] @ tp.T
    rows = []
    for (x, y, _), (u, v, _), weight in zip(hm, hp, np.sqrt(w)):
        rows.append(weight * np.array([x, y, 1, 0, 0, 0, -u * x, -u * y, -u]))
        rows.append(weight * np.array([0, 0, 0, x, y, 1, -v * x, -v * y, -v]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    h = vt[-1].reshape(3, 3)                     # metres -> pixels, normalised
    h = np.linalg.inv(tp) @ h @ tm
    corners = np.c_[COURT_CORNERS_M, np.ones(4)] @ h.T
    if np.any(np.abs(corners[:, 2]) < 1e-12):
        raise CalibrationError("I punti indicati non definiscono una prospettiva valida.")
    return (corners[:, :2] / corners[:, 2:3]).astype(np.float64)


def calibration_from_dict(data: dict) -> CourtCalibration:
    """Rebuild a calibration from the JSON stored on the match row."""
    source = CalibrationSource(data.get("source", CalibrationSource.MANUAL.value))
    fw, fh = data["frame_size"]
    # Validated when it was made; corners computed then may lie far out.
    return build_calibration(
        corners_px=data["corners_px"],
        frame_size=(int(fw), int(fh)),
        source=source,
        net_px=data.get("net_px"),
        k1=float(data.get("k1") or 0.0),
        max_off_frame=DERIVED_MAX_OFF_FRAME,
    )


def _validate_bounds(pts: np.ndarray, w: int, h: int, limit: float = MAX_OFF_FRAME) -> None:
    # Off-frame corners are allowed (partly framed court), within reason.
    if (
        (pts[:, 0] < -limit * w).any() or (pts[:, 0] > (1 + limit) * w).any()
        or (pts[:, 1] < -limit * h).any() or (pts[:, 1] > (1 + limit) * h).any()
    ):
        raise CalibrationError(
            "Un angolo è troppo lontano dall'inquadratura: avvicinalo al bordo del fotogramma."
        )


def _validate_quad(pts: np.ndarray, w: int, h: int, min_area: float = 0.02) -> None:
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
    if abs(signed_area) < min_area * w * h:
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
    method: str        # "net_service" | "default"
    confidence: float  # 0.0 for the geometric default
    # What the handles go on (REFERENCE_POINTS_M) and where.
    reference: str = "corners"
    handles_px: np.ndarray | None = None

    def to_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.corners_px]


def suggest_corners(frame: np.ndarray) -> CornerSuggestion:
    """Propose a calibration for the page to pre-fill: from the net and the
    near service line found in `frame` (app.ml.court_detect) — best the
    court without players, the median of frames over the video.

    A padel court has no painted baseline or side line: it is bounded by its
    walls. What a camera behind the back glass does show is the net, the
    service line and the centre line; the side walls where the turf ends.

    When nothing is found — which it is allowed to be — a generic trapezoid
    for the usual "camera behind the baseline, elevated" setup is returned
    with confidence 0. The user drags four handles either way, so a failed
    suggestion costs seconds, not correctness.
    """
    from app.ml.court_detect import detect_net_service

    h, w = frame.shape[:2]
    try:
        found = detect_net_service(frame)
        if found is not None:
            calib = build_from_reference(found.handles, "net_service", (w, h))
            return CornerSuggestion(calib.corners_px.astype(np.float32), "net_service", found.confidence,
                                    "net_service", found.handles)
    except (cv2.error, CalibrationError, np.linalg.LinAlgError):
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
