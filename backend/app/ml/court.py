"""Court detection & homography calibration.

Padel court (regulation): 20 m × 10 m, net at 10 m.

Detection strategies (tried in order, first success wins):
  1. White-line Hough detection  — works on well-lit outdoor/indoor courts
  2. Court-color segmentation    — works on distinctly blue/green courts
  3. Estimated corners           — last resort; rough estimate from frame size
                                   keeps the pipeline alive on difficult videos

For production quality, replace with a UNet court-segmentation model trained
on padel-specific data (e.g. pabloortegaa/Track-Padel-Match).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

COURT_WIDTH_M  = 10.0
COURT_LENGTH_M = 20.0

# Real-world court corners in metres (TL, TR, BR, BL)
_COURT_CORNERS_M = np.array(
    [[0, COURT_LENGTH_M], [COURT_WIDTH_M, COURT_LENGTH_M], [COURT_WIDTH_M, 0], [0, 0]],
    dtype=np.float32,
)


@dataclass
class CourtCalibration:
    """Homography: image pixels → court coordinates (metres).

    Court coord system: origin = bottom-left corner of court,
    X axis = width (0 → 10 m), Y axis = length (0 → 20 m).
    """
    homography: np.ndarray   # 3×3
    corners_px: np.ndarray   # 4×2, image space
    frame_size: tuple[int, int]  # (w, h)
    estimated: bool = False  # True when falling back to guessed corners

    def pixel_to_court(self, points_px: np.ndarray) -> np.ndarray:
        """Map Nx2 pixel coords → Nx2 court coords (metres)."""
        if points_px.size == 0:
            return points_px
        pts = points_px.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "homography":  self.homography.tolist(),
            "corners_px":  self.corners_px.tolist(),
            "frame_size":  list(self.frame_size),
            "estimated":   self.estimated,
        }


# ── Detection strategies ─────────────────────────────────────────────────────

def _make_calibration(corners: np.ndarray, w: int, h: int, estimated: bool = False) -> CourtCalibration | None:
    corners = _order_corners(corners)
    H, mask = cv2.findHomography(corners, _COURT_CORNERS_M, cv2.RANSAC, 5.0)
    if H is None:
        return None
    return CourtCalibration(homography=H, corners_px=corners, frame_size=(w, h), estimated=estimated)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points as TL, TR, BR, BL (clockwise from top-left)."""
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def _detect_by_white_lines(frame: np.ndarray) -> CourtCalibration | None:
    """Strategy 1: Hough lines on white-thresholded edges."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(white, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=80, maxLineGap=30)
    if lines is None or len(lines) < 4:
        return None

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep only large enough contours (court should cover >5% of frame)
    valid = [c for c in contours if cv2.contourArea(c) > w * h * 0.05]
    if not valid:
        return None

    largest = max(valid, key=cv2.contourArea)

    # Try increasingly coarse approximations until we get exactly 4 corners
    for eps in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        approx = cv2.approxPolyDP(largest, eps * cv2.arcLength(largest, True), True)
        if len(approx) == 4:
            return _make_calibration(approx.reshape(4, 2).astype(np.float32), w, h)

    # Fall back to min-area rectangle of the largest contour
    rect = cv2.minAreaRect(largest)
    box  = cv2.boxPoints(rect).astype(np.float32)
    if cv2.contourArea(largest) > w * h * 0.10:
        return _make_calibration(box, w, h)

    return None


def _detect_by_court_color(frame: np.ndarray) -> CourtCalibration | None:
    """Strategy 2: HSV segmentation for blue/green court surfaces."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue courts (H 90-130)
    blue  = cv2.inRange(hsv, (90, 30, 40), (130, 220, 220))
    # Green courts (H 35-85)
    green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 210))

    mask = cv2.bitwise_or(blue, green)

    # Morphological cleanup to fill gaps and remove noise
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < w * h * 0.15:
        return None

    rect = cv2.minAreaRect(largest)
    box  = cv2.boxPoints(rect).astype(np.float32)
    return _make_calibration(box, w, h)


def _estimate_corners(frame: np.ndarray) -> CourtCalibration:
    """Strategy 3: estimate corners from frame proportions.

    Assumes the camera is positioned behind a baseline at elevation (~3 m)
    with the full court visible — the standard smartphone recording setup.
    This gives geometrically incorrect coordinates but lets the pipeline run.
    The `estimated=True` flag lets callers know to weight these results lower.
    """
    h, w = frame.shape[:2]

    # Perspective ratios empirically tuned for typical padel camera angle
    # Bottom edge: wide (close to camera), top edge: narrower (far side)
    bx = int(w * 0.06)
    tx = int(w * 0.22)
    ty = int(h * 0.10)
    by = int(h * 0.93)

    corners = np.array(
        [[tx, ty], [w - tx, ty], [w - bx, by], [bx, by]], dtype=np.float32
    )
    cal = _make_calibration(corners, w, h, estimated=True)
    # _make_calibration can only return None if homography fails which is
    # unlikely for these well-spread corners; fall back to identity-ish
    if cal is None:
        H = np.eye(3, dtype=np.float32)
        cal = CourtCalibration(homography=H, corners_px=corners,
                               frame_size=(w, h), estimated=True)
    return cal


def detect_court(frame: np.ndarray) -> CourtCalibration | None:
    """Try all detection strategies, return first success (or None).

    None means even the estimated fallback failed — should not happen in
    practice since _estimate_corners always returns something.
    """
    result = _detect_by_white_lines(frame)
    if result is not None:
        return result

    result = _detect_by_court_color(frame)
    if result is not None:
        return result

    return None  # calibrate_from_video will call _estimate_corners


def calibrate_from_video(video_path: str, sample_frames: int = 30) -> CourtCalibration | None:
    """Sample early frames and return the most consistent calibration.

    Uses median of detected corners across multiple frames for robustness.
    Falls back to estimated corners if no frame produces a detection.
    """
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample from the first 5 s (players in position, no ball motion yet)
    end_frame = min(total, int(fps * 5))
    if end_frame < 1:
        end_frame = min(total, sample_frames)
    indices = np.linspace(0, max(end_frame - 1, 0), sample_frames, dtype=int)

    calibs_reliable: list[CourtCalibration] = []
    fallback_frame: np.ndarray | None = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        if fallback_frame is None:
            fallback_frame = frame.copy()
        c = detect_court(frame)
        if c is not None and not c.estimated:
            calibs_reliable.append(c)

    cap.release()

    if calibs_reliable:
        # Median corners across reliable detections for outlier robustness
        corners_stack = np.stack([c.corners_px for c in calibs_reliable])
        median_corners = np.median(corners_stack, axis=0).astype(np.float32)
        w, h = calibs_reliable[0].frame_size
        cal = _make_calibration(median_corners, w, h, estimated=False)
        if cal is not None:
            return cal

    # No reliable detection — use estimated corners from a sample frame
    if fallback_frame is not None:
        return _estimate_corners(fallback_frame)

    return None
