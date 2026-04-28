"""Court detection & homography.

Padel court (regulamento ufficiale): 20m x 10m, rete a 5m.
Strategia:
1. Estrai linee bianche con Hough transform sul primo frame stabile
2. Filtra per orientamento (orizzontali/verticali rispetto al campo)
3. Trova le 4 intersezioni dei corner del campo
4. Calcola omografia camera → court coords (metri)

Approccio robusto: campionare N frame iniziali, scartare outlier, mediana dei corners.
Per MVP usiamo approccio semplificato. In produzione: modello dedicato (es. court-line
segmentation con UNet) molto più robusto a illuminazione.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Padel court real-world dimensions (meters)
COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0


@dataclass
class CourtCalibration:
    """Homography from image pixels → court coordinates (meters).

    Court coord system: origin at bottom-left corner, X = width (0..10), Y = length (0..20).
    """

    homography: np.ndarray  # 3x3
    corners_px: np.ndarray  # 4x2 in image space
    frame_size: tuple[int, int]  # (w, h)

    def pixel_to_court(self, points_px: np.ndarray) -> np.ndarray:
        """Map Nx2 pixel coords → Nx2 court coords (meters)."""
        if points_px.size == 0:
            return points_px
        pts = points_px.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {
            "homography": self.homography.tolist(),
            "corners_px": self.corners_px.tolist(),
            "frame_size": list(self.frame_size),
        }


def detect_court(frame: np.ndarray) -> CourtCalibration | None:
    """Detect court corners and compute homography.

    Args:
        frame: BGR image.

    Returns:
        CourtCalibration or None if detection failed.

    NOTE: questa è una baseline. Per produzione usare un modello segmentation
    (es. UNet trained su court masks) — molto più robusto a illuminazione,
    pubblico, ombre. Repos come `pabloortegaa/Track-Padel-Match` hanno già
    implementazioni di court detection padel-specifiche.
    """
    h, w = frame.shape[:2]

    # 1. Edge detection sulle linee bianche del campo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # White lines: high luminance threshold
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)

    # 2. Hough lines (probabilistic for segments)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50
    )
    if lines is None or len(lines) < 4:
        return None

    # 3. Per MVP: assume camera fissa dietro al campo → semplifichiamo a
    # rilevare il bounding-quadrilatero più grande delle linee bianche.
    # In produzione: clusterizzare linee per orientamento e trovare le
    # 4 linee di bordo campo, poi intersezioni.
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < (w * h * 0.1):  # campo deve coprire >10% frame
        return None

    # Approssima a 4 corner
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) != 4:
        return None

    corners_px = approx.reshape(4, 2).astype(np.float32)
    corners_px = _order_corners(corners_px)  # TL, TR, BR, BL

    # 4. Homography: image corners → court corners (meters)
    # TL = (0, 20), TR = (10, 20), BR = (10, 0), BL = (0, 0)
    court_corners_m = np.array(
        [[0, COURT_LENGTH_M], [COURT_WIDTH_M, COURT_LENGTH_M], [COURT_WIDTH_M, 0], [0, 0]],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(corners_px, court_corners_m, cv2.RANSAC, 5.0)
    if homography is None:
        return None

    return CourtCalibration(homography=homography, corners_px=corners_px, frame_size=(w, h))


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL (clockwise from top-left)."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array(
        [
            pts[np.argmin(s)],   # TL
            pts[np.argmin(diff)],  # TR
            pts[np.argmax(s)],   # BR
            pts[np.argmax(diff)],  # BL
        ],
        dtype=np.float32,
    )


def calibrate_from_video(video_path: str, sample_frames: int = 30) -> CourtCalibration | None:
    """Sample early frames and pick best calibration.

    Strategia: prova su 30 frame distribuiti nei primi 5 secondi, prendi
    la calibrazione con corners più "stabili" (mediana).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(total, int(fps * 5))
    indices = np.linspace(0, end_frame - 1, sample_frames, dtype=int)

    calibs: list[CourtCalibration] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        c = detect_court(frame)
        if c is not None:
            calibs.append(c)
    cap.release()

    if not calibs:
        return None

    # Median of corners per stabilità
    corners_stack = np.stack([c.corners_px for c in calibs])
    median_corners = np.median(corners_stack, axis=0).astype(np.float32)

    court_corners_m = np.array(
        [[0, COURT_LENGTH_M], [COURT_WIDTH_M, COURT_LENGTH_M], [COURT_WIDTH_M, 0], [0, 0]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(median_corners, court_corners_m, cv2.RANSAC, 5.0)
    return CourtCalibration(homography=H, corners_px=median_corners, frame_size=calibs[0].frame_size)
