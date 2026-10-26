"""Is this video filmed from a camera position already calibrated?

A camera on a tripod behind the same glass, match after match: the court is
where it was, give or take the few pixels by which the tripod went back in a
slightly different place. The calibration saved with that position (a
preset) fits the new video once moved by those pixels.

The keyframe of the preset and the new one are compared on what does not
move — walls, fences, roof, posts, adverts: corner features (ORB) matched
between the two, and the homography that carries most of them. A camera
turned on its tripod maps one picture onto the other exactly by a
homography (on the straightened picture, where the lens bend is removed);
a camera moved a little nearly so. Players are a handful of the features and
fall out as outliers.

The transfer is trusted only when many features agree and the camera has
barely moved: beyond that the background and the floor no longer move
together, and the moved calibration would be a guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.ml.court import distort_points, undistort_points

# Pictures are compared at this width: plenty of detail for a few hundred
# features, fast on the Pi.
WORK_WIDTH = 960
FEATURES = 3000
# Lowe's ratio test.
RATIO = 0.75
# RANSAC tolerance, pixels at WORK_WIDTH.
RANSAC_PX = 3.0
# A match: at least this many features agree, and this share of the
# features matched.
MIN_INLIERS = 80
MIN_INLIER_SHARE = 0.35
# The camera barely moved: the court (its corners' quadrilateral, clipped to
# the picture) shifted by at most this share of the picture's diagonal.
MAX_SHIFT = 0.04


@dataclass(frozen=True)
class CameraMatch:
    homography: np.ndarray       # reference -> new, straightened picture pixels
    inliers: int
    share: float                 # inliers over matched features
    shift: float                 # mean displacement of the court, share of the diagonal

    @property
    def same_position(self) -> bool:
        return self.inliers >= MIN_INLIERS and self.share >= MIN_INLIER_SHARE and self.shift <= MAX_SHIFT

    def summary(self) -> dict:
        return {"inliers": self.inliers, "share": round(self.share, 2), "shift": round(self.shift, 4)}


def _prepare(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Grey picture at the reference frame size, then at WORK_WIDTH; and the
    scale from the frame size to the working picture."""
    if (image.shape[1], image.shape[0]) != tuple(size):
        image = cv2.resize(image, tuple(size), interpolation=cv2.INTER_AREA)
    scale = WORK_WIDTH / float(size[0])
    small = cv2.resize(image, (WORK_WIDTH, int(round(size[1] * scale))), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return cv2.equalizeHist(grey), scale


def match_camera(
    reference: np.ndarray,
    image: np.ndarray,
    frame_size: tuple[int, int],
    corners_px: np.ndarray | list,
    k1: float = 0.0,
) -> CameraMatch | None:
    """How `image` relates to `reference`, both frames of `frame_size`
    (resized to it if needed) of a camera with lens `k1`; `corners_px` are
    the reference's court corners, to measure how far the court moved.
    None when too few features match to say anything."""
    a, scale = _prepare(reference, frame_size)
    b, _ = _prepare(image, frame_size)
    orb = cv2.ORB_create(nfeatures=FEATURES, fastThreshold=12)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < RATIO * rest[0].distance]
    if len(good) < MIN_INLIERS:
        return None
    # Back to frame pixels, then straightened: there a turned camera is an
    # exact homography.
    src = np.float64([ka[m.queryIdx].pt for m in good]) / scale
    dst = np.float64([kb[m.trainIdx].pt for m in good]) / scale
    src = undistort_points(src, k1, frame_size)
    dst = undistort_points(dst, k1, frame_size)
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX / scale)
    if h is None or not np.isfinite(h).all():
        return None
    inliers = int(mask.sum())
    return CameraMatch(h, inliers, inliers / len(good), _court_shift(h, corners_px, k1, frame_size))


def transfer(points_px: np.ndarray | list, found: CameraMatch, k1: float, frame_size: tuple[int, int]) -> np.ndarray:
    """Picture points of the reference (corners, net ends) in the new video."""
    straight = undistort_points(np.asarray(points_px, dtype=np.float64), k1, frame_size)
    moved = cv2.perspectiveTransform(straight.reshape(-1, 1, 2), found.homography).reshape(-1, 2)
    return distort_points(moved, k1, frame_size)


def _court_shift(h: np.ndarray, corners_px, k1: float, frame_size: tuple[int, int]) -> float:
    """Mean displacement, as a share of the diagonal, of points of the court
    inside the picture: a grid over the corners' quadrilateral, clipped."""
    w, hgt = frame_size
    corners = undistort_points(np.asarray(corners_px, dtype=np.float64), k1, frame_size)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    to_quad = cv2.getPerspectiveTransform(unit, corners.astype(np.float32))
    u, v = np.meshgrid(np.linspace(0.05, 0.95, 10), np.linspace(0.05, 0.95, 10))
    grid = cv2.perspectiveTransform(np.c_[u.ravel(), v.ravel()].reshape(-1, 1, 2).astype(np.float32),
                                    to_quad).reshape(-1, 2).astype(np.float64)
    inside = (grid[:, 0] >= 0) & (grid[:, 0] < w) & (grid[:, 1] >= 0) & (grid[:, 1] < hgt)
    if not inside.any():
        return 1.0
    grid = grid[inside]
    moved = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), h).reshape(-1, 2)
    return float(np.linalg.norm(moved - grid, axis=1).mean() / np.hypot(w, hgt))
