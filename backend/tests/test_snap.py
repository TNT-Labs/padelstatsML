"""Snap of the calibration onto the painted lines (app/ml/snap.py)."""
import cv2
import numpy as np
import pytest

from app.ml.court import CalibrationError, CalibrationSource, build_calibration, distort_points
from app.ml.snap import PAINTED_LINES, snap_to_lines
from test_court import COURT_CORNERS_M_LIST, FRAME, _pinhole


def _court_picture(project, k1: float, rng) -> np.ndarray:
    """Blue turf, white lines, the net's tape, a bright wall, players and noise."""
    image = np.zeros((FRAME[1], FRAME[0], 3), np.uint8)
    image[:] = (150, 90, 30)
    image[:300] = (170, 170, 170)

    def seen(x, y, z=0.0):
        return distort_points(np.array([project((x, y, z))]), k1, FRAME)[0]

    for a, b, _ in PAINTED_LINES:
        points = np.array([seen(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                           for f in np.linspace(0, 1, 80)], np.int32)
        cv2.polylines(image, [points], False, (235, 235, 235), 4)
    tape = np.array([seen(x, 10.0, 0.9) for x in np.linspace(-0.3, 10.3, 60)], np.int32)
    cv2.polylines(image, [tape], False, (245, 245, 245), 6)
    for _ in range(6):
        fx, fy = seen(rng.uniform(1, 9), rng.uniform(1, 19))
        tall = int(120 * fy / FRAME[1] + 30)
        cv2.rectangle(image, (int(fx - tall * 0.2), int(fy - tall)), (int(fx + tall * 0.2), int(fy)),
                      (40, 40, 200), -1)
    noisy = image.astype(np.int16) + rng.normal(0, 8, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8), seen


def _court_error_m(calib, seen) -> float:
    grid = [(x, y) for x in np.linspace(0.5, 9.5, 7) for y in np.linspace(0.5, 19.5, 15)]
    px = np.array([seen(x, y) for x, y in grid], np.float32)
    visible = (px[:, 0] > 0) & (px[:, 0] < FRAME[0]) & (px[:, 1] > 0) & (px[:, 1] < FRAME[1])
    found = calib.floor_points_to_court(px[visible])
    return float(np.linalg.norm(found - np.array(grid)[visible], axis=1).mean())


@pytest.mark.parametrize("camera, k1", [((3.5, 12.0), 0.0), ((2.0, 3.0), 0.0), ((3.5, 12.0), 0.2)])
def test_corners_placed_roughly_are_snapped_onto_the_lines(camera, k1):
    rng = np.random.default_rng(1)
    image, seen = _court_picture(_pinhole(*camera), k1, rng)
    true = np.array([seen(x, y) for x, y in COURT_CORNERS_M_LIST])
    start = build_calibration(true + rng.normal(0, 25, true.shape), FRAME,
                              CalibrationSource.MANUAL, k1=max(0.0, k1 - 0.04))

    result = snap_to_lines(start, image)

    snapped = build_calibration(result["corners_px"], FRAME, CalibrationSource.MANUAL, k1=result["k1"])
    before, after = _court_error_m(start, seen), _court_error_m(snapped, seen)
    assert before > 0.15
    assert after < 0.12 and after < before / 3
    assert result["improved"] and result["aligned_after"] > result["aligned_before"]
    assert result["after_cm"] is not None and result["after_cm"] < 3
    assert abs(result["k1"] - k1) <= 0.04


def test_a_picture_without_lines_is_refused():
    rng = np.random.default_rng(2)
    image, seen = _court_picture(_pinhole(3.5, 12.0), 0.0, rng)
    true = [seen(x, y) for x, y in COURT_CORNERS_M_LIST]
    blank = np.full_like(image, (150, 90, 30))
    with pytest.raises(CalibrationError, match="linee"):
        snap_to_lines(build_calibration(true, FRAME, CalibrationSource.MANUAL), blank)


def test_an_exact_calibration_stays_where_it_is():
    rng = np.random.default_rng(3)
    image, seen = _court_picture(_pinhole(3.5, 12.0), 0.0, rng)
    true = np.array([seen(x, y) for x, y in COURT_CORNERS_M_LIST])
    calib = build_calibration(true, FRAME, CalibrationSource.MANUAL)

    result = snap_to_lines(calib, image)

    assert result["shift_m"] < 0.1
    assert not result["improved"]
