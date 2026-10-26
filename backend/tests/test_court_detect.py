"""Automatic calibration proposal from the net and the service line
(app/ml/court_detect.py), on a real court."""
from pathlib import Path

import cv2
import pytest
import numpy as np

from app.ml.court import (
    DERIVED_MAX_OFF_FRAME,
    CalibrationSource,
    build_calibration,
    build_from_reference,
    suggest_corners,
)
from app.ml.court_detect import detect_net_service

FIXTURE = Path(__file__).parent / "fixtures" / "court_low_camera.jpg"
FRAME = (1024, 576)
# Calibration of that court measured by hand (net posts, centre line, near
# service line, side wall): camera ~2 m high, close behind the back glass,
# near corners far out of the picture.
TRUE_CORNERS = [[335.28, 284.49], [608.17, 292.34], [1969.99, 805.94], [-1485.24, 838.54]]


def _court_error_m(calib) -> np.ndarray:
    truth = build_calibration(TRUE_CORNERS, FRAME, CalibrationSource.MANUAL, max_off_frame=DERIVED_MAX_OFF_FRAME)
    xs, ys = np.meshgrid(np.linspace(0.5, 9.5, 7), np.linspace(0.5, 19.5, 15))
    court = np.c_[xs.ravel(), ys.ravel()].astype(np.float32)
    px = truth.court_to_pixels(court)
    visible = (px[:, 0] >= 0) & (px[:, 0] < FRAME[0]) & (px[:, 1] >= 0) & (px[:, 1] < FRAME[1])
    return np.linalg.norm(calib.floor_points_to_court(px[visible]) - court[visible], axis=1)


def test_the_net_and_the_service_line_of_a_real_court_are_found():
    found = detect_net_service(cv2.imread(str(FIXTURE)))
    assert found is not None and found.confidence > 0
    errors = _court_error_m(build_from_reference(found.handles, "net_service", FRAME))
    assert errors.mean() < 0.4 and errors.max() < 1.2


def test_the_suggestion_uses_it_and_falls_back_on_nothing():
    suggestion = suggest_corners(cv2.imread(str(FIXTURE)))
    assert suggestion.method == "net_service" and suggestion.reference == "net_service"
    blank = np.full((FRAME[1], FRAME[0], 3), (150, 90, 30), np.uint8)
    assert suggest_corners(blank).method == "default"
    assert detect_net_service(blank) is None


def test_a_picture_without_a_court_proposes_nothing():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (FRAME[1], FRAME[0], 3), dtype=np.uint8)
    assert detect_net_service(noise) is None


def _synthetic_court(height_m: float, tilt_deg: float, back_m: float, focal: float):
    """A court seen from behind the back glass: turf, the surround outside the
    walls, painted service and centre lines, a see-through net with its tape,
    bottom cord and posts, the walls' bars. Returns the picture and the true
    calibration."""
    from test_court import _pinhole

    size = (1280, 720)
    camera = _pinhole(height_m, tilt_deg, back_m=back_m, f=focal)

    def at(x, y, z=0.0):
        u, v = camera((x, y, z))
        return np.array([u - 960 + size[0] / 2, v - 540 + size[1] / 2])

    image = np.full((size[1], size[0], 3), (200, 200, 205), np.uint8)

    def fill(points, colour):
        cv2.fillPoly(image, [np.array([at(*p) for p in points], np.int32)], colour)

    def line(a, b, colour, width):
        cv2.line(image, tuple(np.int32(at(*a))), tuple(np.int32(at(*b))), colour, width)

    fill([(-3, -0.5), (13, -0.5), (13, 25), (-3, 25)], (60, 70, 170))
    fill([(0, -0.5), (10, -0.5), (10, 20), (0, 20)], (150, 90, 30))
    for a, b in (((0, 3.05), (10, 3.05)), ((0, 16.95), (10, 16.95)), ((5, 3.05), (5, 16.95))):
        line(a, b, (235, 235, 235), 3)
    net = np.zeros_like(image)
    cv2.fillPoly(net, [np.array([at(*p) for p in ((0, 10, 0), (10, 10, 0), (10, 10, 0.9), (0, 10, 0.9))],
                                np.int32)], (1, 1, 1))
    image = np.where(net > 0, (image * 0.75).astype(np.uint8), image)
    line((0, 10, 0.02), (10, 10, 0.02), (20, 20, 20), 2)
    line((0, 10, 0.88), (10, 10, 0.88), (245, 245, 245), 5)
    for x in (0, 10):
        line((x, 10, 0), (x, 10, 1.0), (40, 40, 40), 6)
        for y in (2, 6, 14, 18):
            line((x, y, 0), (x, y, 3), (50, 50, 50), 3)
    noise = np.random.default_rng(0).normal(0, 6, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    truth = build_calibration([at(x, y) for x, y in ((0, 20), (10, 20), (10, 0), (0, 0))], size,
                              CalibrationSource.MANUAL, max_off_frame=DERIVED_MAX_OFF_FRAME)
    return image, truth, size


@pytest.mark.parametrize("camera", [(2.0, 3.0, 1.8, 700.0), (3.5, 12.0, 4.0, 1000.0)])
def test_cameras_low_and_high_are_calibrated_from_the_net(camera):
    image, truth, size = _synthetic_court(*camera)
    found = detect_net_service(image)
    assert found is not None
    calib = build_from_reference(found.handles, "net_service", size)
    xs, ys = np.meshgrid(np.linspace(0.5, 9.5, 7), np.linspace(0.5, 19.5, 15))
    court = np.c_[xs.ravel(), ys.ravel()].astype(np.float32)
    px = truth.court_to_pixels(court)
    visible = (px[:, 0] >= 0) & (px[:, 0] < size[0]) & (px[:, 1] >= 0) & (px[:, 1] < size[1])
    errors = np.linalg.norm(calib.floor_points_to_court(px[visible]) - court[visible], axis=1)
    assert errors.mean() < 0.3 and errors.max() < 0.7
