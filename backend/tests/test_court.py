from __future__ import annotations

import numpy as np
import pytest

from app.ml.court import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    CalibrationError,
    CalibrationSource,
    build_calibration,
    calibration_from_dict,
    suggest_corners,
)

FRAME = (1920, 1080)
GOOD = [[600, 300], [1320, 300], [1800, 980], [120, 980]]


def test_corners_map_to_the_court_rectangle(calibration):
    court = calibration.floor_points_to_court(calibration.corners_px)
    expected = np.array(
        [[0, COURT_LENGTH_M], [COURT_WIDTH_M, COURT_LENGTH_M], [COURT_WIDTH_M, 0], [0, 0]]
    )
    assert np.allclose(court, expected, atol=1e-3)


def test_projection_round_trips(calibration):
    points_m = np.array([[5.0, 10.0], [1.0, 3.0], [9.0, 18.0]], dtype=np.float32)
    back = calibration.floor_points_to_court(calibration.court_to_pixels(points_m))
    assert np.allclose(back, points_m, atol=1e-3)


def test_net_error_is_reported_when_the_net_is_marked():
    # Two points that genuinely sit on the Y = 10 m line, projected to pixels.
    calib = build_calibration(GOOD, FRAME, CalibrationSource.MANUAL)
    on_net = calib.court_to_pixels(np.array([[0.0, NET_Y_M], [COURT_WIDTH_M, NET_Y_M]]))
    with_net = build_calibration(GOOD, FRAME, CalibrationSource.MANUAL, net_px=on_net)
    assert with_net.net_error_m is not None
    assert with_net.net_error_m < 0.01


def test_net_error_catches_misplaced_corners():
    calib = build_calibration(GOOD, FRAME, CalibrationSource.MANUAL)
    # Pretend the user marked the net two metres off.
    off = calib.court_to_pixels(np.array([[0.0, 13.0], [COURT_WIDTH_M, 13.0]]))
    checked = build_calibration(GOOD, FRAME, CalibrationSource.MANUAL, net_px=off)
    assert checked.net_error_m == pytest.approx(3.0, abs=0.05)


@pytest.mark.parametrize(
    "corners",
    [
        [[600, 300], [600, 302], [1800, 980], [120, 980]],      # coincident
        [[600, 300], [1800, 980], [1320, 300], [120, 980]],     # bow-tie order
        [[600, 300], [640, 300], [660, 340], [600, 340]],       # far too small
        [[0, 0], [100, 0], [200, 0], [300, 0]],                 # collinear
    ],
)
def test_degenerate_quads_are_rejected(corners):
    with pytest.raises(CalibrationError):
        build_calibration(corners, FRAME, CalibrationSource.MANUAL)


def test_corner_order_is_not_silently_repaired():
    """The old implementation re-sorted corners by pixel sum/difference, which
    mirrored the court on angled cameras. Wrong order must raise, not guess."""
    reversed_order = [GOOD[0], GOOD[3], GOOD[2], GOOD[1]]
    with pytest.raises(CalibrationError):
        build_calibration(reversed_order, FRAME, CalibrationSource.MANUAL)


def test_roundtrip_through_dict(calibration):
    restored = calibration_from_dict(
        {
            "corners_px": calibration.corners_px.tolist(),
            "frame_size": list(calibration.frame_size),
            "source": calibration.source.value,
        }
    )
    assert np.allclose(restored.homography, calibration.homography, atol=1e-6)


def test_roi_covers_the_court_and_leaves_headroom(calibration):
    x1, y1, x2, y2 = calibration.roi_px()
    corners = calibration.corners_px
    assert x1 <= corners[:, 0].min() and x2 >= corners[:, 0].max()
    assert y2 >= corners[:, 1].max()
    assert y1 < corners[:, 1].min()          # room above the floor plane
    assert 0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080


def test_suggestion_always_returns_four_usable_corners():
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    suggestion = suggest_corners(blank)
    assert suggestion.corners_px.shape == (4, 2)
    assert suggestion.method in {"lines", "default"}
    # A blank frame has no lines, so the geometric default must take over and
    # must never masquerade as a confident detection.
    assert suggestion.method == "default"
    assert suggestion.confidence == 0.0
