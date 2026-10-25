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
COURT_CORNERS_M_LIST = [(0.0, 20.0), (10.0, 20.0), (10.0, 0.0), (0.0, 0.0)]


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


def test_corners_outside_the_frame_are_extrapolated_exactly():
    """A close camera leaves the near corners out of the picture: the user
    places them where the visible lines meet, and the court is still
    measured in full."""
    off_frame = [[600, 300], [1320, 300], [2150, 1250], [-230, 1250]]
    calib = build_calibration(off_frame, FRAME, CalibrationSource.MANUAL)
    court = calib.floor_points_to_court(np.array(off_frame, dtype=np.float32))
    expected = [[0, COURT_LENGTH_M], [COURT_WIDTH_M, COURT_LENGTH_M], [COURT_WIDTH_M, 0], [0, 0]]
    assert np.allclose(court, expected, atol=1e-3)
    x1, y1, x2, y2 = calib.roi_px()
    assert 0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080


def test_a_corner_far_beyond_the_frame_is_refused():
    with pytest.raises(CalibrationError, match="lontano"):
        build_calibration(
            [[600, 300], [1320, 300], [1800, 980], [-2500, 980]], FRAME, CalibrationSource.MANUAL
        )


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


def _pinhole(height_m: float, tilt_deg: float, back_m: float = 6.0, f: float = 1100.0):
    """A camera `back_m` behind the near baseline at `height_m`, looking down
    the court tilted down by `tilt_deg`: returns world (x, y, z) -> pixel."""
    tilt = np.radians(tilt_deg)
    centre = np.array([5.0, -back_m, height_m])
    forward = np.array([0.0, np.cos(tilt), -np.sin(tilt)])
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(forward, right)

    def project(point):
        d = np.asarray(point, dtype=float) - centre
        z = d @ forward
        return (960.0 + f * (d @ right) / z, 540.0 + f * (d @ down) / z)

    return project


@pytest.mark.parametrize("height_m, tilt_deg", [(1.7, 0.0), (1.7, 5.0), (3.5, 12.0), (4.0, 18.0)])
def test_the_camera_height_is_read_from_the_players_boxes(height_m, tilt_deg):
    from app.ml.court import estimate_camera_height

    project = _pinhole(height_m, tilt_deg)
    corners = [project((x, y, 0.0)) for x, y in COURT_CORNERS_M_LIST]
    calib = build_calibration(corners, FRAME, CalibrationSource.MANUAL)
    rng = np.random.default_rng(1)
    boxes = []
    for _ in range(200):
        x, y = rng.uniform(0.5, 9.5), rng.uniform(0.5, 19.5)
        # Players are 1.6-1.9 m tall, racket and stance included.
        tall = rng.uniform(1.6, 1.9)
        foot, head = project((x, y, 0.0)), project((x, y, tall))
        left, right = project((x - 0.25, y, 0.0)), project((x + 0.25, y, 0.0))
        boxes.append((left[0], head[1], right[0], foot[1]))

    estimate = estimate_camera_height(calib, boxes)

    assert estimate is not None
    assert abs(estimate - height_m) / height_m < 0.15, estimate


def test_lens_correction_round_trips():
    from app.ml.court import distort_points, undistort_points

    points = np.array([[0, 0], [960, 540], [1900, 50], [-400, 1300], [2300, 900]], dtype=np.float64)
    for k1 in (0.0, 0.1, 0.35):
        back = undistort_points(distort_points(points, k1, FRAME), k1, FRAME)
        assert np.allclose(back, points, atol=1e-6)
    # Barrel distortion pulls the edges towards the centre; the centre stays.
    bent = distort_points(points, 0.2, FRAME)
    assert np.allclose(bent[1], [960, 540])
    assert np.linalg.norm(bent[2] - [960, 540]) < np.linalg.norm(points[2] - [960, 540])


def test_a_calibration_through_a_wide_angle_lens_places_points_right():
    """Seen through a wide-angle lens the court's lines are curves: a plain
    homography misplaces the middle of the court; with the lens correction
    every point of the court lands where it is."""
    from app.ml.court import distort_points

    k1 = 0.25
    project = _pinhole(2.0, 3.0)
    seen = lambda x, y: distort_points(np.array([project((x, y, 0.0))]), k1, FRAME)[0]
    corners = [seen(x, y) for x, y in COURT_CORNERS_M_LIST]
    plain = build_calibration(corners, FRAME, CalibrationSource.MANUAL)
    lens = build_calibration(corners, FRAME, CalibrationSource.MANUAL, k1=k1)

    probes = [(5.0, 3.05), (0.0, 10.0), (10.0, 10.0), (5.0, 16.95), (2.0, 7.0)]
    pixels = np.array([seen(x, y) for x, y in probes], dtype=np.float32)
    plain_error = np.linalg.norm(plain.floor_points_to_court(pixels) - probes, axis=1).max()
    lens_error = np.linalg.norm(lens.floor_points_to_court(pixels) - probes, axis=1).max()
    assert plain_error > 0.3
    assert lens_error < 0.02
    # And back: court to picture follows the curve.
    assert np.allclose(lens.court_to_pixels(np.array(probes, np.float32)), pixels, atol=0.5)
    assert calibration_from_dict({**lens.to_dict(), "k1": k1}).k1 == k1
