"""Filtered paths: distance and speed without the detector's jitter."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.court import CalibrationSource, build_calibration
from app.ml.smoothing import smooth_path
from app.ml.tracking import Observation
from tests.test_court import COURT_CORNERS_M_LIST, FRAME, _pinhole

HZ = 5.0


def _camera(height_m=3.5, tilt_deg=12.0):
    project = _pinhole(height_m, tilt_deg)
    calib = build_calibration([project((x, y, 0.0)) for x, y in COURT_CORNERS_M_LIST], FRAME,
                              CalibrationSource.MANUAL)
    return project, calib


def _observe(project, calib, path, rng, jitter=True):
    """Detections of a player following `path` [(t, x, y)], with the box's
    bottom edge jittering as a detector's does: 1.5% / 2% of its height."""
    out = []
    for t, x, y in path:
        foot = np.array(project((x, y, 0.0)))
        head = np.array(project((x, y, 1.75)))
        height = foot[1] - head[1]
        if jitter:
            foot = foot + rng.normal(0.0, [0.015 * height, 0.02 * height])
        court = calib.floor_points_to_court(foot[None])[0]
        out.append(Observation(
            int(round(t * 30)), float(t),
            (foot[0] - 0.2 * height, foot[1] - height, foot[0] + 0.2 * height, foot[1]),
            (float(foot[0]), float(foot[1])), (float(court[0]), float(court[1])),
            0.9, np.full(72, 1 / 72, np.float32),
        ))
    return out


def _raw_distance(observations):
    steps = [np.hypot(b.foot_court[0] - a.foot_court[0], b.foot_court[1] - a.foot_court[1])
             for a, b in zip(observations, observations[1:])]
    return float(np.sum(steps))


def _shuttle(t0, t1, y, speed=3.0):
    """Side to side across the court at `speed`, sampled at HZ."""
    path, x, direction = [], 1.0, 1.0
    for t in np.arange(t0, t1, 1.0 / HZ):
        path.append((float(t), x, y))
        x += direction * speed / HZ
        if not 1.0 <= x <= 9.0:
            direction = -direction
            x = float(np.clip(x, 1.0, 9.0))
    return path


@pytest.mark.parametrize("height_m, tilt_deg, y", [(3.5, 12.0, 4.0), (3.5, 12.0, 16.0), (1.9, 3.0, 16.0)])
def test_a_player_standing_still_covers_almost_no_ground(height_m, tilt_deg, y):
    """Summed raw, the jitter of a minute standing still was tens of metres."""
    project, calib = _camera(height_m, tilt_deg)
    rng = np.random.default_rng(0)
    observations = _observe(project, calib, [(t, 5.0, y) for t in np.arange(0, 60, 1 / HZ)], rng)

    lengths, _ = smooth_path(observations, calib).steps()

    assert _raw_distance(observations) > 25.0
    assert lengths.sum() < 10.0


@pytest.mark.parametrize("y", [4.0, 16.0])
def test_distance_run_is_measured_within_a_few_percent(y):
    project, calib = _camera()
    rng = np.random.default_rng(1)
    path = _shuttle(0.0, 60.0, y)
    true = sum(np.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(path, path[1:]))
    observations = _observe(project, calib, path, rng)

    smoothed = smooth_path(observations, calib)
    lengths, _ = smoothed.steps()

    assert lengths.sum() == pytest.approx(true, rel=0.06)
    # The speed is the one run, not the jitter's.
    assert np.percentile(smoothed.speeds, 95) == pytest.approx(3.0, abs=0.6)


def test_a_short_gap_is_bridged_and_a_long_one_is_not():
    project, calib = _camera()
    rng = np.random.default_rng(2)
    path = [p for p in _shuttle(0.0, 30.0, 6.0, speed=2.0) if not 10.0 <= p[0] < 11.0]
    path = [p for p in path if not 20.0 <= p[0] < 24.0]

    smoothed = smooth_path(_observe(project, calib, path, rng), calib)

    # One second missing is bridged; four seconds start a new path.
    assert len(np.unique(smoothed.segment)) == 2
    lengths, ends = smoothed.steps()
    bridged = lengths[(ends > 10.9) & (ends < 11.1)]
    assert bridged.size == 1 and bridged[0] > 1.0


def test_a_detection_no_one_could_reach_is_left_out():
    project, calib = _camera()
    rng = np.random.default_rng(3)
    observations = _observe(project, calib, [(t, 5.0, 5.0) for t in np.arange(0, 20, 1 / HZ)], rng)
    # Another person's box, 8 m away, for one sample.
    wrong = _observe(project, calib, [(10.0, 5.0, 13.0)], rng, jitter=False)[0]
    observations = [wrong if abs(o.timestamp_s - 10.0) < 1e-6 else o for o in observations]

    smoothed = smooth_path(observations, calib)

    assert smoothed.outliers == 1
    assert np.abs(smoothed.positions[:, 1] - 5.0).max() < 0.5


def test_a_track_that_moves_to_someone_else_starts_a_new_path():
    """Several detections in a row far from the path: not jitter, a
    different place — the path starts again there instead of discarding
    the rest of the track."""
    project, calib = _camera()
    rng = np.random.default_rng(4)
    path = [(t, 5.0, 5.0) for t in np.arange(0, 10, 1 / HZ)] + [(t, 5.0, 14.0) for t in np.arange(10, 20, 1 / HZ)]

    smoothed = smooth_path(_observe(project, calib, path, rng), calib)

    assert smoothed.outliers == 0
    assert len(np.unique(smoothed.segment)) == 2
    lengths, _ = smoothed.steps()
    assert lengths.sum() < 6.0          # the 9 m jump is not distance run


def test_feet_estimated_under_the_net_are_trusted_as_much_as_they_deserve():
    """A far player behind the net, camera at 1.9 m: the box ends on the tape
    and the feet are estimated from its size, a few per cent off from one
    frame to the next — metres in depth. Trusted like a seen foot, standing
    still they ran tens of metres; a player shuttling across the court keeps
    their distance."""
    project, calib = _camera(1.9, 3.0)
    rng = np.random.default_rng(2)

    def estimated(path):
        out = []
        for t, x, y in path:
            foot = np.array(project((x, y, 0.0)))
            horizon = calib.horizon_y(float(foot[0]))
            # Depth estimated within ~12%: the row's distance to the horizon.
            row = horizon + (foot[1] - horizon) * (1 + rng.normal(0, 0.12))
            seen = foot[1] - 0.5 * (foot[1] - horizon)          # the box ends on the tape
            court = calib.floor_points_to_court(np.array([[foot[0], row]]))[0]
            out.append(Observation(
                int(round(t * 30)), float(t), (foot[0] - 8, seen - 40, foot[0] + 8, seen),
                (float(foot[0]), float(row)), (float(court[0]), float(court[1])),
                0.9, np.full(72, 1 / 72, np.float32),
            ))
        return out

    still = estimated([(t, 5.0, 16.0) for t in np.arange(0, 60, 1 / HZ)])
    assert _raw_distance(still) > 100.0
    assert smooth_path(still, calib).steps()[0].sum() < 15.0

    shuttle = _shuttle(0, 60, 16.0)
    run = sum(np.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(shuttle, shuttle[1:]))
    measured = smooth_path(estimated(shuttle), calib).steps()[0].sum()
    assert 0.75 * run < measured < 1.25 * run
