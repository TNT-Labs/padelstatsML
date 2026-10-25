"""Foot points of boxes cut by the net (app/ml/footpoint.py)."""
import numpy as np
import pytest

from app.ml.court import CalibrationSource, build_calibration
from app.ml.footpoint import MIN_SAMPLES, FootModel
from test_court import COURT_CORNERS_M_LIST, FRAME, _pinhole


def _scene(height_m=3.5, tilt_deg=12.0, seed=0):
    """A camera, its calibration, and a box maker: the box of a player at
    (x, y), closed on the net's tape when the tape hides the feet."""
    project = _pinhole(height_m, tilt_deg)
    calib = build_calibration([project((x, y, 0.0)) for x, y in COURT_CORNERS_M_LIST], FRAME,
                              CalibrationSource.MANUAL)
    tape = np.array([project((x, 10.0, 0.9)) for x in np.linspace(-2, 12, 200)])
    rng = np.random.default_rng(seed)

    def box(x, y):
        foot = project((x, y, 0.0))
        top = project((x, y, 1.75 * rng.uniform(0.95, 1.04)))[1]
        half = rng.uniform(0.22, 0.38)
        width = project((x + half, y, 0.0))[0] - project((x - half, y, 0.0))[0]
        bottom = foot[1]
        tape_row = np.interp(foot[0], tape[:, 0], tape[:, 1])
        cut = y > 10.0 and foot[1] > tape_row
        if cut:
            bottom = tape_row
        return (foot[0] - width / 2, top, foot[0] + width / 2, bottom), cut

    return calib, box, rng


def _trained(calib, box, rng) -> FootModel:
    model = FootModel(calib)
    for _ in range(200):
        model.learn(box(rng.uniform(0.5, 9.5), rng.uniform(0.5, 9.0))[0])
    return model


def _error(calib, point, x, y) -> float:
    court = calib.floor_points_to_court(np.array([point], np.float32))[0]
    return float(np.hypot(court[0] - x, court[1] - y))


@pytest.mark.parametrize("height_m, tilt_deg", [(3.5, 12.0), (4.0, 18.0)])
def test_feet_hidden_by_the_net_are_placed_from_the_rest_of_the_box(height_m, tilt_deg):
    calib, box, rng = _scene(height_m, tilt_deg)
    model = _trained(calib, box, rng)
    raw, fixed = [], []
    for _ in range(150):
        x, y = rng.uniform(0.5, 9.5), rng.uniform(10.3, 19.5)
        bbox, cut = box(x, y)
        if not cut:
            continue
        raw.append(_error(calib, ((bbox[0] + bbox[2]) / 2, bbox[3]), x, y))
        fixed.append(_error(calib, model.foot(bbox), x, y))
    assert len(raw) > 40
    assert np.mean(raw) > 1.5
    assert np.mean(fixed) < 0.7 and np.mean(fixed) < np.mean(raw) / 3


def test_boxes_reaching_the_feet_are_left_alone():
    calib, box, rng = _scene()
    model = _trained(calib, box, rng)
    for x, y in [(2.0, 3.0), (8.0, 8.5), (5.0, 9.8), (3.0, 19.0), (7.0, 18.0)]:
        bbox, cut = box(x, y)
        assert not cut
        assert model.foot(bbox) == ((bbox[0] + bbox[2]) / 2, bbox[3])


def test_nothing_is_corrected_before_the_match_has_taught_the_camera():
    calib, box, rng = _scene()
    model = FootModel(calib)
    for _ in range(MIN_SAMPLES - 1):
        model.learn(box(rng.uniform(0.5, 9.5), rng.uniform(0.5, 9.0))[0])
    bbox, cut = box(5.0, 14.0)
    assert cut
    assert model.foot(bbox) == ((bbox[0] + bbox[2]) / 2, bbox[3])
    assert model.corrected == 0


def test_a_run_recorded_before_the_correction_is_corrected_on_load(tmp_path):
    from app.ml.artifacts import LoadedArtifacts
    from app.ml.footpoint import FOOT_MODEL_VERSION
    from app.ml.reprocess import with_calibration
    from app.ml.tracking import Observation, Tracklet

    calib, box, rng = _scene()

    def observation(i, x, y):
        bbox, _ = box(x, y)
        foot = ((bbox[0] + bbox[2]) / 2, bbox[3])
        court = calib.floor_points_to_court(np.array([foot], np.float32))[0]
        return Observation(i, i * 0.2, bbox, foot, (float(court[0]), float(court[1])), 0.9, np.zeros(72))

    near = Tracklet(0, [observation(i, rng.uniform(1, 9), rng.uniform(1, 9)) for i in range(100)])
    far = Tracklet(1, [observation(i, 5.0, 14.0) for i in range(5)])
    old = LoadedArtifacts(tmp_path, {}, [near, far], calib, None)

    loaded = with_calibration(old, None)

    assert abs(far.observations[0].foot_court[1] - 14.0) > 1.0
    assert all(abs(o.foot_court[1] - 14.0) < 0.8 for o in loaded.tracklets[1].observations)
    assert [o.foot_court for o in loaded.tracklets[0].observations] == pytest.approx(
        [o.foot_court for o in near.observations], abs=1e-4)
    current = LoadedArtifacts(tmp_path, {"foot_model": FOOT_MODEL_VERSION}, [near, far], calib, None)
    assert with_calibration(current, None) is current
