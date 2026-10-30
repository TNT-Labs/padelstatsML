"""The players' posture: the YOLOv8 pose heads decoded, poses attached to
the detections, the estimator on a stand-in for HailoRT, and the keypoints
through tracking and the artifacts."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from app.ml.detect import Detection
from app.ml.pose import Pose, attach_poses, decode_yolov8_pose, to_frame

IMGSZ = (640, 640)


def _heads(row=10, col=20, stride=16, sides=(2, 4, 2, 4), logit=5.0, probabilities=False):
    """Raw outputs of the three scales with one person in one cell of the
    stride-16 grid: box sides of `sides` bins, keypoints half a stride right
    and below the cell, all visible."""
    outputs = {}
    for s in (8, 16, 32):
        n = 640 // s
        box = np.zeros((n, n, 64), np.float32)
        cls = np.full((n, n, 1), 0.0 if probabilities else -10.0, np.float32)
        kpt = np.zeros((n, n, 51), np.float32)
        if s == stride:
            for side, bins in enumerate(sides):
                box[row, col, side * 16 + bins] = 30.0          # one bin, softmax ~1
            cls[row, col, 0] = 0.9 if probabilities else logit
            kpt[row, col, 0::3] = 0.5      # x: stride * (2*0.5 - 0.5) = +8 from the centre
            kpt[row, col, 1::3] = 0.75     # y: +16
            kpt[row, col, 2::3] = 3.0      # visibility logit
        outputs[f"conv_box_{s}"], outputs[f"conv_cls_{s}"], outputs[f"conv_kpt_{s}"] = box, cls, kpt
    return outputs


def test_one_person_is_decoded_where_it_is():
    poses = decode_yolov8_pose(_heads(), IMGSZ)
    assert len(poses) == 1
    box, score, kpts = poses[0]
    cx, cy = 20.5 * 16, 10.5 * 16
    assert box == pytest.approx([cx - 32, cy - 64, cx + 32, cy + 64], abs=0.1)
    assert score == pytest.approx(1 / (1 + np.exp(-5.0)), abs=1e-3)
    assert kpts.shape == (17, 3)
    assert kpts[:, 0] == pytest.approx(cx + 8, abs=0.01)
    assert kpts[:, 1] == pytest.approx(cy + 16, abs=0.01)
    assert kpts[:, 2] == pytest.approx(1 / (1 + np.exp(-3.0)), abs=1e-3)


def test_scores_already_probabilities_are_not_squashed_again():
    poses = decode_yolov8_pose(_heads(probabilities=True), IMGSZ)
    assert len(poses) == 1 and poses[0][1] == pytest.approx(0.9)


def test_nobody_nothing():
    assert decode_yolov8_pose(_heads(logit=-6.0), IMGSZ) == []


def test_poses_go_back_to_frame_pixels():
    decoded = decode_yolov8_pose(_heads(), IMGSZ)
    ratio, pad = 640 / 1920, (0.0, 140.0)
    pose = to_frame(decoded, ratio, pad)[0]
    x1, y1, x2, y2 = pose.bbox
    assert x1 == pytest.approx((20.5 * 16 - 32) / ratio, abs=0.5)
    assert y1 == pytest.approx((10.5 * 16 - 64 - 140) / ratio, abs=0.5)
    assert pose.keypoints[0, 1] == pytest.approx((10.5 * 16 + 16 - 140) / ratio, abs=0.5)


def test_each_pose_goes_to_the_box_it_overlaps():
    near = Detection((100, 100, 160, 260), 0.9)
    far = Detection((800, 50, 830, 120), 0.6)
    pose_near = Pose((102, 98, 158, 262), 0.8, np.ones((17, 3), np.float32))
    pose_elsewhere = Pose((400, 400, 440, 500), 0.8, np.zeros((17, 3), np.float32))

    out = attach_poses([near, far], [pose_elsewhere, pose_near])
    assert out[0].keypoints is pose_near.keypoints
    assert out[1].keypoints is None
    assert attach_poses([near], []) == [near]


# ── The estimator on a stand-in for HailoRT ──────────────────────────────────

def _fake_runtime(outputs: dict[str, np.ndarray], arch="HAILO8L", written=True):
    mod = types.ModuleType("hailo_platform")
    shapes = {name: array.shape for name, array in outputs.items()}

    class Stream:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

        def set_format_type(self, _t): pass

    class Bindings:
        def __init__(self, buffers):
            self.buffers = buffers

        def input(self): return types.SimpleNamespace(set_buffer=lambda _b: None)

        def output(self, name):
            return types.SimpleNamespace(
                get_buffer=lambda: outputs[name] if written else self.buffers[name])

    class Configured:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def create_bindings(self, output_buffers): return Bindings(output_buffers)
        def run(self, _b, _t): pass

    class Model:
        output_names = list(shapes)
        def set_batch_size(self, _n): pass
        def input(self): return Stream("in", (640, 640, 3))
        def output(self, name=None): return Stream(name, shapes[name])
        def configure(self): return Configured()

    class VDevice:
        @staticmethod
        def create_params(): return types.SimpleNamespace(scheduling_algorithm=None)
        def __init__(self, _p): pass
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def create_infer_model(self, _path): return Model()

        def get_physical_devices(self):
            identity = types.SimpleNamespace(device_architecture=arch)
            return [types.SimpleNamespace(control=types.SimpleNamespace(identify=lambda: identity))]

    mod.VDevice = VDevice
    mod.FormatType = types.SimpleNamespace(UINT8="u8", FLOAT32="f32")
    mod.HailoSchedulingAlgorithm = types.SimpleNamespace(ROUND_ROBIN="rr")
    return mod


@pytest.fixture(autouse=True)
def _fresh_device(monkeypatch):
    import app.ml.detect_hailo as detect_hailo

    monkeypatch.setattr(detect_hailo, "_SHARED", {"device": None, "stack": None, "users": 0})


def test_the_estimator_finds_the_pose_in_the_frame(monkeypatch, tmp_path):
    from app.ml.detect_hailo import HailoPoseEstimator

    hef = tmp_path / "yolov8s_pose_h8l_pi.hef"
    hef.write_bytes(b"hef")
    monkeypatch.setitem(sys.modules, "hailo_platform", _fake_runtime(_heads()))
    estimator = HailoPoseEstimator(hef)
    poses = estimator.estimate(np.zeros((1080, 1920, 3), np.uint8))
    assert estimator.name == "yolov8s_pose_h8l_pi.hef"
    assert len(poses) == 1 and poses[0].keypoints.shape == (17, 3)
    estimator.close()


def test_an_estimator_whose_outputs_stay_unwritten_is_not_used(monkeypatch, tmp_path):
    from app.ml.detect import DetectorError
    from app.ml.detect_hailo import HailoPoseEstimator

    hef = tmp_path / "yolov8s_pose_h8l_pi.hef"
    hef.write_bytes(b"hef")
    monkeypatch.setitem(sys.modules, "hailo_platform", _fake_runtime(_heads(), written=False))
    with pytest.raises(DetectorError, match="Postura non disponibile"):
        HailoPoseEstimator(hef)


def test_the_pose_network_for_the_chip_is_found(monkeypatch, tmp_path):
    from app.ml import accel

    for name in ("yolov8s_pose_h8.hef", "yolov8s_pose_h8l_pi.hef", "yolov8s_h8l.hef"):
        (tmp_path / name).write_bytes(b"hef")
    monkeypatch.setattr(accel, "POSE_FOLDERS", (str(tmp_path),))
    found = accel.pose_models()
    assert [p.split("/")[-1] for p in found] == ["yolov8s_pose_h8.hef", "yolov8s_pose_h8l_pi.hef"]
    assert accel.pick_model(found, "Hailo-8L").endswith("yolov8s_pose_h8l_pi.hef")
    assert accel.pick_model(found, "Hailo-8").endswith("yolov8s_pose_h8.hef")


# ── Through tracking and the artifacts ───────────────────────────────────────

def test_keypoints_travel_with_the_observation_into_the_artifacts(tmp_path, calibration):
    from app.ml.artifacts import ArtifactWriter, read_observations
    from app.ml.tracking import CourtTracker
    from tests.synthetic import SyntheticPlayer, render_frame

    tracker = CourtTracker(calibration, max_speed_ms=8.0, max_age_s=1.2)
    frame, detections = render_frame(calibration, [SyntheticPlayer((3.0, 4.0), 0)])
    points = np.tile(np.array([[1234.56, 789.01, 0.876]], np.float32), (17, 1))
    detections = [Detection(d.bbox, d.confidence, keypoints=points) for d in detections]

    tracked = tracker.update(frame_index=0, timestamp_s=0.0, detections=detections, frame=frame)
    assert tracked and tracked[0][1].keypoints is points
    with ArtifactWriter(tmp_path) as writer:
        writer.write_observations(tracked)
    _, stored = read_observations(tmp_path)[0]
    assert stored.keypoints.shape == (17, 3)
    assert stored.keypoints[0] == pytest.approx([1234.6, 789.0, 0.88])
