"""The AI accelerator (Raspberry Pi AI HAT+, Hailo): found or not, and the
detector on it — against a stand-in for HailoRT, which only runs where the
board is."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from app.ml import accel
from app.ml.detect import DetectorError
from app.ml.detect_hailo import (
    HailoPersonDetector,
    HailoRuntimeError,
    letterbox_rgb,
    person_boxes,
)

W, H = 1920, 1080


# ── Stand-in for hailo_platform ──────────────────────────────────────────────

class _FakeHailo:
    """The few calls of the HailoRT InferModel API the detector makes."""

    def __init__(self, result=None, fail_run=False, fail_open=False, arch="HAILO8", loads=None):
        self.result = result if result is not None else [np.zeros((0, 5), np.float32)] * 80
        self.fail_run, self.fail_open, self.arch = fail_run, fail_open, arch
        # Networks the chip accepts (by file name ending); None: any.
        self.loads = loads
        self.tried: list[str] = []
        self.thresholds: dict[str, float] = {}
        self.inputs: list[np.ndarray] = []
        self.closed = False

    def module(self):
        fake = self
        mod = types.ModuleType("hailo_platform")

        class FormatType:
            UINT8, FLOAT32 = "uint8", "float32"

        class Output:
            name, shape = "yolov8s/yolov8_nms_postprocess", (80, 5, 100)

            def set_format_type(self, _t): pass
            def set_nms_score_threshold(self, v): fake.thresholds["score"] = v
            def set_nms_iou_threshold(self, v): fake.thresholds["iou"] = v

        class Input:
            shape = (640, 640, 3)

            def set_format_type(self, _t): pass

        class Bindings:
            def __init__(self):
                self._in = types.SimpleNamespace(set_buffer=fake.inputs.append)
                self._out = types.SimpleNamespace(get_buffer=lambda: fake.result)

            def input(self): return self._in
            def output(self): return self._out

        class Configured:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def create_bindings(self, output_buffers): return Bindings()

            def run(self, _bindings, _timeout):
                if fake.fail_run:
                    raise RuntimeError("HAILO_TIMEOUT")

        class Model:
            def set_batch_size(self, _n): pass
            def input(self): return Input()
            def output(self): return Output()
            def configure(self): return Configured()

        class VDevice:
            @staticmethod
            def create_params(): return object()

            def __init__(self, _params):
                if fake.fail_open:
                    raise RuntimeError("HAILO_DRIVER_VERSION_MISMATCH")

            def __enter__(self): return self

            def __exit__(self, *_):
                fake.closed = True
                return False

            def create_infer_model(self, path):
                fake.tried.append(path)
                if fake.loads is not None and not path.endswith(fake.loads):
                    raise RuntimeError("HAILO_INVALID_HEF")
                return Model()

            def get_physical_devices(self):
                identity = types.SimpleNamespace(device_architecture=fake.arch)
                return [types.SimpleNamespace(control=types.SimpleNamespace(identify=lambda: identity))]

        mod.FormatType, mod.VDevice = FormatType, VDevice
        return mod


@pytest.fixture
def hef(tmp_path):
    path = tmp_path / "yolov8s_h8.hef"
    path.write_bytes(b"hef")
    return path


def _install(monkeypatch, fake: _FakeHailo):
    monkeypatch.setitem(sys.modules, "hailo_platform", fake.module())


def _normalised(box, ratio=640 / W, top=(640 - H * 640 / W) / 2):
    """A frame box as the network reports it: y0, x0, y1, x1 in [0, 1] of
    its letterboxed picture."""
    x1, y1, x2, y2 = box
    return [(y1 * ratio + top) / 640, x1 * ratio / 640, (y2 * ratio + top) / 640, x2 * ratio / 640]


# ── The detector ─────────────────────────────────────────────────────────────

def test_boxes_come_back_in_frame_pixels(monkeypatch, hef):
    person = (900.0, 500.0, 960.0, 680.0)
    classes = [np.array([_normalised(person) + [0.8]], np.float32)] + [np.zeros((0, 5), np.float32)] * 79
    fake = _FakeHailo(result=classes)
    _install(monkeypatch, fake)

    detector = HailoPersonDetector(hef, conf_threshold=0.12, iou_threshold=0.55)
    found = detector.detect(np.zeros((H, W, 3), np.uint8))

    assert detector.name == "Hailo-8 · yolov8s_h8.hef"
    assert detector.imgsz == (640, 640)
    assert len(found) == 1
    assert found[0].bbox == pytest.approx(person, abs=1.0)
    assert found[0].confidence == pytest.approx(0.8)
    # The network got the RGB letterboxed picture, and the weak detections
    # are not cut by the threshold it was compiled with.
    assert fake.inputs[0].shape == (640, 640, 3) and fake.inputs[0].dtype == np.uint8
    assert fake.thresholds["score"] == pytest.approx(0.12)
    detector.close()
    assert fake.closed


def test_a_crop_is_placed_back_in_the_frame(monkeypatch, hef):
    roi = (400, 200, 1520, 520)              # 1120 x 320
    _, ratio, (_, top) = letterbox_rgb(np.zeros((320, 1120, 3), np.uint8), (640, 640))
    person = (300.0, 100.0, 330.0, 160.0)    # in the crop
    box = _normalised(person, ratio=ratio, top=top) + [0.6]
    _install(monkeypatch, _FakeHailo(result=[[np.array([box], np.float32)]]))   # batch of one

    found = HailoPersonDetector(hef).detect(np.zeros((H, W, 3), np.uint8), roi=roi)
    assert found[0].bbox == pytest.approx((700.0, 300.0, 730.0, 360.0), abs=1.0)


def test_other_classes_and_weak_boxes_are_left_out(monkeypatch, hef):
    person = _normalised((100.0, 100.0, 140.0, 220.0))
    ball = _normalised((500.0, 500.0, 510.0, 510.0))
    classes = [np.array([person + [0.05]], np.float32)] + [np.array([ball + [0.9]], np.float32)] * 79
    _install(monkeypatch, _FakeHailo(result=classes))
    assert HailoPersonDetector(hef, conf_threshold=0.12).detect(np.zeros((H, W, 3), np.uint8)) == []


def test_a_board_that_cannot_be_opened_is_a_detector_error(monkeypatch, hef):
    _install(monkeypatch, _FakeHailo(fail_open=True))
    with pytest.raises(DetectorError, match="DRIVER_VERSION_MISMATCH"):
        HailoPersonDetector(hef)


def test_a_failure_during_the_analysis_is_told_apart(monkeypatch, hef):
    _install(monkeypatch, _FakeHailo(fail_run=True))
    detector = HailoPersonDetector(hef)
    with pytest.raises(HailoRuntimeError):
        detector.detect(np.zeros((H, W, 3), np.uint8))


def test_the_raw_nms_buffer_is_read_too():
    """Per class: the number of boxes, then five values each."""
    flat = np.array([2, 0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.5, 0.6, 0.6, 0.4, 1, 0, 0, 1, 1, 0.7], np.float32)
    boxes = person_boxes(flat)
    assert boxes.shape == (2, 5)
    assert boxes[1, 4] == pytest.approx(0.4)
    assert person_boxes(np.zeros(10, np.float32)).shape == (0, 5)


def test_the_letterbox_keeps_the_proportions():
    picture, ratio, (left, top) = letterbox_rgb(np.zeros((H, W, 3), np.uint8), (640, 640))
    assert picture.shape == (640, 640, 3)
    assert ratio == pytest.approx(1 / 3) and left == 0 and top == 140


# ── Found or not ─────────────────────────────────────────────────────────────

def test_no_board_no_accelerator(monkeypatch, tmp_path):
    monkeypatch.setattr(accel, "DEVICE_GLOB", str(tmp_path / "hailo*"))
    status = accel.hailo_status("auto")
    assert not status.usable and status.device is None and status.reason == "non rilevato"


def test_turned_off_whatever_is_fitted():
    assert accel.hailo_status("cpu").reason == "disattivato (ACCELERATOR=cpu)"


def test_board_without_the_runtime_or_the_network(monkeypatch, tmp_path, hef):
    (tmp_path / "hailo0").write_text("")
    monkeypatch.setattr(accel, "DEVICE_GLOB", str(tmp_path / "hailo*"))
    monkeypatch.setattr(accel.importlib.util, "find_spec", lambda name: None)
    assert "runtime HailoRT assente" in accel.hailo_status("auto").reason

    monkeypatch.setattr(accel.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(accel, "DEFAULT_MODELS", ())
    monkeypatch.setattr(accel, "FALLBACK_MODELS", ())
    assert "modello yolov8s_h8.hef non trovato" in accel.hailo_status("auto").reason

    status = accel.hailo_status("auto", str(hef))
    assert status.usable and status.model == str(hef)


def test_the_worker_records_what_it_found(monkeypatch, tmp_path, hef):
    (tmp_path / "hailo0").write_text("")
    monkeypatch.setattr(accel, "DEVICE_GLOB", str(tmp_path / "hailo*"))
    monkeypatch.setattr(accel.importlib.util, "find_spec", lambda name: object())
    _install(monkeypatch, _FakeHailo(arch="HAILO8L"))

    status = accel.probe_and_record("auto", str(hef), tmp_path)
    assert status.usable and status.reason == "Hailo-8L · yolov8s_h8.hef"
    assert accel.recorded_status(tmp_path)["usable"] is True

    _install(monkeypatch, _FakeHailo(fail_open=True))
    status = accel.probe_and_record("auto", str(hef), tmp_path)
    assert not status.usable and "DRIVER_VERSION_MISMATCH" in accel.recorded_status(tmp_path)["reason"]


# ── The network for the chip ─────────────────────────────────────────────────

def test_the_network_compiled_for_the_chip_is_chosen():
    both = ["/w/yolov8s_h8.hef", "/w/yolov8s_h8l.hef"]
    assert accel.pick_model(both, "Hailo-8L") == "/w/yolov8s_h8l.hef"
    assert accel.pick_model(both, "Hailo-8") == "/w/yolov8s_h8.hef"
    # Only the 8L one: it runs on a Hailo-8 too.
    assert accel.pick_model(["/w/yolov8s_h8l.hef"], "Hailo-8") == "/w/yolov8s_h8l.hef"
    assert accel.pick_model(both, "Hailo") == "/w/yolov8s_h8.hef"


def test_a_hailo_8l_board_runs_the_8l_network(monkeypatch, tmp_path, hef):
    """The AI HAT+ 13 TOPS: Board Name Hailo-8, Device Architecture HAILO8L.
    A network compiled for the Hailo-8 does not load on it."""
    small = tmp_path / "yolov8s_h8l.hef"
    small.write_bytes(b"hef")
    fake = _FakeHailo(arch="HAILO8L", loads="_h8l.hef")
    _install(monkeypatch, fake)

    detector = HailoPersonDetector(hef, alternatives=(str(hef), str(small)))
    assert detector.name == "Hailo-8L · yolov8s_h8l.hef"
    assert fake.tried == [str(small)]


def test_a_chip_that_does_not_say_which_it_is_gets_the_network_that_loads(monkeypatch, tmp_path, hef):
    small = tmp_path / "yolov8s_h8l.hef"
    small.write_bytes(b"hef")
    fake = _FakeHailo(arch="?", loads="_h8l.hef")
    _install(monkeypatch, fake)

    detector = HailoPersonDetector(hef, alternatives=(str(small),))
    assert detector.model_path == str(small)
    assert fake.tried == [str(hef), str(small)]


def test_no_network_for_the_chip_is_a_detector_error(monkeypatch, hef):
    _install(monkeypatch, _FakeHailo(arch="HAILO8L", loads="_h8l.hef"))
    with pytest.raises(DetectorError, match="HAILO_INVALID_HEF"):
        HailoPersonDetector(hef)
