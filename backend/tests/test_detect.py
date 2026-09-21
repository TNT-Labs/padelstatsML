"""Detector geometry tests.

These cover the part most likely to be silently wrong: undoing the letterbox
and the ROI offset. A constant scale error here would shift every player by a
fixed amount on the court and would never raise an exception.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.detect import PersonDetector, _letterbox


def test_letterbox_preserves_aspect_and_pads():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    blob, ratio, (dw, dh) = _letterbox(image, (480, 480))

    assert blob.shape == (1, 3, 480, 480)
    assert ratio == pytest.approx(480 / 640)
    assert dw == 0                       # width is the limiting dimension
    assert dh == pytest.approx((480 - 360 * ratio) / 2, abs=1)
    assert blob.dtype == np.float32 and 0.0 <= blob.max() <= 1.0


def _stub_detector(imgsz: int = 480, conf: float = 0.3) -> PersonDetector:
    detector = object.__new__(PersonDetector)
    detector.imgsz = (imgsz, imgsz)
    detector.conf_threshold = conf
    detector.iou_threshold = 0.55
    return detector


# YOLOv8 at imgsz 480 emits 4725 anchors over 80 classes; the detector
# distinguishes the channel axis from the anchor axis by size, so the tests
# use a realistic anchor count rather than a handful.
ANCHORS, CHANNELS = 4725, 84


def _prediction(boxes: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    """Build a (1, 84, anchors) YOLOv8-style output from (cx, cy, w, h, score)."""
    out = np.zeros((1, CHANNELS, ANCHORS), dtype=np.float32)
    for i, (cx, cy, w, h, score) in enumerate(boxes):
        out[0, 0, i], out[0, 1, i], out[0, 2, i], out[0, 3, i] = cx, cy, w, h
        out[0, 4, i] = score          # channel 4 = person
    return out


def test_boxes_are_mapped_back_to_full_frame_coordinates():
    detector = _stub_detector()
    frame_shape = (360, 640)
    _, ratio, pad = _letterbox(np.zeros((*frame_shape, 3), np.uint8), (480, 480))

    # A box at the centre of the original frame, expressed in letterbox space.
    cx = 320 * ratio + pad[0]
    cy = 180 * ratio + pad[1]
    raw = _prediction([(cx, cy, 40 * ratio, 100 * ratio, 0.9)])

    detections = detector._postprocess(raw, ratio, pad, (0.0, 0.0), frame_shape, frame_shape)
    assert len(detections) == 1
    x1, y1, x2, y2 = detections[0].bbox
    assert (x1 + x2) / 2 == pytest.approx(320, abs=1)
    assert (y1 + y2) / 2 == pytest.approx(180, abs=1)
    assert detections[0].foot_px[1] == pytest.approx(230, abs=1)


def test_roi_offset_is_added_back():
    """Detections found in a court crop must come back in frame coordinates,
    otherwise every player lands in the wrong place on the court."""
    detector = _stub_detector()
    crop_shape, frame_shape = (300, 400), (1080, 1920)
    _, ratio, pad = _letterbox(np.zeros((*crop_shape, 3), np.uint8), (480, 480))

    cx = 200 * ratio + pad[0]
    cy = 150 * ratio + pad[1]
    raw = _prediction([(cx, cy, 30 * ratio, 80 * ratio, 0.8)])

    detections = detector._postprocess(raw, ratio, pad, (500.0, 200.0), crop_shape, frame_shape)
    x1, y1, x2, y2 = detections[0].bbox
    assert (x1 + x2) / 2 == pytest.approx(700, abs=1)     # 500 + 200
    assert (y1 + y2) / 2 == pytest.approx(350, abs=1)     # 200 + 150


def test_low_confidence_predictions_are_dropped():
    detector = _stub_detector(conf=0.5)
    raw = _prediction([(240, 240, 20, 60, 0.2)])
    assert detector._postprocess(raw, 1.0, (0.0, 0.0), (0.0, 0.0), (480, 480), (480, 480)) == []


def test_transposed_output_layout_is_handled():
    """Some export toolchains emit (1, anchors, 84) instead of (1, 84, anchors)."""
    detector = _stub_detector()
    raw = _prediction([(240, 240, 20, 60, 0.9)])
    transposed = np.ascontiguousarray(np.transpose(raw, (0, 2, 1)))
    assert transposed.shape == (1, ANCHORS, CHANNELS)
    detections = detector._postprocess(
        transposed, 1.0, (0.0, 0.0), (0.0, 0.0), (480, 480), (480, 480)
    )
    assert len(detections) == 1


def test_overlapping_boxes_are_suppressed():
    detector = _stub_detector()
    raw = _prediction(
        [(240, 240, 40, 120, 0.9), (243, 242, 40, 120, 0.7), (100, 100, 40, 120, 0.8)]
    )
    detections = detector._postprocess(raw, 1.0, (0.0, 0.0), (0.0, 0.0), (480, 480), (480, 480))
    assert len(detections) == 2
    assert detections[0].confidence > detections[1].confidence


def test_missing_model_fails_loudly(tmp_path):
    """No silent fallback. The previous ball tracker quietly degraded to a
    background subtractor and produced noise that looked like data."""
    from app.ml.detect import DetectorError

    with pytest.raises(DetectorError, match="non trovato"):
        PersonDetector(tmp_path / "nope.onnx")


def test_pytorch_weights_are_refused(tmp_path):
    from app.ml.detect import DetectorError

    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"stub")
    with pytest.raises(DetectorError, match="onnx"):
        PersonDetector(weights)
