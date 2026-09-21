"""Person detector running on onnxruntime — no torch, no ultralytics.

Why not ultralytics on the Pi
-----------------------------
`ultralytics` imports torch unconditionally, even when the weights are ONNX.
torch + torchvision are ~250 MB of wheels, several seconds of import time and
a large resident memory footprint, all to hand a numpy array to onnxruntime.
Pre/post-processing for YOLOv8 is ~100 lines, so the Pi image drops torch
entirely and keeps ultralytics off-device, where it is used once to export
the ONNX model (see scripts/export_yolo_onnx.py).

The second reason is structural: `model.track(source=video)` owns its own
decode loop, which made single-pass processing impossible. Detection here is
a pure function of one frame, so `FrameSampler` stays in charge of decoding.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in full-frame px
    confidence: float

    @property
    def foot_px(self) -> tuple[float, float]:
        """Mid-bottom of the box: the player's contact point with the floor."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class DetectorError(RuntimeError):
    pass


class PersonDetector:
    """YOLOv8 person detector over onnxruntime (CPU)."""

    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 480,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.55,
        threads: int = 0,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise DetectorError(
                f"Modello non trovato: {path}. Esportalo con "
                f"`python scripts/export_yolo_onnx.py` (vedi INSTALL_RASPBERRY.md §6)."
            )
        if path.suffix.lower() != ".onnx":
            raise DetectorError(
                f"Atteso un modello .onnx, ricevuto '{path.suffix}'. "
                "Il Pi non carica pesi PyTorch."
            )

        import onnxruntime as ort

        opts = ort.SessionOptions()
        n = threads if threads > 0 else min(_physical_cores(), 4)
        opts.intra_op_num_threads = n
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        # Respect a statically-shaped export; fall back to the requested size.
        shape = self._session.get_inputs()[0].shape
        static_h = shape[2] if isinstance(shape[2], int) else None
        static_w = shape[3] if isinstance(shape[3], int) else None
        self.imgsz: tuple[int, int] = (
            int(static_h or imgsz),
            int(static_w or imgsz),
        )

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.model_path = str(path)
        self.threads = n

    def detect(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> list[Detection]:
        """Detect people in `frame`.

        `roi` is an (x1, y1, x2, y2) crop applied before inference; boxes are
        returned in full-frame coordinates regardless. Cropping to the court
        makes distant players larger in the letterboxed input, which is where
        recall is otherwise lost.
        """
        if roi is not None:
            x1, y1, x2, y2 = roi
            work = frame[y1:y2, x1:x2]
            offset = (float(x1), float(y1))
            if work.size == 0:
                work, offset = frame, (0.0, 0.0)
        else:
            work, offset = frame, (0.0, 0.0)

        blob, ratio, pad = _letterbox(work, self.imgsz)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._postprocess(outputs[0], ratio, pad, offset, work.shape[:2], frame.shape[:2])

    def _postprocess(
        self,
        raw: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        offset: tuple[float, float],
        work_shape: tuple[int, int],
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        pred = np.squeeze(raw)
        if pred.ndim != 2:
            raise DetectorError(f"Output del modello inatteso: shape {raw.shape}")
        # Ultralytics exports (1, 4+nc, anchors); some toolchains transpose it.
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T                      # -> (anchors, 4+nc)
        if pred.shape[1] < 5:
            raise DetectorError(f"Output del modello inatteso: shape {raw.shape}")

        scores = pred[:, 4 + PERSON_CLASS_ID]
        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return []

        boxes_xywh = pred[keep, :4]
        scores = scores[keep]

        # Letterbox space -> crop space
        cx, cy, bw, bh = (boxes_xywh[:, i] for i in range(4))
        x1 = (cx - bw / 2.0 - pad[0]) / ratio
        y1 = (cy - bh / 2.0 - pad[1]) / ratio
        x2 = (cx + bw / 2.0 - pad[0]) / ratio
        y2 = (cy + bh / 2.0 - pad[1]) / ratio

        wh, ww = work_shape
        x1 = np.clip(x1, 0, ww)
        x2 = np.clip(x2, 0, ww)
        y1 = np.clip(y1, 0, wh)
        y2 = np.clip(y2, 0, wh)

        widths, heights = x2 - x1, y2 - y1
        valid = (widths > 2) & (heights > 2)
        if not np.any(valid):
            return []
        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
        widths, heights, scores = widths[valid], heights[valid], scores[valid]

        idxs = cv2.dnn.NMSBoxes(
            np.stack([x1, y1, widths, heights], axis=1).tolist(),
            scores.astype(np.float32).tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(idxs) == 0:
            return []
        idxs = np.asarray(idxs).reshape(-1)

        fh, fw = frame_shape
        ox, oy = offset
        out: list[Detection] = []
        for i in idxs:
            out.append(
                Detection(
                    bbox=(
                        float(np.clip(x1[i] + ox, 0, fw)),
                        float(np.clip(y1[i] + oy, 0, fh)),
                        float(np.clip(x2[i] + ox, 0, fw)),
                        float(np.clip(y2[i] + oy, 0, fh)),
                    ),
                    confidence=float(scores[i]),
                )
            )
        out.sort(key=lambda d: -d.confidence)
        return out


def _letterbox(image: np.ndarray, imgsz: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize preserving aspect ratio, pad to imgsz, return the NCHW blob."""
    target_h, target_w = imgsz
    h, w = image.shape[:2]
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    dw, dh = (target_w - new_w) / 2.0, (target_h - new_h) / 2.0
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    canvas[top: top + new_h, left: left + new_w] = resized

    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
    return blob, ratio, (float(left), float(top))


def _physical_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))   # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return os.cpu_count() or 4
