"""Person detector on a Hailo accelerator (Raspberry Pi AI HAT+).

The same contract as app.ml.detect.PersonDetector — `detect(frame, roi)`
returns the people in full-frame pixels, `imgsz` is the network's input — so
everything after detection is unchanged. The network is a YOLOv8 compiled
for the board (.hef) with its non-maximum suppression on the chip: it hands
back, per class, the boxes as (y_min, x_min, y_max, x_max, score) in [0, 1]
of its input picture. Only the person class is read; the boxes then go
through the same clipping and suppression as on the CPU (finish_boxes), so a
box of part of a person is dropped here too.

The frame is letterboxed as for the CPU model — the network was trained on
letterboxed pictures; stretching 16:9 into a square makes everyone thinner.

hailo_platform is imported only here and only when the accelerator is in
use: the image without HailoRT, or a machine without the board, never
touches it.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import cv2
import numpy as np

from app.ml.detect import PERSON_CLASS_ID, Detection, DetectorError, finish_boxes

logger = logging.getLogger("padel.hailo")

# One inference takes a few milliseconds; this only catches a stuck device.
TIMEOUT_MS = 10_000
LETTERBOX_FILL = 114


class HailoRuntimeError(DetectorError):
    """The accelerator failed during the analysis."""


class HailoPersonDetector:
    """YOLOv8 person detector on a Hailo-8 / Hailo-8L."""

    # The pose network can share its board (pipeline._poser).
    on_accelerator = True

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.30,
                 iou_threshold: float = 0.55, alternatives: tuple[str, ...] | list[str] = ()) -> None:
        """`alternatives`: other networks found; the one compiled for the
        chip on the board is used (accel.pick_model)."""
        from app.ml.accel import pick_model

        path = Path(model_path)
        if not path.is_file():
            raise DetectorError(f"Modello Hailo non trovato: {path}")
        self.model_path = str(path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self._stack = contextlib.ExitStack()
        try:
            from hailo_platform import FormatType

            device = self._stack.enter_context(shared_device())
            self.device = _architecture(device)
            # The network compiled for this chip first; should the chip not
            # say which it is, the next one is tried when one does not load.
            candidates = [self.model_path, *(m for m in alternatives if m != self.model_path)]
            first = pick_model(candidates, self.device)
            candidates.remove(first)
            for attempt, candidate in enumerate([first, *candidates]):
                self.model_path = candidate
                try:
                    model = device.create_infer_model(candidate)
                    model.set_batch_size(1)
                    model.input().set_format_type(FormatType.UINT8)
                    output = model.output()
                    output.set_format_type(FormatType.FLOAT32)
                    # The network's own suppression was set when it was
                    # compiled (score 0.2 for the hailo-models networks):
                    # lower it to the weak detections the tracker uses to
                    # keep a far player.
                    _try(output, "set_nms_score_threshold", self.conf_threshold)
                    _try(output, "set_nms_iou_threshold", self.iou_threshold)
                    self._configured = self._stack.enter_context(model.configure())
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == len(candidates):
                        raise
                    logger.info("%s non caricato su %s: %s", Path(candidate).name, self.device, exc)
            height, width = (int(v) for v in model.input().shape[:2])
            self.imgsz: tuple[int, int] = (height, width)
            self._output_name = output.name
            # One inference on an empty picture now: a network that loads
            # but cannot run fails here, with its reason, not at the first
            # frame of an analysis. It also finds the output buffer the
            # runtime accepts (output_buffers).
            self._new_output = self._working_output(output_buffers(output))
        except DetectorError:
            self._stack.close()
            raise
        except Exception as exc:          # noqa: BLE001 - any HailoRT failure means: not usable
            self._stack.close()
            chip = getattr(self, "device", "Hailo")
            raise DetectorError(
                f"Acceleratore {chip} non utilizzabile con {Path(self.model_path).name}: {exc}") from exc

    @property
    def name(self) -> str:
        return f"{self.device} · {Path(self.model_path).name}"

    def detect(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> list[Detection]:
        if roi is not None:
            x1, y1, x2, y2 = roi
            work = frame[y1:y2, x1:x2]
            offset = (float(x1), float(y1))
            if work.size == 0:
                work, offset = frame, (0.0, 0.0)
        else:
            work, offset = frame, (0.0, 0.0)

        picture, ratio, pad = letterbox_rgb(work, self.imgsz)
        try:
            raw = self._infer(picture, self._new_output)
        except Exception as exc:          # noqa: BLE001
            raise HailoRuntimeError(f"Errore dell'acceleratore Hailo: {exc}") from exc
        return self._detections(person_boxes(raw), ratio, pad, offset, work.shape[:2], frame.shape[:2])

    def _infer(self, picture: np.ndarray, new_output):
        bindings = self._configured.create_bindings(output_buffers={self._output_name: new_output()})
        bindings.input().set_buffer(picture)
        self._configured.run([bindings], TIMEOUT_MS)
        raw = bindings.output().get_buffer()
        # HailoRT can report a failed inference only in its log and return:
        # the buffer, filled with NaN, then comes back unwritten.
        if isinstance(raw, np.ndarray) and raw.size and not np.isfinite(raw.reshape(-1)[0]):
            raise RuntimeError("la scheda non ha scritto il risultato (inferenza non eseguita)")
        return raw

    def _working_output(self, candidates):
        """The first output buffer with which an inference runs and reads
        back as boxes; DetectorError with every reason otherwise."""
        blank = np.full((*self.imgsz, 3), LETTERBOX_FILL, dtype=np.uint8)
        reasons = []
        for label, new_output in candidates:
            try:
                person_boxes(self._infer(blank, new_output))
            except Exception as exc:      # noqa: BLE001
                reasons.append(f"{label}: {exc}")
                logger.info("Buffer di uscita %s non accettato: %s", label, exc)
                continue
            logger.info("Buffer di uscita Hailo: %s", label)
            return new_output
        raise DetectorError("inferenza di prova non riuscita (" + " | ".join(reasons) + ")")

    def _detections(self, boxes, ratio, pad, offset, work_shape, frame_shape) -> list[Detection]:
        boxes = boxes[boxes[:, 4] >= self.conf_threshold]
        if not len(boxes):
            return []
        height, width = self.imgsz
        # Picture of the network, [0, 1] -> crop pixels.
        x1 = (boxes[:, 1] * width - pad[0]) / ratio
        y1 = (boxes[:, 0] * height - pad[1]) / ratio
        x2 = (boxes[:, 3] * width - pad[0]) / ratio
        y2 = (boxes[:, 2] * height - pad[1]) / ratio
        return finish_boxes(x1, y1, x2, y2, boxes[:, 4], offset, work_shape, frame_shape,
                            self.conf_threshold, self.iou_threshold)

    def close(self) -> None:
        self._stack.close()


def output_buffers(output) -> list[tuple[str, object]]:
    """Ways to allocate the network's output, in order: as large as its
    shape says, then as large as its non-maximum suppression needs — per
    class the box count and room for the most boxes it keeps, five values
    each. Which one a HailoRT version accepts for an NMS output is not
    documented the same way across versions."""
    candidates: list[tuple[str, object]] = []
    shape = tuple(int(v) for v in getattr(output, "shape", ()) or ())
    if shape:
        candidates.append((f"shape {shape}", lambda: np.full(shape, np.nan, dtype=np.float32)))
    nms = getattr(output, "nms_shape", None)
    classes = getattr(nms, "number_of_classes", None)
    boxes = getattr(nms, "max_bboxes_per_class", None) or getattr(nms, "max_bboxes_total", None)
    if classes and boxes:
        size = int(classes) * (1 + 5 * int(boxes))
        candidates.append((f"NMS {classes} classi x {boxes} riquadri",
                           lambda: np.full(size, np.nan, dtype=np.float32)))
    frame_size = getattr(output, "get_frame_size", None)
    if callable(frame_size):
        try:
            count = int(frame_size()) // 4
        except Exception:                 # noqa: BLE001
            count = 0
        if count:
            candidates.append((f"frame {count * 4} byte", lambda: np.full(count, np.nan, dtype=np.float32)))
    return candidates


def person_boxes(raw) -> np.ndarray:
    """The person class's boxes, (n, 5): y_min, x_min, y_max, x_max, score.

    HailoRT hands NMS results back either as a list per class of (n, 5)
    arrays (possibly inside a list per picture of the batch), or as the raw
    float buffer: per class, the number of boxes followed by their five
    values. Person is the first class of COCO."""
    if isinstance(raw, (list, tuple)):
        classes = raw
        if len(classes) == 1 and isinstance(classes[0], (list, tuple)):
            classes = classes[0]                      # batch of one
        if len(classes) <= PERSON_CLASS_ID:
            return np.zeros((0, 5), dtype=np.float32)
        return np.asarray(classes[PERSON_CLASS_ID], dtype=np.float32).reshape(-1, 5)
    flat = np.asarray(raw, dtype=np.float32).reshape(-1)
    index = 0
    for _ in range(PERSON_CLASS_ID):
        index += 1 + 5 * int(flat[index])
    if index >= flat.size:
        return np.zeros((0, 5), dtype=np.float32)
    count = int(flat[index])
    boxes = flat[index + 1: index + 1 + 5 * count]
    return boxes[: 5 * (boxes.size // 5)].reshape(-1, 5)


def letterbox_rgb(image: np.ndarray, imgsz: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    """The picture the network takes: RGB, uint8, aspect kept, padded grey.
    Returns it with the scale and the padding (left, top)."""
    target_h, target_w = imgsz
    h, w = image.shape[:2]
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), LETTERBOX_FILL, dtype=np.uint8)
    left, top = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[top: top + new_h, left: left + new_w] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(canvas), ratio, (float(left), float(top))


# One device for the process: the detector and the pose network run on it
# side by side, HailoRT's scheduler taking turns between them. A second
# VDevice in the same process would find the board taken.
_SHARED: dict = {"device": None, "stack": None, "users": 0}


@contextlib.contextmanager
def shared_device():
    """The process's Hailo device, opened by its first user and closed by
    its last."""
    if _SHARED["device"] is None:
        from hailo_platform import VDevice

        stack = contextlib.ExitStack()
        _SHARED["device"] = stack.enter_context(VDevice(_device_params(VDevice)))
        _SHARED["stack"] = stack
    _SHARED["users"] += 1
    try:
        yield _SHARED["device"]
    finally:
        _SHARED["users"] -= 1
        if _SHARED["users"] <= 0:
            stack, _SHARED["device"], _SHARED["stack"], _SHARED["users"] = _SHARED["stack"], None, None, 0
            stack.close()


class HailoPoseEstimator:
    """YOLOv8 pose network on the same board as the detector: the 17 body
    points of every person in the frame (app.ml.pose)."""

    def __init__(self, model_path: str | Path, alternatives: tuple[str, ...] | list[str] = (),
                 score_threshold: float = 0.3) -> None:
        from app.ml.accel import pick_model
        from app.ml.pose import decode_yolov8_pose

        self._decode = decode_yolov8_pose
        self.score_threshold = float(score_threshold)
        self.model_path = str(model_path)
        self._stack = contextlib.ExitStack()
        try:
            from hailo_platform import FormatType

            device = self._stack.enter_context(shared_device())
            self.device = _architecture(device)
            self.model_path = pick_model([self.model_path, *alternatives], self.device)
            model = device.create_infer_model(self.model_path)
            model.set_batch_size(1)
            model.input().set_format_type(FormatType.UINT8)
            self._outputs = []
            for name in _output_names(model):
                stream = model.output(name)
                stream.set_format_type(FormatType.FLOAT32)
                self._outputs.append((name, tuple(int(v) for v in stream.shape)))
            self._configured = self._stack.enter_context(model.configure())
            height, width = (int(v) for v in model.input().shape[:2])
            self.imgsz: tuple[int, int] = (height, width)
            self._infer(np.full((height, width, 3), LETTERBOX_FILL, dtype=np.uint8))
        except Exception as exc:          # noqa: BLE001 - any HailoRT failure means: no posture
            self._stack.close()
            raise DetectorError(f"Postura non disponibile con {Path(self.model_path).name}: {exc}") from exc

    @property
    def name(self) -> str:
        return Path(self.model_path).name

    def estimate(self, frame: np.ndarray) -> list:
        """The poses in `frame`, in frame pixels."""
        from app.ml.pose import to_frame

        picture, ratio, pad = letterbox_rgb(frame, self.imgsz)
        try:
            outputs = self._infer(picture)
        except Exception as exc:          # noqa: BLE001
            raise HailoRuntimeError(f"Errore della postura sull'acceleratore: {exc}") from exc
        return to_frame(self._decode(outputs, self.imgsz, score_threshold=self.score_threshold), ratio, pad)

    def _infer(self, picture: np.ndarray) -> dict[str, np.ndarray]:
        buffers = {name: np.full(shape, np.nan, dtype=np.float32) for name, shape in self._outputs}
        bindings = self._configured.create_bindings(output_buffers=buffers)
        bindings.input().set_buffer(picture)
        self._configured.run([bindings], TIMEOUT_MS)
        outputs = {name: np.asarray(bindings.output(name).get_buffer()) for name, _ in self._outputs}
        if any(a.size and not np.isfinite(a.reshape(-1)[0]) for a in outputs.values()):
            raise RuntimeError("la scheda non ha scritto il risultato (inferenza non eseguita)")
        return outputs

    def close(self) -> None:
        self._stack.close()


def _output_names(model) -> list[str]:
    names = getattr(model, "output_names", None)
    if names:
        return list(names)
    return [stream.name for stream in model.outputs]


def _device_params(vdevice_class):
    """The device's parameters with HailoRT's scheduler on: it activates the
    network for each inference. Without it the network loads but is never
    activated, and every inference fails (HAILO_STREAM_NOT_ACTIVATED)."""
    params = vdevice_class.create_params()
    try:
        from hailo_platform import HailoSchedulingAlgorithm

        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    except Exception as exc:              # noqa: BLE001
        logger.info("Scheduler HailoRT non impostato: %s", exc)
    return params


def _try(target, method: str, value: float) -> None:
    """Setters that only some HailoRT versions have."""
    setter = getattr(target, method, None)
    if setter is None:
        return
    try:
        setter(value)
    except Exception as exc:              # noqa: BLE001
        logger.info("HailoRT: %s(%s) non applicato: %s", method, value, exc)


def _architecture(device) -> str:
    """'Hailo-8' / 'Hailo-8L' from the board itself; 'Hailo' if it does not say."""
    try:
        physical = device.get_physical_devices()[0]
        arch = str(physical.control.identify().device_architecture)
    except Exception:                     # noqa: BLE001
        return "Hailo"
    arch = arch.upper()
    if "8L" in arch:
        return "Hailo-8L"
    if "10" in arch:
        return "Hailo-10"
    if "8" in arch:
        return "Hailo-8"
    return "Hailo"
