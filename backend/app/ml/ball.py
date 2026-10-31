"""The ball, found by the network trained on the user's marked frames.

training/ball/train_tracknet.py trains it: three consecutive frames in,
scaled to 512 x 288 and stacked into 9 channels (BGR, oldest first), a heat
map of where the ball is in the last one out. Compiled for the Hailo
(hailo_compile.sh: the chip scales the bytes to 0-1 itself), it is fed the
bytes as they are.

The ball is where the heat map is over THRESHOLD: the centre of the largest
such blob, in the video's pixels.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 512, 288
THRESHOLD = 0.5
# Where the compiled network is looked for: weights/ on the Pi.
BALL_MODELS = ("/srv/weights/ball_h8l.hef", "/srv/weights/ball_h8.hef")


def find_model(configured: str | None = None) -> str | None:
    for path in ([configured] if configured else []) + list(BALL_MODELS):
        if path and Path(path).is_file():
            return path
    return None


def small(frame: np.ndarray) -> np.ndarray:
    """A frame as the network sees it: 512 x 288, as in training."""
    return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def stack(frames: list[np.ndarray]) -> np.ndarray:
    """Three frames (already small), oldest first -> 288 x 512 x 9 bytes."""
    return np.ascontiguousarray(np.concatenate(frames, axis=2))


def locate(heat: np.ndarray, threshold: float = THRESHOLD) -> tuple[float, float, float] | None:
    """(x, y, peak) of the ball in heat-map pixels, or None."""
    peak = float(heat.max())
    if peak < threshold:
        return None
    mask = (heat >= threshold).astype(np.uint8)
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(centres[best][0]), float(centres[best][1]), peak


class HailoBallDetector:
    """The ball network on the AI HAT+, on the device the players' networks
    share (detect_hailo.shared_device)."""

    def __init__(self, model_path: str) -> None:
        from app.ml.detect_hailo import TIMEOUT_MS, _output_names, shared_device

        self.model_path = model_path
        self._timeout = TIMEOUT_MS
        self._stack = contextlib.ExitStack()
        try:
            from hailo_platform import FormatType

            device = self._stack.enter_context(shared_device())
            model = device.create_infer_model(model_path)
            model.set_batch_size(1)
            model.input().set_format_type(FormatType.UINT8)
            self._output = _output_names(model)[0]
            stream = model.output(self._output)
            stream.set_format_type(FormatType.FLOAT32)
            self._shape = tuple(int(v) for v in stream.shape)
            self.input_shape = tuple(int(v) for v in model.input().shape)
            self._configured = self._stack.enter_context(model.configure())
            self.heat(np.zeros((HEIGHT, WIDTH, 9), np.uint8))     # trial inference
        except Exception:
            self._stack.close()
            raise

    @property
    def name(self) -> str:
        return Path(self.model_path).name

    def heat(self, frames9: np.ndarray) -> np.ndarray:
        """288 x 512 heat map of the last of the three frames."""
        buffer = np.full(self._shape, np.nan, dtype=np.float32)
        bindings = self._configured.create_bindings(output_buffers={self._output: buffer})
        bindings.input().set_buffer(frames9)
        self._configured.run([bindings], self._timeout)
        out = np.asarray(bindings.output(self._output).get_buffer(), dtype=np.float32)
        if out.size and not np.isfinite(out.reshape(-1)[0]):
            raise RuntimeError("la scheda non ha scritto il risultato (inferenza non eseguita)")
        return out.reshape(HEIGHT, WIDTH)

    def find(self, frames: list[np.ndarray], frame_size: tuple[int, int]) -> tuple[float, float, float] | None:
        """The ball in the last of three small frames, in video pixels
        (frame_size = width, height of the video): (x, y, peak) or None."""
        found = locate(self.heat(stack(frames)))
        if found is None:
            return None
        width, height = frame_size
        return found[0] * width / WIDTH, found[1] * height / HEIGHT, found[2]

    def close(self) -> None:
        self._stack.close()
