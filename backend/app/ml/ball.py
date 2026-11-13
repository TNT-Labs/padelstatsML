"""The ball, found by the network trained on the user's marked frames.

training/ball/train_tracknet.py trains it: three consecutive frames in,
scaled to 512 x 288 and stacked into 9 channels (BGR, oldest first), a heat
map of where the ball is in the last one out. Compiled for the Hailo
(hailo_compile.sh: the chip scales the bytes to 0-1 itself), it is fed the
bytes as they are.

The ball is where the heat map is over THRESHOLD: the centre of the largest
such blob, in the video's pixels.

On a quarter of the frames with the ball in sight the brightest point of the
map is not the ball (validation, training/ball/README.md): it is often still
there, a weaker peak. The weaker peaks (weak_peaks: local maxima over
WEAK_THRESHOLD, at most WEAK_MAX) are kept apart, in ball_weak.jsonl:
alone they are mostly noise, but one where a flight of the ball was heading
is the ball (ball_track.recover).

The far half is a band across the middle of the picture: from far away the
ball is about 8 pixels of a 1080p video, 2 once the whole frame is scaled to
512 x 288 — averaged into the floor. far_crop is that half and the air above
it (lobs) at a fixed 16:9 size, scaled by about half: the same network,
looking a second time there, sees the far ball twice as large
(ball_far.jsonl, merged with the whole frame's ball when read:
ball_track.merge_far).
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 512, 288
THRESHOLD = 0.5
# The weaker peaks kept beside the ball: over this, at most this many a frame,
# and not within WEAK_APART pixels of the map of the ball or of each other.
WEAK_THRESHOLD = 0.15
WEAK_MAX = 3
WEAK_APART = 6.0
# Where the compiled network is looked for: weights/ on the Pi.
BALL_MODELS = ("/srv/weights/ball_h8l.hef", "/srv/weights/ball_h8.hef")


# The far crop: the far half between the net posts, MARGIN_M aside, and up
# from just below the net to the top of the lobs. Worth it only when it is
# much narrower than the frame (FAR_MAX_SHARE): otherwise it enlarges little.
FAR_MARGIN_M = 0.5
FAR_BELOW_NET_PX = 0.02          # of the frame's height, below the net line
FAR_MAX_SHARE = 0.7


def far_crop(calibration) -> tuple[int, int, int, int] | None:
    """(x, y, width, height) in video pixels of the far crop, 16:9, inside
    the frame; None when the far half is not much narrower than the frame."""
    from app.ml.court import COURT_WIDTH_M, NET_Y_M

    width, height = calibration.frame_size
    net = calibration.court_to_pixels(np.array(
        [[-FAR_MARGIN_M, NET_Y_M], [COURT_WIDTH_M + FAR_MARGIN_M, NET_Y_M]], dtype=np.float32))
    if not np.isfinite(net).all():
        return None
    x1, x2 = float(net[:, 0].min()), float(net[:, 0].max())
    w = int(np.ceil((x2 - x1) / 16.0)) * 16
    h = w * 9 // 16
    if w > FAR_MAX_SHARE * width or w < 64:
        return None
    bottom = min(float(net[:, 1].max()) + FAR_BELOW_NET_PX * height, height)
    x = int(round((x1 + x2) / 2 - w / 2))
    y = int(round(bottom - h))
    x = min(max(x, 0), width - w)
    y = min(max(y, 0), height - h)
    return x, y, w, h


def small_crop(frame: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """The far crop of a frame as the network sees it: 512 x 288."""
    x, y, w, h = crop
    return cv2.resize(frame[y:y + h, x:x + w], (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


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


def weak_peaks(heat: np.ndarray, ball: tuple[float, float] | None,
               threshold: float = WEAK_THRESHOLD, count: int = WEAK_MAX) -> list[tuple[float, float, float]]:
    """Local maxima of the heat map over `threshold`, brightest first, away
    from `ball` (the blob taken for the ball): (x, y, peak) in heat-map
    pixels, at most `count`."""
    if count <= 0 or float(heat.max()) < threshold:
        return []
    heat = np.asarray(heat, dtype=np.float32)
    top = (heat >= cv2.dilate(heat, np.ones((5, 5), np.uint8))) & (heat >= threshold)
    ys, xs = np.nonzero(top)
    if len(xs) > 64:                       # a flat bright patch: its brightest pixels
        keep = np.argpartition(-heat[ys, xs], 64)[:64]
        ys, xs = ys[keep], xs[keep]
    values = heat[ys, xs]
    out: list[tuple[float, float, float]] = []
    for i in np.argsort(-values, kind="stable"):
        x, y = float(xs[i]), float(ys[i])
        if ball is not None and np.hypot(x - ball[0], y - ball[1]) <= WEAK_APART:
            continue
        if any(np.hypot(x - o[0], y - o[1]) <= WEAK_APART for o in out):
            continue                       # the same peak: a plateau
        out.append((x, y, round(float(values[i]), 3)))
        if len(out) == count:
            break
    return out


class HailoBallDetector:
    """The ball network on the AI HAT+, on the device the players' networks
    share (detect_hailo.shared_device)."""

    def __init__(self, model_path: str, threshold: float = THRESHOLD) -> None:
        from app.ml.detect_hailo import TIMEOUT_MS, _output_names, shared_device

        self.model_path = model_path
        self.threshold = float(threshold)
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
        found = locate(self.heat(stack(frames)), self.threshold)
        if found is None:
            return None
        width, height = frame_size
        return found[0] * width / WIDTH, found[1] * height / HEIGHT, found[2]

    def find_all(self, frames: list[np.ndarray], frame_size: tuple[int, int],
                 crop: tuple[int, int, int, int] | None = None):
        """As find, and the weaker peaks (weak_peaks) in video pixels:
        (ball or None, [(x, y, peak)]). With `crop` (far_crop) the frames
        are that crop of the video (small_crop), and so are the answers."""
        heat = self.heat(stack(frames))
        found = locate(heat, self.threshold)
        weak = weak_peaks(heat, found[:2] if found else None)
        if crop is None:
            x0, y0, (w, h) = 0, 0, frame_size
        else:
            x0, y0, w, h = crop
        sx, sy = w / WIDTH, h / HEIGHT
        ball = (x0 + found[0] * sx, y0 + found[1] * sy, found[2]) if found else None
        return ball, [(x0 + x * sx, y0 + y * sy, p) for x, y, p in weak]

    def close(self) -> None:
        self._stack.close()
