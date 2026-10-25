"""Video access: probe, keyframe extraction, and a single-pass frame sampler.

Why this module exists
----------------------
The previous pipeline opened and fully decoded the same video five times
(court calibration, player tracking, ball tracking, pose, crops). On a Pi 5
reading from USB storage, H.264 decode is the dominant cost of the whole
analysis — more than the neural network at the sampling rates we can afford.

`FrameSampler` decodes the video exactly once, front to back, and uses
`VideoCapture.grab()` for frames it is going to throw away. `grab()` decodes
but skips colour conversion and the copy into a numpy array, which is roughly
2-3x cheaper than a full `read()`. No seeking is used anywhere: seeking in
long-GOP H.264 is both slow and unreliable, and it was the cause of the
warm-up bug in the old MOG2 detector.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class Frame:
    index: int          # frame number in the source video
    timestamp_s: float
    image: np.ndarray   # BGR


class VideoError(RuntimeError):
    pass


def probe(video_path: str | Path) -> VideoInfo:
    """Read container metadata. Raises VideoError if the file is unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(f"Impossibile aprire il video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    # Containers occasionally report nonsense; fall back to sane defaults
    # rather than propagating a divide-by-zero deep into the pipeline.
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    if w <= 0 or h <= 0:
        raise VideoError(f"Dimensioni video non valide: {w}x{h}")
    return VideoInfo(fps=fps, frame_count=max(count, 0), width=w, height=h)


def extract_keyframe(
    video_path: str | Path,
    out_path: str | Path,
    at_second: float = 5.0,
    jpeg_quality: int = 88,
) -> tuple[int, int]:
    """Grab one representative frame and write it as JPEG.

    Used by the calibration UI. Decodes sequentially up to `at_second` instead
    of seeking, so the result is exact and works on any container.
    Returns the frame size (w, h).
    """
    info = probe(video_path)
    target = int(max(0.0, at_second) * info.fps)
    if info.frame_count:
        target = min(target, max(info.frame_count - 1, 0))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoError(f"Impossibile aprire il video: {video_path}")
    try:
        last: np.ndarray | None = None
        idx = 0
        while idx <= target:
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
            idx += 1
        if last is None:
            raise VideoError("Nessun frame leggibile nel video")

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), last, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise VideoError(f"Scrittura keyframe fallita: {out_path}")
        h, w = last.shape[:2]
        return w, h
    finally:
        cap.release()


class FrameSampler:
    """Iterate a video once, yielding roughly `target_hz` frames per second.

    The sampler never seeks and never buffers more than one frame.

    Timestamps are the video's own (presentation time of each frame), not
    the frame number over the frame rate. Phones record at a variable rate:
    the frame rate in the header is an average, and a frame number divided
    by it drifts from the real time — by most of a second in six seconds on
    a test clip — which bends every speed and every rally boundary, and puts
    the boxes out of step with the video in the browser, which plays by the
    real times. A container that gives no timestamps falls back to the
    frame number, as before.

    Which frames are sampled stays a matter of frame numbers (every `step`
    frames from the first one analysed), so that a stored run replays on
    exactly the same frames.
    """

    def __init__(
        self,
        video_path: str | Path,
        target_hz: float,
        info: VideoInfo | None = None,
        max_duration_s: float | None = None,
        start_s: float = 0.0,
    ) -> None:
        self.video_path = str(video_path)
        self.info = info or probe(video_path)
        self.target_hz = max(0.1, float(target_hz))
        # step >= 1: how many source frames per emitted sample
        self.step = max(1, int(round(self.info.fps / self.target_hz)))
        # Effective rate after integer rounding — report this, not the request.
        self.effective_hz = self.info.fps / self.step
        # Measured from the start, not from the beginning of the video.
        self.max_duration_s = max_duration_s
        self._requested_start_s = max(0.0, float(start_s))
        # Known for certain only once the pass reaches it: until then, the
        # estimate from the average frame rate.
        self.start_frame = int(self._requested_start_s * self.info.fps)
        self._start_ts: float | None = None
        # Largest gap seen between the real time of a frame and the one its
        # number implies, in frames: above ~1.5 the video has a variable rate.
        self.max_drift_frames = 0.0

    @property
    def start_s(self) -> float:
        """Video time of the first emitted frame (the requested start
        until the pass has reached it)."""
        return self._start_ts if self._start_ts is not None else self._requested_start_s

    @property
    def variable_frame_rate(self) -> bool:
        return self.max_drift_frames > 1.5

    @property
    def expected_samples(self) -> int:
        total = self.info.frame_count
        if self.max_duration_s is not None and self.info.fps > 0:
            total = min(total or 0, self.start_frame + int(self.max_duration_s * self.info.fps))
        total -= self.start_frame
        if total <= 0:
            # Unknown length (some containers); caller must handle a 0 estimate.
            return 0
        return (total + self.step - 1) // self.step

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise VideoError(f"Impossibile aprire il video: {self.video_path}")

        fps = self.info.fps
        half_frame = 0.5 / fps
        try:
            idx = 0
            previous: float | None = None
            start: int | None = None
            while True:
                if not cap.grab():
                    break
                timestamp = _frame_time(cap, idx, fps, previous)
                previous = timestamp
                self.max_drift_frames = max(self.max_drift_frames, abs(timestamp - idx / fps) * fps)
                # Frames before the start are decoded in order, not sought:
                # seeking lands a few frames off on some videos, and the
                # frame numbers must be exact.
                if start is None and timestamp >= self._requested_start_s - half_frame:
                    start = idx
                    self.start_frame, self._start_ts = idx, timestamp
                if start is not None:
                    if self.max_duration_s is not None and timestamp - self._start_ts >= self.max_duration_s:
                        break
                    if (idx - start) % self.step == 0:
                        ok, image = cap.retrieve()
                        if ok and image is not None:
                            yield Frame(index=idx, timestamp_s=timestamp, image=image)
                idx += 1
        finally:
            cap.release()


def _frame_time(cap, index: int, fps: float, previous: float | None) -> float:
    """The frame's presentation time in seconds; from its number when the
    container gives none, or one that goes backwards."""
    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if ms is not None and np.isfinite(ms) and (ms > 0 or index == 0):
        seconds = float(ms) / 1000.0
        if previous is None or seconds > previous:
            return seconds
    estimate = index / fps
    if previous is not None and estimate <= previous:
        estimate = previous + 1.0 / fps
    return estimate
