from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.ml.video import FrameSampler, VideoError, extract_keyframe, probe


def _write_video(path: Path, frames: int = 90, fps: int = 30, size=(320, 180)) -> Path:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        pytest.skip("Questa build di OpenCV non sa scrivere file MP4")
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    return _write_video(tmp_path_factory.mktemp("video") / "clip.mp4")


def test_probe_reads_metadata(sample_video):
    info = probe(sample_video)
    assert info.width == 320 and info.height == 180
    assert info.fps == pytest.approx(30.0, abs=0.5)
    assert info.duration_s == pytest.approx(3.0, abs=0.2)


def test_probe_rejects_a_non_video(tmp_path):
    broken = tmp_path / "not-a-video.mp4"
    broken.write_bytes(b"definitely not h264")
    with pytest.raises(VideoError):
        probe(broken)


def test_sampler_emits_the_requested_rate(sample_video):
    sampler = FrameSampler(sample_video, target_hz=5.0)
    assert sampler.step == 6                      # 30 fps / 5 Hz
    assert sampler.effective_hz == pytest.approx(5.0)

    frames = list(sampler)
    assert len(frames) == pytest.approx(15, abs=1)
    # Indices must be exact multiples of the step: the frame index is what
    # ties every stage of the pipeline together.
    assert [f.index for f in frames[:4]] == [0, 6, 12, 18]
    assert frames[2].timestamp_s == pytest.approx(12 / 30.0, abs=1e-6)


def test_sampler_respects_the_duration_cap(sample_video):
    frames = list(FrameSampler(sample_video, target_hz=10.0, max_duration_s=1.0))
    assert frames
    assert max(f.timestamp_s for f in frames) < 1.05


def test_sampler_never_drops_below_one_frame_per_step(sample_video):
    # Asking for more samples than the video has frames must not divide by zero
    sampler = FrameSampler(sample_video, target_hz=1000.0)
    assert sampler.step == 1
    assert len(list(sampler)) == 90


def test_extract_keyframe_writes_a_readable_jpeg(sample_video, tmp_path):
    import cv2

    out = tmp_path / "key.jpg"
    size = extract_keyframe(sample_video, out, at_second=1.0)
    assert size == (320, 180)
    assert cv2.imread(str(out)) is not None


def test_extract_keyframe_clamps_past_the_end(sample_video, tmp_path):
    out = tmp_path / "key2.jpg"
    assert extract_keyframe(sample_video, out, at_second=9999.0) == (320, 180)
