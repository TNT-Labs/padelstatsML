"""Marking the ball by hand: clips inside the rallies, frames cut out of the
video, labels kept, the export for training."""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ml.ball_data import CLIP_FRAMES, NOT_VISIBLE, BallData, Clip, choose_clips


def test_clips_fall_inside_the_rallies_spread_over_the_match():
    rallies = [{"start_s": 10, "end_s": 40}, {"start_s": 100, "end_s": 104}, {"start_s": 300, "end_s": 330}]
    clips = choose_clips(rallies, fps=30.0, count=7, length=60)
    assert len(clips) == 7
    for clip in clips:
        t0, t1 = clip.start_frame / 30, (clip.start_frame + clip.frames) / 30
        assert any(r["start_s"] < t0 and t1 <= r["end_s"] for r in rallies)
    # Both long rallies have clips; the short one (4 s) has room for one.
    assert {int(c.start_s // 100) for c in clips} == {0, 1, 3}
    assert [c.id for c in clips] == list(range(7))
    # Asked for fewer, an even spread of them.
    assert len(choose_clips(rallies, fps=30.0, count=4, length=60)) == 4


def test_no_rally_long_enough_no_clips():
    assert choose_clips([{"start_s": 0, "end_s": 1.5}], fps=30.0) == []


def _video(path: Path, frames: int = 90) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    if not writer.isOpened():
        pytest.skip("Questa build di OpenCV non sa scrivere file MP4")
    for i in range(frames):
        image = np.full((180, 320, 3), 40, np.uint8)
        cv2.circle(image, (10 + 3 * i, 90), 3, (255, 255, 255), -1)       # the ball, moving right
        writer.write(image)
    writer.release()
    return path


def test_frames_are_cut_labelled_and_exported(tmp_path):
    video = _video(tmp_path / "match.mp4")
    data = BallData(tmp_path / "ball")
    clip = Clip(0, 30, 20, 1.0)
    data.save_clips([clip])
    assert data.extract(clip, video) == 20
    assert data.has_frames(clip)
    first = cv2.imread(str(data.frame_path(clip, 0)))
    assert first.shape == (180, 320, 3)

    data.update_labels({"30": [100.0, 90.0], "31": NOT_VISIBLE, "32": [106.0, 90.0]})
    labels = data.update_labels({"32": None})
    assert labels == {"30": [100.0, 90.0], "31": NOT_VISIBLE}

    target = data.export_zip("abcdef1234", (320, 180), tmp_path / "export.zip")
    with zipfile.ZipFile(target) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("labels.csv").decode())))
        assert len(rows) == 20 and rows[0]["file"] in archive.namelist()
        assert rows[0]["visible"] == "1"
        assert float(rows[0]["x"]) == pytest.approx(100.0 * 640 / 320)    # scaled to EXPORT_WIDTH
        assert rows[1]["visible"] == "0" and rows[2]["visible"] == ""
        image = cv2.imdecode(np.frombuffer(archive.read(rows[0]["file"]), np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (360, 640)


def test_a_clip_past_the_end_of_the_video_is_cut_short(tmp_path):
    video = _video(tmp_path / "short.mp4", frames=40)
    data = BallData(tmp_path / "ball")
    assert data.extract(Clip(0, 30, CLIP_FRAMES, 1.0), video) == 10
