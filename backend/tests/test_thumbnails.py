"""Each player's picture: a sample that is theirs, and theirs alone."""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.ml.identity import PlayerTrack
from app.ml.thumbnails import choose_pictures, pictures_from_video
from tests.synthetic import make_observation


def _samples(calibration, shirt, frames, x=3.0):
    return [make_observation(calibration, (x, 5.0), f / 5.0, shirt, frame_index=f) for f in frames]


def test_the_picture_is_a_sample_that_looks_like_the_player(calibration):
    """One sample of somebody else in the other pair's kit — the largest and
    most confident box — used to become the player's picture."""
    own = _samples(calibration, 0, range(0, 40))
    stranger = replace(make_observation(calibration, (3.0, 2.0), 50 / 5.0, 1, frame_index=50),
                       confidence=0.99)
    player = PlayerTrack(0, 0, sorted(own + [stranger], key=lambda o: o.frame_index), [1])

    chosen = choose_pictures([player])[0]

    assert chosen is not stranger
    assert chosen in own


def test_the_picture_has_nobody_else_in_it(calibration):
    """The best-looking sample of the player is half covered by another
    player: a picture of two people. A slightly smaller one alone wins."""
    near = _samples(calibration, 0, range(0, 10), x=3.0)
    covered = replace(near[5], confidence=0.99)
    samples = near[:5] + [covered] + near[6:]
    other = replace(covered, foot_court=(3.3, 5.0), bbox=tuple(v + 20 for v in covered.bbox))
    players = [PlayerTrack(0, 0, samples, [1]), PlayerTrack(1, 0, [other], [2])]

    chosen = choose_pictures(players)[0]

    assert chosen.frame_index != covered.frame_index


def test_the_picture_is_cut_from_the_frame_the_analysis_saw(tmp_path):
    """Frames are counted the way the analysis counts them, reading the
    video in order: a player in motion is cut where the box says."""
    size = (320, 240)
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    if not writer.isOpened():
        pytest.skip("Questa build di OpenCV non sa scrivere file MP4")
    for i in range(40):
        frame = np.full((size[1], size[0], 3), 60, np.uint8)
        frame[100:200, 5 * i:5 * i + 40] = (40, 40, 220)          # moves 5 px a frame
        writer.write(frame)
    writer.release()

    from app.ml.tracking import Observation
    x1 = 5 * 30
    obs = Observation(30, 1.2, (x1, 100.0, x1 + 40, 200.0), (x1 + 20, 200.0), (5.0, 5.0), 0.9,
                      np.full(72, 1 / 72, np.float32))

    picture = pictures_from_video(path, {0: obs})[0]
    image = cv2.imdecode(np.frombuffer(picture, np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    centre = image[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    red = (centre[:, :, 2] > 150) & (centre[:, :, 0] < 110)
    assert red.mean() > 0.9, "il ritaglio non è centrato sul giocatore di quel fotogramma"
