"""The ball network's input and output, without the board."""
from __future__ import annotations

import numpy as np

from app.ml.ball import HEIGHT, WIDTH, find_model, locate, small, stack


def test_frames_are_stacked_oldest_first_as_in_training():
    frames = [small(np.full((1080, 1920, 3), v, np.uint8)) for v in (10, 20, 30)]
    out = stack(frames)
    assert out.shape == (HEIGHT, WIDTH, 9) and out.dtype == np.uint8
    assert list(out[0, 0]) == [10] * 3 + [20] * 3 + [30] * 3


def test_the_ball_is_the_largest_blob_over_the_threshold():
    heat = np.zeros((HEIGHT, WIDTH), np.float32)
    heat[100:103, 200:203] = 0.9          # the ball
    heat[10, 10] = 0.6                    # a lone pixel
    x, y, peak = locate(heat)
    assert (round(x), round(y), peak) == (201, 101, np.float32(0.9))
    assert locate(heat * 0.5) is None


def test_no_network_no_ball(tmp_path):
    model = tmp_path / "ball.hef"
    model.write_bytes(b"x")
    assert find_model(str(model)) == str(model)
