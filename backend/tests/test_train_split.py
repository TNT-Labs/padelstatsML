"""The ball network's validation split (training/ball/train_tracknet.py)."""
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "ball"))
import train_tracknet as tt  # noqa: E402

KEYS = [("/c/palla_terzo.zip", str(i)) for i in range(20)] + [("/c/palla_secondo.zip", str(i)) for i in range(21)]


def test_a_whole_match_held_out_never_reaches_training():
    train, val = tt.split(KEYS, "palla_secondo.zip", random.Random(7))
    assert {k[0] for k in val} == {"/c/palla_secondo.zip"} and len(val) == 21
    assert {k[0] for k in train} == {"/c/palla_terzo.zip"} and len(train) == 20
    assert tt.split(KEYS, "terzo", random.Random(7))[1] == sorted(k for k in KEYS if "terzo" in k[0])


def test_a_match_held_out_takes_its_far_half_too():
    keys = KEYS + [("/c/lontana_secondo.zip", str(i)) for i in range(21)]
    train, val = tt.split(keys, "secondo", random.Random(7))
    assert {k[0] for k in val} == {"/c/palla_secondo.zip", "/c/lontana_secondo.zip"}
    assert {k[0] for k in train} == {"/c/palla_terzo.zip"}


def test_a_name_that_leaves_nothing_to_train_on_stops():
    for bad in ("palla", "quarto"):            # every zip, or none
        with pytest.raises(SystemExit):
            tt.split(KEYS, bad, random.Random(7))
    with pytest.raises(SystemExit):
        tt.split(KEYS[:20], "terzo", random.Random(7))      # nothing left to train on


def test_without_a_holdout_the_split_is_the_one_of_before():
    # The same clips as the earlier versions drew with the same seed: a new
    # training compares with an old one on the same validation clips.
    random.seed(7)
    keys = sorted(KEYS)
    random.shuffle(keys)
    n_val = max(1, int(round(len(keys) * tt.VAL_SHARE)))
    random.seed(7)
    train, val = tt.split(KEYS, None, random)
    assert val == keys[:n_val] and train == keys[n_val:]


def test_the_strong_variations_move_the_mark_with_the_ball():
    import numpy as np

    frame = np.zeros((tt.HEIGHT, tt.WIDTH, 3), np.uint8)
    frame[140:143, 300:303] = 255                            # the ball at (301, 141)
    for seed in range(30):
        random.seed(seed)
        np.random.seed(seed)
        stack, visible, x, y = tt._strong([frame, frame, frame], True, 301.0, 141.0)
        assert all(im.shape == frame.shape and im.dtype == np.uint8 for im in stack)
        if visible:
            gray = stack[2].astype(float).sum(axis=2)
            iy, ix = np.unravel_index(int(gray.argmax()), gray.shape)
            assert abs(ix - x) <= 2 and abs(iy - y) <= 2, seed
        else:
            assert (x, y) == (0.0, 0.0)
