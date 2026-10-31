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


def _flight(start, n, x0, y0, dx, dy):
    return [(start + i, (start + i) / 30, x0 + dx * i, y0 + dy * i, 0.8) for i in range(n)]


def test_a_flight_is_kept_and_stray_answers_are_dropped():
    from app.ml.ball_track import flights

    ball = _flight(0, 30, 400, 600, 12, -8)
    # A light in the roof every few frames, a ball on the next court twice.
    noise = [(f, f / 30, 100, 200, 0.6) for f in range(0, 30, 3)] + [(7, 7 / 30, 50, 520, 0.55),
                                                                     (8, 8 / 30, 53, 521, 0.55)]
    found = sorted(ball + noise)
    # One answer per frame: the noise takes the frame (smaller x sorts
    # first), and the ball is missing there.
    seen: dict[int, tuple] = {}
    for record in found:
        seen.setdefault(record[0], record)
    out = flights(sorted(seen.values()), 1080)
    assert len(out) == 1
    assert out[0].xs[0] == 412 and out[0].frames[-1] == 29 and 7 not in out[0].frames


def test_the_ball_hidden_for_a_few_frames_is_the_same_flight():
    from app.ml.ball_track import flights

    found = _flight(0, 10, 300, 500, 10, 2) + _flight(14, 10, 440, 528, 10, 2)
    assert len(flights(found, 1080)) == 1
    far = _flight(0, 10, 300, 500, 10, 2) + _flight(30, 10, 440, 528, 10, 2)
    assert len(flights(far, 1080)) == 2


def test_something_still_is_not_the_ball():
    from app.ml.ball_track import flights

    still = [(f, f / 30, 900 + (f % 2), 300, 0.7) for f in range(40)]
    assert flights(still, 1080) == []


def test_the_payload_has_each_flight():
    from app.ml.ball_track import track_payload

    payload = track_payload(_flight(0, 8, 100, 100, 20, 0), 1920, 1080)
    assert payload["frame"] == [1920, 1080] and payload["seen"] == 8
    assert payload["flights"][0]["x"][:2] == [100, 120]


def test_a_point_off_the_course_after_a_gap_starts_a_new_flight():
    from app.ml.ball_track import flights

    # Down to the right, then hidden 4 frames, then an answer far up and
    # back: within reach of a fast ball, but not where this one was going.
    ball = _flight(0, 10, 700, 300, 8, 10)
    stray = _flight(14, 6, 760, 150, 6, -3)
    out = flights(ball + stray, 1080)
    assert [len(f.frames) for f in out] == [10, 6]
