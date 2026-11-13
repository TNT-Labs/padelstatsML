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


# ── The weaker peaks, and the flights followed further with them ────────────

def test_the_weaker_peaks_are_the_other_local_maxima_brightest_first():
    from app.ml.ball import weak_peaks

    heat = np.zeros((HEIGHT, WIDTH), np.float32)
    heat[100:105, 200:205] = 0.9                  # the ball: a plateau
    heat[50, 60], heat[250, 400], heat[10, 10] = 0.3, 0.45, 0.1
    heat[251, 401] = 0.44                         # the same peak, one pixel on
    found = locate(heat)
    assert weak_peaks(heat, found[:2]) == [(400.0, 250.0, 0.45), (60.0, 50.0, 0.3)]
    # Without a ball, the brightest peak comes first, the plateau once.
    peaks = weak_peaks(heat, None)
    assert peaks[0][2] == 0.9 and len(peaks) == 3 and peaks[1] == (400.0, 250.0, 0.45)
    assert weak_peaks(heat, found[:2], count=1) == [(400.0, 250.0, 0.45)]
    assert weak_peaks(np.zeros((HEIGHT, WIDTH), np.float32), None) == []


def _weak(records):
    out = {}
    for frame, t, x, y, p in records:
        out.setdefault(frame, (t, []))[1].append((x, y, p))
    return out


def test_without_the_weaker_peaks_the_flights_are_those_of_before():
    from app.ml.ball_track import flights

    found = _flight(0, 10, 300, 500, 10, 2) + _flight(20, 3, 520, 540, 10, 2)
    plain = flights(found, 1080)
    assert [f.frames for f in flights(found, 1080, None)] == [f.frames for f in plain]
    assert [len(f.frames) for f in plain] == [10]


def test_a_flight_is_followed_past_its_ends_onto_weak_peaks_where_it_was_heading():
    from app.ml.ball_track import RECOVER_MAX, flights

    # The ball seen strongly on frames 10-19, weakly before and after; noise
    # elsewhere in the picture on the same frames.
    strong = _flight(10, 10, 400, 500, 10, -6)
    weak = (_flight(2, 8, 320, 548, 10, -6) + _flight(20, 30, 500, 440, 10, -6)
            + [(f, f / 30, 1500, 200 + 3 * f, 0.2) for f in range(0, 50)])
    out = flights(strong, 1080, _weak(weak))
    assert len(out) == 1
    frames = out[0].frames
    assert frames[0] == 2                                  # back to where the weak track starts
    assert frames[-1] == 19 + RECOVER_MAX                  # forward, no further than allowed
    assert frames == sorted(frames) and len(set(frames)) == len(frames)
    assert all(x < 1000 for x in out[0].xs)               # the noise never taken
    assert out[0].times == sorted(out[0].times)


def test_a_gap_is_filled_from_the_line_between_its_ends():
    from app.ml.ball_track import flights

    ball = _flight(0, 20, 300, 500, 10, 2)
    found = [r for r in ball if r[0] not in (8, 9, 10)]
    weak = _weak([(8, 8 / 30, 380, 516, 0.2),          # on the line: the ball
                  (9, 9 / 30, 390, 600, 0.3),          # 80 px off: not it
                  (10, 10 / 30, 401, 520, 0.2)])
    out = flights(found, 1080, weak)
    assert 8 in out[0].frames and 10 in out[0].frames and 9 not in out[0].frames


def test_an_answer_the_chaining_left_out_is_taken_when_on_course():
    from app.ml.ball_track import flights

    # Three answers before the flight, too jittery to say where they were
    # going: the chaining cannot join them to it, and three are too few for
    # a flight. Going back along the flight, they are where it came from.
    found = sorted([(0, 0.0, 300, 488, 0.6), (1, 1 / 30, 310, 502, 0.6), (2, 2 / 30, 320, 478, 0.6)]
                   + _flight(6, 10, 360, 500, 10, 2))
    assert [f.frames[0] for f in flights(found, 1080)] == [6]
    out = flights(found, 1080, {})
    assert [f.frames[:4] for f in out] == [[0, 1, 2, 6]]


def test_the_weaker_peaks_are_stored_and_read_back(tmp_path):
    from app.ml.artifacts import ArtifactWriter, read_ball_weak

    assert read_ball_weak(tmp_path) is None
    with ArtifactWriter(tmp_path) as writer:
        writer.write_ball_weak(5, 0.1667, [(10.04, 20.0, 0.234), (30.0, 40.0, 0.2)])
        writer.write_ball_weak(6, 0.2, [])
    assert read_ball_weak(tmp_path) == {5: (0.167, [(10.0, 20.0, 0.23), (30.0, 40.0, 0.2)])}


def test_the_detector_gives_the_ball_and_the_weaker_peaks_in_video_pixels():
    from app.ml.ball import HailoBallDetector

    heat = np.zeros((HEIGHT, WIDTH), np.float32)
    heat[100:103, 200:203] = 0.9
    heat[250, 400] = 0.3
    detector = object.__new__(HailoBallDetector)        # no board: the map is given
    detector.threshold = 0.5
    detector.heat = lambda frames9: heat
    frames = [np.zeros((HEIGHT, WIDTH, 3), np.uint8)] * 3
    ball, weak = detector.find_all(frames, (1920, 1080))
    assert ball == detector.find(frames, (1920, 1080))
    assert ball[:2] == (201 * 3.75, 101 * 3.75) and weak == [(400 * 3.75, 250 * 3.75, 0.3)]
    heat[:] = 0
    heat[250, 400] = 0.3
    assert detector.find_all(frames, (1920, 1080)) == (None, [(1500.0, 937.5, 0.3)])
