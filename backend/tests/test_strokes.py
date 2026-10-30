"""High strokes from the posture: smash and bandeja (racket and free arm
up, near the net), lob (racket up, further back, nearly still)."""
from __future__ import annotations

import numpy as np

from app.ml.identity import PlayerTrack
from app.ml.rallies import Rally
from app.ml.smoothing import SmoothedPath
from app.ml.strokes import classify, detect_strokes, for_hand
from app.ml.tracking import Observation


def _pose(up: str = "", half: str = "") -> np.ndarray:
    """A standing player: shoulders at 60, hips at 120 (torso 60 px). A
    wrist in `up` at 20 (2/3 of the torso above the shoulders), one in
    `half` at 50 (1/6 above), the others hanging at 130."""
    points = np.zeros((17, 3), np.float32)
    points[:, 2] = 0.9
    points[[5, 6], 1] = 60
    points[[11, 12], 1] = 120
    for side, i in (("L", 9), ("R", 10)):
        points[i, 1] = 20 if side in up else 50 if side in half else 130
    return points


def _player(pid: int, y_m: float, poses: dict[float, np.ndarray], speed: float = 0.5):
    times = np.round(np.arange(0, 20, 0.2), 2)
    observations = [
        Observation(frame_index=int(t * 5), timestamp_s=float(t), bbox=(0, 0, 50, 200), foot_px=(25, 200),
                    foot_court=(3.0, y_m), confidence=0.9, color=np.zeros(4, np.float32),
                    keypoints=poses.get(float(t), _pose()))
        for t in times
    ]
    path = SmoothedPath(times=times, positions=np.tile([3.0, y_m], (len(times), 1)),
                        velocities=np.tile([speed, 0.0], (len(times), 1)),
                        segment=np.zeros(len(times), int), outliers=0)
    return PlayerTrack(pid, 0, observations), path


RALLY = [Rally(index=0, start_s=2.0, end_s=15.0, players_tracked=4)]


def _strokes(y_m, poses, hand="R", speed=0.5):
    player, path = _player(0, y_m, poses, speed)
    return for_hand(detect_strokes([player], {0: path}, RALLY)[0], hand)


def test_racket_and_free_arm_up_near_the_net_is_a_smash():
    # The free arm rises first, pointing at the ball; the racket the sample after.
    assert _strokes(6.0, {4.0: _pose(up="L"), 4.2: _pose(up="R")}) == [{"t": 4.0, "kind": "overhead"}]
    # Both at once, the free arm a little lower.
    assert _strokes(6.0, {4.0: _pose(up="R", half="L")}) == [{"t": 4.0, "kind": "overhead"}]


def test_the_racket_up_with_the_free_arm_down_is_not_a_smash():
    """Nine «nothing» and seven other strokes checked on a real set: the free
    arm hung low every time."""
    assert _strokes(6.0, {4.0: _pose(up="R")}) == []


def test_further_back_and_nearly_still_it_is_a_lob():
    assert _strokes(1.0, {8.0: _pose(up="R")}) == [{"t": 8.0, "kind": "lob"}]
    # Running back there: a stroke on the run, or nothing; not counted.
    assert _strokes(1.0, {8.0: _pose(up="R")}, speed=3.0) == []


def test_between_points_a_raised_arm_is_not_a_stroke():
    assert _strokes(6.0, {17.0: _pose(up="RL"), 1.0: _pose(up="RL")}) == []


def test_the_racket_hand_decides_which_wrist_counts():
    stroke = {"t": 4.0, "net_m": 4.0, "speed": 1.0, "up": "L", "high": {"L": 0.7, "R": 0.1}}
    assert classify(stroke, "L") == "overhead"
    assert classify(stroke, "R") is None              # the left arm up, for a right-hander
    assert classify(stroke, None) == "overhead"


def test_results_from_before_the_free_arm_are_kept_as_they_were():
    old = [{"t": 1.0, "kind": "overhead", "wrists": "R"}, {"t": 2.0, "kind": "lob", "wrists": "L"}]
    assert for_hand(old, "R") == [{"t": 1.0, "kind": "overhead"}]
    assert len(for_hand(old, None)) == 2


def test_without_shoulders_and_hips_nothing_is_counted():
    pose = _pose(up="RL")
    pose[[11, 12], 2] = 0.1
    assert _strokes(6.0, {4.0: pose}) == []
