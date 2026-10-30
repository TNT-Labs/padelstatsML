"""Forehand and backhand from the posture between the samples: the racket
wrist's swing, the side of the body it comes from, the racket hand."""
from __future__ import annotations

import numpy as np

from app.ml.identity import PlayerTrack
from app.ml.rallies import Rally
from app.ml.smoothing import SmoothedPath
from app.ml.strokes import detect_swings, ground_for_hand, pose_series
from app.ml.tracking import Observation, Tracklet

RALLY = [Rally(index=0, start_s=0.0, end_s=20.0, players_tracked=4)]
HZ = 15.0


def _pose(wrist_x: float, facing_camera: bool = False) -> np.ndarray:
    """Shoulders at y 60, hips at y 120 (torso 60 px), hips' centre at x 100.
    Seen from behind the right shoulder is on the picture's right; facing
    the camera, on its left. Both wrists at waist height, the right at
    `wrist_x` (picture pixels from the hips' centre), the left hanging."""
    p = np.zeros((17, 3), np.float32)
    p[:, 2] = 0.9
    right_x = -20 if facing_camera else 20
    p[5] = (100 - right_x, 60, 0.9)       # left shoulder
    p[6] = (100 + right_x, 60, 0.9)       # right shoulder
    p[11] = (100 - right_x / 2, 120, 0.9)
    p[12] = (100 + right_x / 2, 120, 0.9)
    p[9] = (100 - right_x, 130, 0.9)      # left wrist, hanging
    p[10] = (100 + wrist_x, 110, 0.9)     # right wrist
    return p


def _series(swing: list[float], start: float = 5.0, facing_camera: bool = False):
    """The right wrist still beside the body, then the positions of `swing`
    one pose apart."""
    times = np.round(np.arange(0, 10, 1 / HZ), 3)
    xs = np.full(len(times), 40.0 if not facing_camera else -40.0)
    k = int(start * HZ)
    xs[k:k + len(swing)] = swing
    xs[k + len(swing):] = swing[-1]
    points = np.stack([_pose(x, facing_camera) for x in xs])
    return times, points


def _path(y_m: float) -> SmoothedPath:
    times = np.arange(0, 10, 0.2)
    return SmoothedPath(times=times, positions=np.tile([3.0, y_m], (len(times), 1)),
                        velocities=np.zeros((len(times), 2)), segment=np.zeros(len(times), int), outliers=0)


def _strokes(series, hand="R", y_m=2.0):
    swings = detect_swings({0: series}, {0: _path(y_m)}, RALLY)[0]
    return ground_for_hand(swings, hand)


def test_a_swing_from_the_racket_side_is_a_forehand():
    # Seen from behind: the right wrist from the right (+x) across the body.
    strokes = _strokes(_series([60.0, -40.0, -60.0]))
    assert [s["kind"] for s in strokes] == ["forehand"]
    assert strokes[0]["t"] == round(5.0 + 1 / HZ, 2) and strokes[0]["volley"] is False


def test_a_swing_from_the_other_side_is_a_backhand():
    # The racket taken back to the left, slower, then swung across fast:
    # the fastest moment is the stroke, and it starts on the left.
    strokes = _strokes(_series([30.0, 15.0, 0.0, -15.0, -30.0, -45.0, -60.0, 40.0, 60.0]))
    assert [s["kind"] for s in strokes] == ["backhand"]


def test_facing_the_camera_the_sides_are_the_other_way_round():
    """A far player: his right is the picture's left. The wrist coming from
    the picture's left (-x) comes from his right: a forehand."""
    strokes = _strokes(_series([-60.0, 40.0, 60.0], facing_camera=True), y_m=18.0)
    assert strokes[0]["volley"] is False
    assert [s["kind"] for s in strokes] == ["forehand"]


def test_a_left_hander_has_the_forehand_on_the_other_side():
    swings = detect_swings({0: _series([60.0, -40.0, -60.0])}, {0: _path(2.0)}, RALLY)[0]
    # The same movement of the right wrist: not his racket, nothing.
    assert ground_for_hand(swings, "L") == []


def test_near_the_net_it_is_a_volley_and_without_the_hand_nothing():
    swings = detect_swings({0: _series([60.0, -40.0, -60.0])}, {0: _path(7.5)}, RALLY)[0]
    assert ground_for_hand(swings, "R")[0]["volley"] is True
    assert ground_for_hand(swings, None) == []


def test_walking_arms_are_not_swings():
    # The wrist swaying 5 px a pose: a torso every two seconds.
    strokes = _strokes(_series([45.0, 50.0, 45.0, 40.0]))
    assert strokes == []


def test_poses_between_the_samples_go_to_the_player_of_their_track():
    def obs(t, x):
        return Observation(frame_index=int(t * 30), timestamp_s=t, bbox=(0, 0, 10, 10), foot_px=(5, 10),
                           foot_court=(x, 2.0), confidence=0.9, color=np.zeros(4, np.float32))

    track_a = Tracklet(7, [obs(0.0, 1.0), obs(0.2, 1.1)])
    track_b = Tracklet(9, [obs(0.0, 8.0), obs(0.2, 8.1)])
    players = [PlayerTrack(0, 0, list(track_a.observations)), PlayerTrack(1, 0, list(track_b.observations))]
    pose = _pose(40.0)
    series = pose_series(players, [track_a, track_b], [(0.067, 9, pose), (0.133, 7, pose), (5.0, 7, pose)])
    assert list(series[0][0]) == [0.133]          # the one at 5 s is too far from any sample
    assert list(series[1][0]) == [0.067]
