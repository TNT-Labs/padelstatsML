"""High strokes from the posture: smash and bandeja near the net, lob
further back, the racket hand's only."""
from __future__ import annotations

import numpy as np

from app.ml.identity import PlayerTrack
from app.ml.rallies import Rally
from app.ml.smoothing import SmoothedPath
from app.ml.strokes import detect_strokes, for_hand
from app.ml.tracking import Observation


def _pose(raised: str = "") -> np.ndarray:
    """A standing player 200 px tall: shoulders at 60, hips at 120; a raised
    wrist at 20 (above the shoulders by 2/3 of the torso), else at 130."""
    points = np.zeros((17, 3), np.float32)
    points[:, 2] = 0.9
    points[[5, 6], 1] = 60
    points[[11, 12], 1] = 120
    points[9, 1] = 20 if "L" in raised else 130
    points[10, 1] = 20 if "R" in raised else 130
    return points


def _player(pid: int, y_m: float, raised_at: dict[float, str]) -> tuple[PlayerTrack, SmoothedPath]:
    times = np.round(np.arange(0, 20, 0.2), 2)
    observations = [
        Observation(frame_index=int(t * 5), timestamp_s=float(t), bbox=(0, 0, 50, 200), foot_px=(25, 200),
                    foot_court=(3.0, y_m), confidence=0.9, color=np.zeros(4, np.float32),
                    keypoints=_pose(raised_at.get(float(t), "")))
        for t in times
    ]
    path = SmoothedPath(times=times, positions=np.tile([3.0, y_m], (len(times), 1)),
                        velocities=np.zeros((len(times), 2)), segment=np.zeros(len(times), int), outliers=0)
    return PlayerTrack(pid, 0, observations), path


RALLY = [Rally(index=0, start_s=2.0, end_s=15.0, players_tracked=4)]


def test_a_raised_racket_near_the_net_is_a_smash_further_back_a_lob():
    near_net, near_path = _player(0, 6.0, {4.0: "R", 4.2: "R"})
    back, back_path = _player(1, 1.0, {8.0: "R"})
    strokes = detect_strokes([near_net, back], {0: near_path, 1: back_path}, RALLY)
    assert strokes[0] == [{"t": 4.0, "kind": "overhead", "wrists": "R"}]     # two samples, one stroke
    assert strokes[1] == [{"t": 8.0, "kind": "lob", "wrists": "R"}]


def test_between_points_a_raised_arm_is_not_a_stroke():
    player, path = _player(0, 6.0, {17.0: "R", 1.0: "LR"})
    assert detect_strokes([player], {0: path}, RALLY)[0] == []


def test_the_free_arm_is_left_out_once_the_hand_is_known():
    player, path = _player(0, 6.0, {4.0: "L", 9.0: "R", 12.0: "LR"})
    strokes = detect_strokes([player], {0: path}, RALLY)[0]
    assert [s["t"] for s in for_hand(strokes, "R")] == [9.0, 12.0]
    assert [s["t"] for s in for_hand(strokes, "L")] == [4.0, 12.0]
    assert len(for_hand(strokes, None)) == 3


def test_without_shoulders_and_hips_nothing_is_counted():
    player, path = _player(0, 6.0, {4.0: "R"})
    player.observations[20].keypoints[[11, 12], 2] = 0.1        # t = 4.0: the hips hidden
    assert detect_strokes([player], {0: path}, RALLY)[0] == []
