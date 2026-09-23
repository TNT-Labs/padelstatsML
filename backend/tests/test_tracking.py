from __future__ import annotations

import numpy as np

from app.ml.detect import Detection
from app.ml.tracking import COLOR_DIM, CourtTracker, Observation, histogram_distance, _torso_histogram
from tests.synthetic import SyntheticPlayer, render_frame

HZ = 5.0
DT = 1.0 / HZ


def _run(calibration, trajectories, hz: float = HZ) -> list:
    """trajectories: list of per-frame [SyntheticPlayer, ...]"""
    tracker = CourtTracker(calibration, max_speed_ms=8.0, max_age_s=1.2)
    for i, players in enumerate(trajectories):
        frame, detections = render_frame(calibration, players)
        tracker.update(frame_index=i, timestamp_s=i / hz, detections=detections, frame=frame)
    return tracker.finish()


def test_four_players_produce_four_tracklets(calibration):
    starts = [(2.0, 4.0), (7.0, 4.0), (3.0, 16.0), (8.0, 16.0)]
    frames = []
    for i in range(60):
        players = [
            SyntheticPlayer((x + 1.2 * np.sin(i / 6.0 + k), y + 0.8 * np.cos(i / 5.0 + k)), k)
            for k, (x, y) in enumerate(starts)
        ]
        frames.append(players)

    tracklets = _run(calibration, frames)
    assert len(tracklets) == 4
    assert all(len(t) == 60 for t in tracklets)


def test_fast_movement_does_not_break_association(calibration):
    """At 5 Hz a sprinting player's boxes barely overlap, which is exactly
    where pixel-IoU tracking fails. Court-metre gating must hold."""
    frames = []
    for i in range(40):
        y = 2.0 + (i * DT) * 5.0            # 5 m/s straight down the court
        frames.append([SyntheticPlayer((5.0, min(y, 18.0)), 0)])

    tracklets = _run(calibration, frames)
    assert len(tracklets) == 1
    assert len(tracklets[0]) == 40


def test_teleporting_detection_starts_a_new_track(calibration):
    """A jump no human could make must not be absorbed into an existing
    track: that is how one player's distance ends up including another's."""
    frames = [[SyntheticPlayer((1.0, 2.0), 0)] for _ in range(10)]
    frames += [[SyntheticPlayer((9.0, 18.0), 1)] for _ in range(10)]

    tracklets = _run(calibration, frames)
    assert len(tracklets) == 2


def test_detections_outside_the_court_are_rejected(calibration):
    """Spectators and passers-by project far outside the playing surface."""
    tracker = CourtTracker(calibration)
    frame = np.full((1080, 1920, 3), 60, dtype=np.uint8)
    # Feet at (10, 400) project to roughly x = -6 m: six metres left of the
    # side wall, i.e. someone standing outside the court.
    far_outside = Detection(bbox=(0.0, 300.0, 20.0, 400.0), confidence=0.9)
    tracker.update(0, 0.0, [far_outside], frame)
    assert tracker.detections_rejected_off_court >= 1
    assert tracker.finish() == []


def test_tracks_shorter_than_the_minimum_are_discarded(calibration):
    frames = [[SyntheticPlayer((5.0, 5.0), 0)] for _ in range(2)]
    assert _run(calibration, frames) == []


def test_torso_histograms_separate_different_shirts(calibration):
    frame, detections = render_frame(
        calibration, [SyntheticPlayer((2.0, 5.0), 0), SyntheticPlayer((8.0, 5.0), 1)]
    )
    a = _torso_histogram(frame, detections[0].bbox)
    b = _torso_histogram(frame, detections[1].bbox)
    assert histogram_distance(a, a) < 1e-6
    assert histogram_distance(a, b) > 0.5


def test_a_track_survives_missed_detections_while_the_player_runs(calibration):
    """A track must be re-associated after the detector loses the player.

    The gate used to be sized for one sampling interval, whatever the track's
    own age. Here the player runs at 5 m/s and is missed for five samples — a
    full second, five metres of real movement. The constant-velocity
    prediction is clamped to a 0.4 s horizon, so it accounts for only two of
    those metres; the remaining three exceed the one-step gate of 2.2 m, the
    re-association was rejected and the track abandoned.
    """
    frames: list[list[SyntheticPlayer]] = []
    y = 2.0
    for i in range(30):
        y = min(y + 5.0 * DT, 18.0)
        frames.append([] if 10 <= i <= 14 else [SyntheticPlayer((5.0, y), 0)])

    tracklets = _run(calibration, frames)
    assert len(tracklets) == 1, f"traccia spezzata in {len(tracklets)} frammenti"
    assert len(tracklets[0]) == 25


def test_a_stale_track_does_not_outbid_a_fresh_one(calibration):
    """Widening the gate for stale tracks must not let them steal detections:
    the spatial cost stays normalised by the one-step gate, so the track
    actually next to the detection still wins."""
    frames = []
    for i in range(12):
        near = SyntheticPlayer((2.0, 4.0 + i * 0.1), 0)
        far = SyntheticPlayer((8.0, 15.0), 1)
        # The far player disappears for four samples, going stale.
        frames.append([near] if 6 <= i <= 9 else [near, far])

    tracklets = _run(calibration, frames)
    by_position = sorted(tracklets, key=lambda t: t.mean_position()[0])
    assert len(by_position) == 2
    # The near player keeps every sample: nothing was stolen by the stale track.
    assert len(by_position[0]) == 12


def test_an_impossible_jump_is_still_rejected_after_a_gap(calibration):
    """The widened gate is physical, not unlimited: after a 0.4 s gap a player
    can cover about 3.8 m, not the whole court."""
    frames = [[SyntheticPlayer((1.0, 2.0), 0)] for _ in range(8)]
    frames += [[], []]
    frames += [[SyntheticPlayer((9.0, 18.0), 1)] for _ in range(8)]

    tracklets = _run(calibration, frames)
    assert len(tracklets) == 2


def _observation(t: float, position: tuple[float, float]) -> Observation:
    color = np.full(COLOR_DIM, 1.0 / COLOR_DIM, dtype=np.float32)
    return Observation(
        frame_index=int(round(t * HZ)),
        timestamp_s=t,
        bbox=(0.0, 0.0, 10.0, 40.0),
        foot_px=(5.0, 40.0),
        foot_court=position,
        confidence=0.9,
        color=color,
    )


def test_one_displaced_sample_does_not_kill_the_track(calibration):
    """The gate is measured from the last observed position, not from the
    prediction.

    A player runs at 4 m/s (0.8 m per sample). One box is cut short —
    occlusion by the net or by another player — and its foot point lands
    1.2 m behind the real position. Velocity from the last two points turns
    that into -2 m/s and the prediction overshoots backwards: the next,
    correct, detection is 2.4 m from the prediction but only 2.0 m from
    where the player was last seen, inside the 2.2 m one-step gate. Gating
    on the prediction rejected it and split the track at the first bad
    sample.
    """
    true_y = [5.0 + 0.8 * i for i in range(10)]
    observed = list(true_y)
    observed[4] -= 1.2                                  # the cropped box

    tracker = CourtTracker(calibration, max_speed_ms=8.0, max_age_s=1.2)
    for i, y in enumerate(observed):
        tracker.update_observations(i * DT, [_observation(i * DT, (5.0, y))])
    tracklets = tracker.finish()

    assert len(tracklets) == 1, f"traccia spezzata in {len(tracklets)} frammenti"
    assert len(tracklets[0]) == 10


def test_update_observations_matches_update(calibration):
    """The replay entry point must associate exactly like the live one."""
    frames = []
    for i in range(30):
        frames.append([
            SyntheticPlayer((2.0 + 0.1 * i, 4.0), 0),
            SyntheticPlayer((8.0, 16.0 - 0.1 * i), 1),
        ] if i % 7 else [SyntheticPlayer((2.0 + 0.1 * i, 4.0), 0)])

    live = CourtTracker(calibration)
    replay = CourtTracker(calibration)
    for i, players in enumerate(frames):
        frame, detections = render_frame(calibration, players)
        tracked = live.update(frame_index=i, timestamp_s=i * DT, detections=detections, frame=frame)
        replay.update_observations(i * DT, [obs for _, obs in tracked])

    def partition(tracklets):
        return sorted(tuple(o.frame_index for o in t.observations) for t in tracklets)

    assert partition(live.finish()) == partition(replay.finish())
