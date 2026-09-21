from __future__ import annotations

import numpy as np

from app.ml.detect import Detection
from app.ml.tracking import CourtTracker, histogram_distance, _torso_histogram
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
