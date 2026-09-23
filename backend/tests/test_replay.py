from __future__ import annotations

import pytest

from app.ml.artifacts import ArtifactWriter, load_artifacts, read_observations
from app.ml.replay import replay_tracking
from app.ml.tracking import CourtTracker
from tests.synthetic import SyntheticPlayer, render_frame

FPS, STEP = 30.0, 6          # 5 Hz, as in production


def _partition(tracklets):
    return sorted(tuple(o.frame_index for o in t.observations) for t in tracklets)


def _live_run(calibration, directory, frames):
    """Run the live tracker and store its output exactly as the pipeline does."""
    tracker = CourtTracker(calibration, max_speed_ms=8.0, max_age_s=1.2)
    with ArtifactWriter(directory) as writer:
        for i, players in enumerate(frames):
            index = i * STEP
            frame, detections = render_frame(calibration, players)
            tracked = tracker.update(
                frame_index=index, timestamp_s=index / FPS, detections=detections, frame=frame,
            )
            writer.write_observations(tracked)
        writer.write_meta({
            "calibration": calibration.to_dict(),
            "video": {"fps": round(FPS, 3)},
            "sample_hz": FPS / STEP,
            "sample_step": STEP,
            "frames_sampled": len(frames),
            "analysed_s": (len(frames) - 1) * STEP / FPS,
            "min_observations": 3,
            "config": {"max_player_speed_ms": 8.0, "track_max_age_s": 1.2},
        })
    return tracker.finish()


def test_a_replay_reproduces_the_live_tracker_across_empty_frames(tmp_path, calibration):
    """Tracks die, players vanish, a one-off detection appears: the replay
    must still hand back the live tracker's tracklets, one for one."""
    frames = []
    for i in range(60):
        a = SyntheticPlayer((2.0 + 0.05 * i, 4.0), 0)
        b = SyntheticPlayer((8.0, 16.0 - 0.05 * i), 1)
        if 20 <= i < 30:
            frames.append([])                       # everybody lost for 2 s
        elif i == 40:
            frames.append([a, b, SyntheticPlayer((5.0, 10.0), 2)])   # a one-off
        else:
            frames.append([a, b])

    live = _live_run(calibration, tmp_path / "run", frames)
    replayed = replay_tracking(load_artifacts(tmp_path / "run"))

    assert len(live) == 4          # the 2 s gap exceeds max_age: each player splits
    assert _partition(replayed) == _partition(live)


def test_every_sampled_frame_is_replayed_even_when_empty(tmp_path, calibration, monkeypatch):
    """Frames without detections are not stored, but the tracker takes its
    sampling period from consecutive calls: skipping them would stretch the
    period and change the cost scale. They must be fed, at their own time."""
    frames = [[SyntheticPlayer((2.0, 4.0), 0)] for _ in range(12)]
    for i in (3, 4, 8):
        frames[i] = []
    _live_run(calibration, tmp_path / "run", frames)

    seen: list[float] = []
    original = CourtTracker.update_observations

    def record(self, timestamp_s, observations):
        seen.append(timestamp_s)
        return original(self, timestamp_s, observations)

    monkeypatch.setattr(CourtTracker, "update_observations", record)
    replay_tracking(load_artifacts(tmp_path / "run"))

    assert seen == [pytest.approx(i * STEP / FPS, abs=1e-6) for i in range(12)]


def test_the_replay_reads_short_tracks_too(tmp_path, calibration):
    """The one-off detection is dropped from the tracklets, but it was part of
    the tracker's input and must be part of the replay's."""
    frames = [[SyntheticPlayer((2.0, 4.0), 0)] for _ in range(10)]
    frames[5] = frames[5] + [SyntheticPlayer((8.0, 16.0), 1)]
    _live_run(calibration, tmp_path / "run", frames)

    assert len(load_artifacts(tmp_path / "run").tracklets) == 1
    assert len(read_observations(tmp_path / "run")) == 11


def test_parameters_can_be_overridden(tmp_path, calibration):
    """--max-age in scripts/retrack.py: a longer memory bridges the gap."""
    frames = [[SyntheticPlayer((2.0, 4.0), 0)] for _ in range(10)]
    frames += [[] for _ in range(8)]                # 1.6 s without the player
    frames += [[SyntheticPlayer((2.5, 4.0), 0)] for _ in range(10)]
    _live_run(calibration, tmp_path / "run", frames)
    artifacts = load_artifacts(tmp_path / "run")

    assert len(replay_tracking(artifacts)) == 2
    assert len(replay_tracking(artifacts, max_age_s=2.0)) == 1
