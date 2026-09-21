from __future__ import annotations

import pytest

from app.ml.identity import PlayerTrack
from app.ml.rallies import detect_rallies
from tests.synthetic import make_observation

HZ = 5.0


def _moving(calibration, shirt: int, x: float, y0: float, t0: float, t1: float, speed: float):
    """A player oscillating at a known mean speed, sampled at HZ."""
    observations = []
    steps = int((t1 - t0) * HZ)
    y = y0
    direction = 1.0
    for i in range(steps + 1):
        t = t0 + i / HZ
        observations.append(make_observation(calibration, (x, y), t, shirt))
        y += direction * speed / HZ
        if not 1.0 <= y <= 19.0:
            direction *= -1.0
    return observations


def _players(calibration, segments):
    """segments: [(t0, t1, speed), ...] applied to all four players."""
    tracks = []
    for shirt, x in enumerate((2.0, 8.0, 3.0, 7.0)):
        observations = []
        base_y = 4.0 if shirt < 2 else 16.0
        for t0, t1, speed in segments:
            observations += _moving(calibration, shirt, x, base_y, t0, t1, speed)
        observations.sort(key=lambda o: o.timestamp_s)
        tracks.append(PlayerTrack(shirt, 0 if shirt < 2 else 1, observations, [shirt]))
    return tracks


def test_active_play_is_separated_from_the_pause_between_points(calibration):
    players = _players(
        calibration,
        [(0.0, 12.0, 3.0), (12.2, 25.0, 0.15), (25.2, 40.0, 3.0)],
    )
    rallies = detect_rallies(players, sample_hz=HZ)

    assert len(rallies) == 2
    assert rallies[0].start_s < 2.0 and rallies[0].end_s == pytest.approx(12.0, abs=1.5)
    assert rallies[1].start_s == pytest.approx(25.2, abs=1.5)


def test_short_bursts_are_not_rallies(calibration):
    """Someone picking up a ball must not count as a point."""
    players = _players(calibration, [(0.0, 1.5, 3.0), (1.6, 30.0, 0.1)])
    assert detect_rallies(players, sample_hz=HZ, min_duration_s=3.0) == []


def test_a_brief_dip_does_not_split_one_rally_in_two(calibration):
    players = _players(
        calibration,
        [(0.0, 10.0, 3.0), (10.2, 11.0, 0.2), (11.2, 22.0, 3.0)],
    )
    rallies = detect_rallies(players, sample_hz=HZ, merge_gap_s=1.5)
    assert len(rallies) == 1
    assert rallies[0].duration_s > 18.0


def test_a_standing_still_match_produces_no_rallies(calibration):
    players = _players(calibration, [(0.0, 60.0, 0.1)])
    assert detect_rallies(players, sample_hz=HZ) == []


def test_segmentation_needs_at_least_two_tracked_players(calibration):
    """One player moving is a warm-up, not a point."""
    solo = _players(calibration, [(0.0, 30.0, 3.0)])[:1]
    assert detect_rallies(solo, sample_hz=HZ, min_players=2) == []


def test_implausible_speeds_do_not_create_rallies(calibration):
    """An identity swap teleports a player across the court. A mean filter
    would read that as furious activity; the median filter must not."""
    players = _players(calibration, [(0.0, 30.0, 0.1)])
    for track in players:
        # Inject a single 20 m/s jump in the middle of a quiet period.
        middle = len(track.observations) // 2
        bad = make_observation(
            calibration, (9.0, 18.0), track.observations[middle].timestamp_s, 0
        )
        track.observations[middle] = bad
    assert detect_rallies(players, sample_hz=HZ, max_speed_ms=8.0) == []


def test_no_players_is_handled(calibration):
    assert detect_rallies([], sample_hz=HZ) == []
