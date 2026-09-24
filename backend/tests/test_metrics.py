from __future__ import annotations

import pytest

from app.ml.identity import IdentityResult, PlayerTrack
from app.ml.metrics import MetricsInput, compute_metrics
from app.ml.rallies import Rally
from tests.synthetic import make_observation, walk

HZ = 5.0


def _input(calibration, players, rallies=None, analysed_s=40.0, sample_hz=HZ, warnings=None):
    return MetricsInput(
        players=players,
        rallies=rallies or [],
        identity=IdentityResult(players, warnings or [], 0, len(players), 0),
        calibration=calibration,
        sample_hz=sample_hz,
        frames_sampled=int(analysed_s * sample_hz),
        analysed_s=analysed_s,
        max_speed_ms=8.0,
        detector_model="yolov8n.onnx",
    )


def test_distance_matches_the_path_actually_walked(calibration):
    # 10 metres straight down the court over 10 seconds.
    observations = walk(calibration, 0, (5.0, 4.0), (5.0, 14.0), 0.0, 10.0, hz=HZ)
    player = PlayerTrack(0, 0, observations, [0])
    result = compute_metrics(_input(calibration, [player], analysed_s=10.0))
    assert result["per_player"]["0"]["distance_m"] == pytest.approx(10.0, abs=0.3)


def test_teleport_steps_are_excluded_from_distance(calibration):
    observations = walk(calibration, 0, (5.0, 4.0), (5.0, 8.0), 0.0, 10.0, hz=HZ)
    # A single identity swap would otherwise add ~15 m to this player's total.
    observations.insert(
        len(observations) // 2,
        make_observation(calibration, (9.5, 19.0), observations[len(observations) // 2].timestamp_s + 0.01, 0),
    )
    observations.sort(key=lambda o: o.timestamp_s)
    result = compute_metrics(_input(calibration, [PlayerTrack(0, 0, observations, [0])], analysed_s=10.0))
    stats = result["per_player"]["0"]
    assert stats["distance_m"] < 6.0
    assert stats["rejected_steps"] >= 1


def test_gaps_in_tracking_do_not_count_as_movement(calibration):
    """A player lost for 10 s and found elsewhere has not run that distance."""
    first = walk(calibration, 0, (2.0, 4.0), (3.0, 5.0), 0.0, 5.0, hz=HZ)
    second = walk(calibration, 0, (8.0, 16.0), (8.5, 16.5), 15.0, 20.0, hz=HZ)
    player = PlayerTrack(0, 0, first + second, [0, 1])
    result = compute_metrics(_input(calibration, [player], analysed_s=20.0))
    # Only the two walked segments, never the 15 m jump between them.
    assert result["per_player"]["0"]["distance_m"] < 4.0


def test_zone_distribution_reflects_position(calibration):
    at_net = walk(calibration, 0, (5.0, 9.0), (5.0, 11.0), 0.0, 20.0, hz=HZ)
    result = compute_metrics(_input(calibration, [PlayerTrack(0, 0, at_net, [0])], analysed_s=20.0))
    zones = result["per_player"]["0"]["zone_pct"]
    assert zones["net"] == pytest.approx(1.0, abs=0.01)
    assert zones["back"] == 0.0


def test_tracked_ratio_exposes_partial_coverage(calibration):
    # Ten seconds of tracking inside a forty-second match.
    observations = walk(calibration, 0, (5.0, 4.0), (5.0, 6.0), 0.0, 10.0, hz=HZ)
    result = compute_metrics(_input(calibration, [PlayerTrack(0, 0, observations, [0])], analysed_s=40.0))
    stats = result["per_player"]["0"]
    assert stats["tracked_ratio"] == pytest.approx(0.25, abs=0.02)
    assert any("meno della metà" in w for w in result["data_quality"]["warnings"])


def test_heatmap_weights_sum_to_one(calibration):
    observations = walk(calibration, 0, (2.0, 4.0), (8.0, 16.0), 0.0, 20.0, hz=HZ)
    result = compute_metrics(_input(calibration, [PlayerTrack(0, 0, observations, [0])], analysed_s=20.0))
    points = result["heatmaps"]["0"]
    assert points
    assert sum(p[2] for p in points) == pytest.approx(1.0, abs=0.02)
    assert all(0.0 <= p[0] <= 10.0 and 0.0 <= p[1] <= 20.0 for p in points)


def test_rally_summary_is_computed(calibration):
    observations = walk(calibration, 0, (5.0, 4.0), (5.0, 14.0), 0.0, 40.0, hz=HZ)
    rallies = [Rally(0, 0.0, 10.0, 4), Rally(1, 20.0, 26.0, 4)]
    result = compute_metrics(
        _input(calibration, [PlayerTrack(0, 0, observations, [0])], rallies=rallies, analysed_s=40.0)
    )
    summary = result["summary"]
    assert summary["rallies_count"] == 2
    assert summary["total_rally_s"] == pytest.approx(16.0)
    assert summary["longest_rally_s"] == pytest.approx(10.0)
    assert summary["active_ratio"] == pytest.approx(0.4, abs=0.01)


def test_ball_derived_metrics_are_absent_and_declared(calibration):
    """The point of the rewrite: no shot types, no winners, no ball speed —
    and the omission is stated in the payload, not left for the reader to
    notice."""
    observations = walk(calibration, 0, (5.0, 4.0), (5.0, 6.0), 0.0, 10.0, hz=HZ)
    result = compute_metrics(_input(calibration, [PlayerTrack(0, 0, observations, [0])], analysed_s=10.0))

    stats = result["per_player"]["0"]
    for absent in ("shots", "winners", "errors", "ball_speed_ms"):
        assert absent not in stats

    quality = result["data_quality"]
    assert quality["metrics_tier"] == 1
    assert "vincenti/errori" in quality["excluded_metrics"]


def test_low_sampling_rate_is_declared(calibration):
    observations = walk(calibration, 0, (5.0, 4.0), (5.0, 6.0), 0.0, 10.0, hz=3.0)
    result = compute_metrics(
        _input(calibration, [PlayerTrack(0, 0, observations, [0])], analysed_s=10.0, sample_hz=3.0)
    )
    assert any("sottostimata" in w for w in result["data_quality"]["warnings"])


def test_net_error_above_half_a_metre_is_flagged():
    import numpy as np

    from app.ml.court import CalibrationSource, build_calibration

    corners = [[600, 300], [1320, 300], [1800, 980], [120, 980]]
    base = build_calibration(corners, (1920, 1080), CalibrationSource.MANUAL)
    off_net = base.court_to_pixels(np.array([[0.0, 12.0], [10.0, 12.0]]))
    calib = build_calibration(corners, (1920, 1080), CalibrationSource.MANUAL, net_px=off_net)

    observations = walk(calib, 0, (5.0, 4.0), (5.0, 6.0), 0.0, 10.0, hz=HZ)
    result = compute_metrics(_input(calib, [PlayerTrack(0, 0, observations, [0])], analysed_s=10.0))
    assert any("rete indicata" in w for w in result["data_quality"]["warnings"])


def test_the_side_of_the_pair_is_reported_when_known(calibration):
    from app.ml.roles import DRIVE

    observations = walk(calibration, 0, (7.0, 5.0), (8.0, 5.0), 0.0, 10.0)
    by_role = PlayerTrack(0, 0, observations, [0], role=DRIVE)
    by_link = PlayerTrack(1, 0, observations, [1])
    result = compute_metrics(_input(calibration, [by_role, by_link], analysed_s=10.0))
    assert result["per_player"]["0"]["role"] == "drive"
    assert result["per_player"]["1"]["role"] is None
