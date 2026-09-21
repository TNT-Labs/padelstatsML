"""End-to-end pipeline test on a synthetic match.

A real ONNX detector is not available in CI, so `PersonDetector` is replaced
by a colour-blob finder over the same synthetic frames. Everything else —
sampling, tracking, identity, rally segmentation, metrics, thumbnails — is
the production code path, wired exactly as the worker wires it.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ml.court import CalibrationSource, build_calibration
from app.ml.detect import Detection
from app.ml.pipeline import AnalysisPipeline, PipelineConfig

W, H, FPS = 640, 360, 30
DURATION_S = 30
BACKGROUND = 70

SHIRTS = [(40, 40, 220), (220, 60, 40), (40, 200, 220), (200, 40, 200)]
# Two pairs holding their halves, as padel teams do.
HOME = [(2.5, 5.0), (7.5, 5.0), (2.5, 15.0), (7.5, 15.0)]

# The middle third of the clip is the changeover: everybody stands still.
RALLY_1 = (0.0, 11.0)
PAUSE = (11.0, 19.0)
RALLY_2 = (19.0, 30.0)


@pytest.fixture(scope="module")
def calibration():
    return build_calibration(
        [[200, 90], [440, 90], [600, 330], [40, 330]], (W, H), CalibrationSource.MANUAL
    )


def _positions(t: float) -> list[tuple[float, float]]:
    """Four players: moving during the rallies, nearly still in the pause."""
    active = RALLY_1[0] <= t < RALLY_1[1] or RALLY_2[0] <= t < RALLY_2[1]
    amplitude = 2.2 if active else 0.02
    out = []
    for i, (x, y) in enumerate(HOME):
        phase = i * 1.7
        out.append(
            (
                x + amplitude * math.sin(1.3 * t + phase) * 0.6,
                y + amplitude * math.cos(1.1 * t + phase),
            )
        )
    return out


def _render(calibration, t: float) -> np.ndarray:
    frame = np.full((H, W, 3), BACKGROUND, dtype=np.uint8)
    for i, court_xy in enumerate(_positions(t)):
        foot = calibration.court_to_pixels(np.array([court_xy], dtype=np.float32))[0]
        fx, fy = float(foot[0]), float(foot[1])
        height = 34.0 + 34.0 * (fy / H)
        width = height * 0.4
        x1 = int(np.clip(fx - width / 2, 0, W - 1))
        x2 = int(np.clip(fx + width / 2, 1, W))
        y1 = int(np.clip(fy - height, 0, H - 1))
        y2 = int(np.clip(fy, 1, H))
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = SHIRTS[i]
    return frame


@pytest.fixture(scope="module")
def synthetic_match(calibration, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("match") / "match.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        pytest.skip("Questa build di OpenCV non sa scrivere file MP4")
    for i in range(DURATION_S * FPS):
        writer.write(_render(calibration, i / FPS))
    writer.release()
    return path


class _BlobDetector:
    """Stands in for the ONNX model: finds the coloured rectangles.

    Mirrors the real detector's contract, including returning boxes in
    full-frame coordinates when a ROI crop was used.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        self.imgsz = (480, 480)

    def detect(self, frame: np.ndarray, roi=None) -> list[Detection]:
        offset = (0.0, 0.0)
        work = frame
        if roi is not None:
            x1, y1, x2, y2 = roi
            work = frame[y1:y2, x1:x2]
            offset = (float(x1), float(y1))
            if work.size == 0:
                work, offset = frame, (0.0, 0.0)

        diff = cv2.absdiff(work, np.full_like(work, BACKGROUND))
        mask = (diff.max(axis=2) > 40).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: list[Detection] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 5 or h < 15:
                continue
            out.append(
                Detection(
                    bbox=(x + offset[0], y + offset[1], x + w + offset[0], y + h + offset[1]),
                    confidence=0.9,
                )
            )
        return out


@pytest.fixture(scope="module")
def result(synthetic_match, calibration):
    import app.ml.pipeline as pipeline_module

    original = pipeline_module.PersonDetector
    pipeline_module.PersonDetector = _BlobDetector
    try:
        pipeline = AnalysisPipeline(
            PipelineConfig(detector_model="stub.onnx", sample_hz=5.0)
        )
        events: list[tuple[int, str]] = []
        output = pipeline.run(
            synthetic_match, calibration=calibration, progress=lambda p, m: events.append((p, m))
        )
        output["_progress"] = events
        return output
    finally:
        pipeline_module.PersonDetector = original


def test_four_players_are_identified(result):
    assert len(result["per_player"]) == 4
    assert sorted(result["per_player"]) == ["0", "1", "2", "3"]


def test_teams_are_two_and_two(result):
    teams = [p["team"] for p in result["per_player"].values()]
    assert sorted(teams) == [0, 0, 1, 1]


def test_players_are_tracked_for_most_of_the_match(result):
    for stats in result["per_player"].values():
        assert stats["tracked_ratio"] > 0.9


def test_distance_is_in_a_physically_sane_range(result):
    """Thirty seconds of padel-like movement, not a marathon and not zero."""
    for stats in result["per_player"].values():
        assert 5.0 < stats["distance_m"] < 200.0
        assert stats["peak_speed_ms"] <= 8.0


def test_the_two_rallies_are_found_and_the_pause_is_not(result):
    rallies = result["rallies"]
    assert len(rallies) == 2
    assert rallies[0]["start_s"] < 3.0
    assert rallies[0]["end_s"] == pytest.approx(RALLY_1[1], abs=2.5)
    assert rallies[1]["start_s"] == pytest.approx(RALLY_2[0], abs=2.5)
    # The changeover must not be inside any rally.
    midpoint = (PAUSE[0] + PAUSE[1]) / 2
    assert not any(r["start_s"] <= midpoint <= r["end_s"] for r in rallies)


def test_summary_is_consistent_with_the_rally_list(result):
    summary = result["summary"]
    assert summary["rallies_count"] == len(result["rallies"])
    assert summary["total_rally_s"] == pytest.approx(
        sum(r["duration_s"] for r in result["rallies"]), abs=0.2
    )
    assert 0.0 < summary["active_ratio"] < 1.0


def test_heatmaps_stay_inside_the_court(result):
    for points in result["heatmaps"].values():
        assert points
        assert all(0.0 <= x <= 10.0 and 0.0 <= y <= 20.0 for x, y, _ in points)


def test_every_player_gets_a_thumbnail(result):
    crops = result["player_crops_data"]
    assert len(crops) == 4
    for image_bytes in crops.values():
        decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape == (240, 160, 3)


def test_data_quality_is_reported(result):
    quality = result["data_quality"]
    assert quality["calibration_source"] == "manual"
    assert quality["metrics_tier"] == 1
    assert quality["players_found"] == 4
    assert quality["frames_sampled"] > 100


def test_progress_is_monotonic_and_reaches_one_hundred(result):
    percents = [p for p, _ in result["_progress"]]
    assert percents == sorted(percents)
    assert percents[-1] == 100
