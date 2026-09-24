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


PIPELINE_CONFIG = PipelineConfig(detector_model="stub.onnx", sample_hz=5.0)


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("artifacts") / "match"


@pytest.fixture(scope="module")
def result(synthetic_match, calibration, artifacts_dir):
    import app.ml.pipeline as pipeline_module

    original = pipeline_module.PersonDetector
    pipeline_module.PersonDetector = _BlobDetector
    try:
        pipeline = AnalysisPipeline(PIPELINE_CONFIG)
        events: list[tuple[int, str]] = []
        output = pipeline.run(
            synthetic_match,
            calibration=calibration,
            progress=lambda p, m: events.append((p, m)),
            artifacts_dir=artifacts_dir,
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


# ── Artifacts and replay ─────────────────────────────────────────────────────


def test_artifacts_are_written_during_the_run(result, artifacts_dir):
    assert (artifacts_dir / "meta.json").exists()
    assert (artifacts_dir / "tracks.jsonl").exists()
    assert (artifacts_dir / "result.json").exists()

    lines = (artifacts_dir / "tracks.jsonl").read_text().splitlines()
    total_samples = sum(p["samples"] for p in result["per_player"].values())
    # Every observation is recorded, including those from tracklets that the
    # identity stage later discarded.
    assert len(lines) >= total_samples


def test_meta_records_what_the_run_was_configured_with(result, artifacts_dir):
    import json

    meta = json.loads((artifacts_dir / "meta.json").read_text())
    assert meta["schema_version"] == 1
    assert meta["sample_hz"] == pytest.approx(5.0)
    assert meta["frames_sampled"] == result["data_quality"]["frames_sampled"]
    assert meta["calibration"]["source"] == "manual"
    assert meta["config"]["rally_speed_threshold_ms"] == PIPELINE_CONFIG.rally_speed_threshold_ms
    assert meta["inference_seconds"] >= 0


def test_replaying_the_artifacts_reproduces_the_analysis(result, artifacts_dir):
    """The point of storing artifacts: the cheap stages, re-run on the stored
    observations, must give the same answer as the original run. If they
    drift, every threshold tuned with scripts/retune.py is tuned against a
    system that does not exist."""
    from app.ml.artifacts import load_artifacts
    from app.ml.pipeline import analyse_tracklets

    artifacts = load_artifacts(artifacts_dir)
    replayed, identity, rallies = analyse_tracklets(
        tracklets=artifacts.tracklets,
        calibration=artifacts.calibration,
        config=PIPELINE_CONFIG,
        sample_hz=artifacts.sample_hz,
        frames_sampled=artifacts.frames_sampled,
        analysed_s=artifacts.analysed_s,
    )

    # Counts and segmentation must match exactly: these are decisions, and a
    # stored value must never be able to flip one.
    assert replayed["rallies"] == result["rallies"]
    assert replayed["summary"]["rallies_count"] == result["summary"]["rallies_count"]
    assert len(identity.players) == 4
    assert set(replayed["per_player"]) == set(result["per_player"])

    # Continuous quantities agree to well within displayed precision. The
    # record is lossy on purpose (see app/ml/artifacts.py), so this is a
    # tolerance, not float equality — but the tolerance is far tighter than
    # anything the UI shows or a human could act on.
    for pid, expected in result["per_player"].items():
        actual = replayed["per_player"][pid]
        assert actual["team"] == expected["team"]
        assert actual["samples"] == expected["samples"]
        assert actual["distance_m"] == pytest.approx(expected["distance_m"], abs=0.05)
        assert actual["distance_rally_m"] == pytest.approx(expected["distance_rally_m"], abs=0.05)
        assert actual["avg_speed_ms"] == pytest.approx(expected["avg_speed_ms"], abs=0.02)
        assert actual["peak_speed_ms"] == pytest.approx(expected["peak_speed_ms"], abs=0.02)
        assert actual["coverage_m2"] == pytest.approx(expected["coverage_m2"], abs=0.3)
        for zone, share in expected["zone_pct"].items():
            assert actual["zone_pct"][zone] == pytest.approx(share, abs=0.005)

    for key in ("total_rally_s", "avg_rally_s", "longest_rally_s", "active_ratio"):
        assert replayed["summary"][key] == pytest.approx(result["summary"][key], abs=0.05)


def test_replaying_the_tracker_reproduces_the_stored_tracklets(result, artifacts_dir):
    """scripts/retrack.py compares a stored run with a replay of the current
    tracker. That comparison only means something if, with unchanged code,
    the replay gives back exactly the tracklets the live pass produced."""
    from app.ml.artifacts import load_artifacts
    from app.ml.replay import replay_tracking

    artifacts = load_artifacts(artifacts_dir)
    replayed = replay_tracking(artifacts)

    def partition(tracklets):
        return sorted(tuple(o.frame_index for o in t.observations) for t in tracklets)

    assert partition(replayed) == partition(artifacts.tracklets)


def test_retuning_a_threshold_changes_the_outcome(result, artifacts_dir):
    """A sanity check on the tuning loop itself: a much stricter rally
    threshold must find fewer points than the default."""
    from dataclasses import replace

    from app.ml.artifacts import load_artifacts
    from app.ml.pipeline import analyse_tracklets

    artifacts = load_artifacts(artifacts_dir)
    strict = replace(PIPELINE_CONFIG, rally_speed_threshold_ms=6.0)
    retuned, _, rallies = analyse_tracklets(
        tracklets=artifacts.tracklets,
        calibration=artifacts.calibration,
        config=strict,
        sample_hz=artifacts.sample_hz,
        frames_sampled=artifacts.frames_sampled,
        analysed_s=artifacts.analysed_s,
    )
    assert len(rallies) < len(result["rallies"])
    assert retuned["summary"]["active_ratio"] < result["summary"]["active_ratio"]


# ── Debug overlay ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def overlay_context(result, artifacts_dir):
    from app.ml.artifacts import load_artifacts
    from app.ml.overlay import build_context
    from app.ml.pipeline import analyse_tracklets

    artifacts = load_artifacts(artifacts_dir)
    _, identity, rallies = analyse_tracklets(
        tracklets=artifacts.tracklets,
        calibration=artifacts.calibration,
        config=PIPELINE_CONFIG,
        sample_hz=artifacts.sample_hz,
        frames_sampled=artifacts.frames_sampled,
        analysed_s=artifacts.analysed_s,
    )
    return build_context(artifacts, identity.players, rallies)


def test_every_tracklet_maps_to_a_canonical_player(overlay_context):
    """An unmapped detection draws grey, which is the signal that the
    identity stage discarded it. On a clean synthetic match there should be
    almost none."""
    from app.ml.tracking import observation_key

    assert set(overlay_context.owner.values()) == {0, 1, 2, 3}
    drawn = [obs for entries in overlay_context.observations_by_frame.values() for _, obs in entries]
    owned = sum(1 for obs in drawn if observation_key(obs) in overlay_context.owner)
    assert owned / len(drawn) > 0.95


def test_rendering_a_frame_annotates_without_altering_the_source(overlay_context, calibration):
    from app.ml.overlay import render_frame

    source = _render(calibration, 5.0)
    before = source.copy()
    frame_index = next(iter(sorted(overlay_context.observations_by_frame)))

    annotated = render_frame(source, frame_index, 5.0, overlay_context)
    assert annotated.shape == source.shape
    assert np.array_equal(source, before), "render_frame non deve modificare il frame originale"
    assert not np.array_equal(annotated, source), "nessuna annotazione disegnata"


def test_overlay_handles_a_frame_with_no_observations(overlay_context, calibration):
    """Gaps in tracking are normal; the renderer must still draw the court."""
    from app.ml.overlay import render_frame

    source = _render(calibration, 5.0)
    annotated = render_frame(source, 10**9, 5.0, overlay_context)
    assert not np.array_equal(annotated, source)


def test_stills_are_written_for_frames_that_contain_players(
    synthetic_match, overlay_context, tmp_path
):
    from app.ml.overlay import render_stills

    written = render_stills(synthetic_match, overlay_context, tmp_path / "stills", count=4, end_s=10.0)
    assert len(written) == 4
    for path in written:
        assert cv2.imread(str(path)) is not None


def test_overlay_video_is_rendered_for_a_time_window(synthetic_match, overlay_context, tmp_path):
    from app.ml.overlay import render_video

    out = tmp_path / "overlay.mp4"
    written = render_video(synthetic_match, overlay_context, out, start_s=2.0, end_s=8.0, scale=0.5)
    assert written > 0
    assert out.exists() and out.stat().st_size > 0

    capture = cv2.VideoCapture(str(out))
    try:
        assert capture.isOpened()
        ok, frame = capture.read()
        assert ok and frame is not None
    finally:
        capture.release()


def test_labels_stay_inside_the_frame_for_edge_players(overlay_context, calibration):
    """Players on the far baseline sit under the banner and players at the
    right edge run off it. The renderer clamps both; a clipped id makes the
    overlay useless for spotting identity errors."""
    from dataclasses import replace

    from app.ml.overlay import render_frame

    frame_index = next(iter(sorted(overlay_context.observations_by_frame)))
    entries = overlay_context.observations_by_frame[frame_index]
    track_id, obs = entries[0]

    # Top-left corner and far-right edge: the two clamping cases.
    edge_cases = [
        replace(obs, bbox=(2.0, 1.0, 26.0, 60.0)),
        replace(obs, bbox=(W - 30.0, H - 70.0, W - 4.0, H - 6.0)),
    ]
    patched = dict(overlay_context.observations_by_frame)
    patched[frame_index] = [(track_id, o) for o in edge_cases]
    ctx = replace(overlay_context, observations_by_frame=patched)

    source = _render(calibration, 5.0)
    annotated = render_frame(source, frame_index, 5.0, ctx)
    assert annotated.shape == source.shape
    assert not np.array_equal(annotated, source)


def test_minimap_is_blended_not_pasted(overlay_context, calibration):
    """An opaque panel would hide the part of the court it covers, which on a
    real recording is exactly what needs checking."""
    from app.ml.overlay import MINIMAP_MARGIN, MINIMAP_W, render_frame

    source = _render(calibration, 5.0)
    frame_index = next(iter(sorted(overlay_context.observations_by_frame)))
    annotated = render_frame(source, frame_index, 5.0, overlay_context)

    x0 = W - MINIMAP_W - MINIMAP_MARGIN
    patch_before = source[MINIMAP_MARGIN + 40: MINIMAP_MARGIN + 60, x0 + 10: x0 + 60]
    patch_after = annotated[MINIMAP_MARGIN + 40: MINIMAP_MARGIN + 60, x0 + 10: x0 + 60]
    assert not np.array_equal(patch_before, patch_after)

    # The panel background must be a mix of the fill and the frame beneath it,
    # never the fill alone — that is what "blended" means here.
    from app.ml.overlay import MINIMAP_ALPHA

    fill = np.array([40, 60, 45], dtype=np.float32)
    expected = MINIMAP_ALPHA * fill + (1 - MINIMAP_ALPHA) * patch_before.astype(np.float32)
    assert np.allclose(patch_after.astype(np.float32), expected, atol=1.5)
    assert not np.allclose(patch_after.astype(np.float32), fill, atol=1.0)


def test_meta_is_written_before_the_pass_and_marked_incomplete(
    synthetic_match, calibration, tmp_path
):
    """Il caso che ha tratto in inganno durante la prima analisi reale: leggere
    la cartella mentre l'analisi gira dava i totali della run precedente
    accanto a un tracks.jsonl appena troncato. Ora, in ogni istante della
    passata, meta.json descrive *questa* run e dichiara di non essere finita."""
    import json

    import app.ml.pipeline as pipeline_module

    directory = tmp_path / "artifacts"
    seen: list[dict] = []

    class _WatchingDetector(_BlobDetector):
        """Fotografa meta.json durante la passata, non solo alla fine."""

        def detect(self, frame, roi=None):
            meta_file = directory / "meta.json"
            if meta_file.exists() and len(seen) < 3:
                seen.append(json.loads(meta_file.read_text()))
            return super().detect(frame, roi=roi)

    original = pipeline_module.PersonDetector
    pipeline_module.PersonDetector = _WatchingDetector
    try:
        AnalysisPipeline(PIPELINE_CONFIG).run(
            synthetic_match, calibration=calibration, artifacts_dir=directory
        )
    finally:
        pipeline_module.PersonDetector = original

    assert seen, "meta.json non esisteva durante la passata"
    for snapshot in seen:
        assert snapshot["complete"] is False
        assert snapshot["frames_sampled"] is None
        # Ma la parte già nota c'è, così la cartella è leggibile fin da subito.
        assert snapshot["calibration"]["source"] == "manual"
        assert snapshot["sample_hz"] == pytest.approx(5.0)

    final = json.loads((directory / "meta.json").read_text())
    assert final["complete"] is True
    assert final["frames_sampled"] > 0


def test_a_reanalysis_never_mixes_two_generations(synthetic_match, calibration, tmp_path):
    """Rianalizzare la stessa partita deve sostituire gli artefatti, non
    sovrapporli: nessun totale della run precedente può sopravvivere accanto
    alle tracce della nuova."""
    import json

    import app.ml.pipeline as pipeline_module

    directory = tmp_path / "artifacts"
    original = pipeline_module.PersonDetector
    pipeline_module.PersonDetector = _BlobDetector
    try:
        pipeline = AnalysisPipeline(PIPELINE_CONFIG)
        pipeline.run(synthetic_match, calibration=calibration, artifacts_dir=directory)
        first = json.loads((directory / "meta.json").read_text())

        pipeline.run(synthetic_match, calibration=calibration, artifacts_dir=directory)
        second = json.loads((directory / "meta.json").read_text())
    finally:
        pipeline_module.PersonDetector = original

    assert first["frames_sampled"] == second["frames_sampled"]
    lines = (directory / "tracks.jsonl").read_text().splitlines()
    # Troncato e riscritto, non accodato.
    assert len(lines) == len({(json.loads(line)["f"], json.loads(line)["id"]) for line in lines})


def test_a_run_with_a_re_id_model_stores_embeddings(synthetic_match, calibration, tmp_path):
    """With a re-ID model installed every kept detection carries an
    embedding, into the artifacts and back, and the run names the model in
    meta.json. The stand-in model is tests/fixtures/tiny_reid.onnx."""
    import json
    from dataclasses import replace

    import app.ml.pipeline as pipeline_module
    from app.ml.artifacts import read_observations

    model = Path(__file__).parent / "fixtures" / "tiny_reid.onnx"
    original = pipeline_module.PersonDetector
    pipeline_module.PersonDetector = _BlobDetector
    try:
        output = AnalysisPipeline(replace(PIPELINE_CONFIG, reid_model=str(model))).run(
            synthetic_match, calibration=calibration, artifacts_dir=tmp_path / "run",
        )
    finally:
        pipeline_module.PersonDetector = original

    observations = read_observations(tmp_path / "run")
    assert observations and all(obs.embedding is not None for _, obs in observations)
    assert json.loads((tmp_path / "run" / "meta.json").read_text())["config"]["reid_model"] == model.name
    assert len(output["per_player"]) == 4


# ── Player thumbnails ────────────────────────────────────────────────────────


def _crop_obs(frame_index, bbox, confidence=0.9):
    from app.ml.tracking import Observation

    x1, y1, x2, y2 = bbox
    return Observation(frame_index, frame_index / 5.0, bbox, ((x1 + x2) / 2, y2),
                       (5.0, float(frame_index)), confidence, np.full(72, 1 / 72, np.float32))


def test_a_player_far_from_the_camera_still_gets_a_thumbnail():
    """Candidates used to be capped at 32 for the whole match, ranked by box
    size. On a fragmented match the near players' many tracks filled the
    cap, and the far players — small boxes — ended up with no picture."""
    from app.ml.identity import PlayerTrack
    from app.ml.pipeline import _collect_crops, _encode_player_crops

    frame = np.full((1080, 1920, 3), 90, dtype=np.uint8)
    crops: dict = {}
    for track_id in range(40):                       # near side: big boxes
        _collect_crops(crops, [(track_id, _crop_obs(track_id, (100.0 + 40 * track_id, 600.0, 180.0 + 40 * track_id, 900.0)))], frame)
    far = _crop_obs(99, (900.0, 200.0, 925.0, 260.0))  # 60 px tall
    _collect_crops(crops, [(500, far)], frame)

    player = PlayerTrack(player_id=2, team=1, observations=[far], source_tracklets=[500])
    thumbnails = _encode_player_crops([player], crops)

    assert 2 in thumbnails


def test_a_thumbnail_keeps_the_player_proportions():
    """The box used to be stretched to the thumbnail's 2:3, squashing or
    elongating the player; it is framed at 2:3 around the player instead."""
    from app.ml.pipeline import _CROP_H, _CROP_W, _thumbnail

    frame = np.full((1080, 1920, 3), 90, dtype=np.uint8)
    frame[300:700, 900:1000] = (40, 40, 220)          # a 100 x 400 player: 1:4
    jpeg = _thumbnail(frame, (900.0, 300.0, 1000.0, 700.0))
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    assert image.shape[:2] == (_CROP_H, _CROP_W)
    red = (image[:, :, 2] > 150) & (image[:, :, 0] < 100)
    rows, cols = np.where(red)
    ratio = (cols.max() - cols.min() + 1) / (rows.max() - rows.min() + 1)
    assert ratio == pytest.approx(0.25, rel=0.1)


def test_the_players_boxes_for_the_video_are_ready_at_the_end(result, artifacts_dir):
    """Written by the analysis from the identity it computed, so the web
    player opens at once, and says that the boxes are the players of the
    statistics."""
    import gzip
    import json

    from app.ml.player_boxes import player_boxes_gz

    cached = list(artifacts_dir.glob("players-*.json.gz"))
    assert len(cached) == 1
    # The API finds it from the statistics, without recomputing anything.
    data = player_boxes_gz(artifacts_dir, result["per_player"], default_speed_ms=8.0)
    assert data == cached[0].read_bytes()
    payload = json.loads(gzip.decompress(data))
    assert payload["matches_stats"] is True
    assert {str(p["id"]) for p in payload["players"]} == set(result["per_player"])
