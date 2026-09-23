"""Re-identification runtime, with a stand-in model.

The real OSNet cannot be built in CI (it needs torch and a checkpoint from
Google Drive), so tests/fixtures/tiny_reid.onnx stands in: same interface — a
batch of 3x256x128 crops in, one vector per crop out — computing the mean
colour of each crop. Everything around the network is what is tested.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.ml.reid import ReidEmbedder, ReidError

MODEL = Path(__file__).parent / "fixtures" / "tiny_reid.onnx"


def _frame_with(*boxes):
    frame = np.full((400, 600, 3), 70, dtype=np.uint8)
    for (x1, y1, x2, y2), bgr in boxes:
        frame[y1:y2, x1:x2] = bgr
    return frame


def test_one_unit_vector_per_player_in_one_batch():
    embedder = ReidEmbedder(MODEL)
    assert embedder.input_size == (256, 128)
    frame = _frame_with(((50, 50, 120, 250), (40, 40, 220)), ((300, 60, 370, 260), (220, 60, 40)))

    red, blue = embedder.embed(frame, [(50, 50, 120, 250), (300, 60, 370, 260)])

    assert red is not None and blue is not None
    assert abs(float(np.linalg.norm(red)) - 1.0) < 1e-5
    assert float(red @ blue) < 0.99          # two different people
    same = embedder.embed(frame, [(52, 55, 118, 245)])[0]
    assert float(red @ same) > 0.999         # the same one again


def test_a_box_too_small_to_describe_anybody_gets_none():
    embedder = ReidEmbedder(MODEL)
    frame = _frame_with(((50, 50, 120, 250), (40, 40, 220)))
    vectors = embedder.embed(frame, [(50, 50, 120, 250), (10, 10, 14, 14), (595, 395, 700, 500)])
    assert vectors[0] is not None
    assert vectors[1] is None                # 4 x 4 px
    assert vectors[2] is None                # clipped by the frame edge to 5 x 5
    assert embedder.embed(frame, []) == []


def test_a_missing_or_foreign_model_is_refused(tmp_path):
    with pytest.raises(ReidError):
        ReidEmbedder(tmp_path / "absent.onnx")
    weights = tmp_path / "osnet.pt"
    weights.write_bytes(b"torch")
    with pytest.raises(ReidError):
        ReidEmbedder(weights)


def test_the_tracker_attaches_embeddings_to_kept_detections_only(calibration):
    from app.ml.tracking import CourtTracker
    from tests.synthetic import SyntheticPlayer, render_frame

    calls: list[int] = []
    embedder = ReidEmbedder(MODEL)
    original = embedder.embed

    def counting(frame, bboxes):
        calls.append(len(bboxes))
        return original(frame, bboxes)

    embedder.embed = counting
    tracker = CourtTracker(calibration, embedder=embedder)
    frame, detections = render_frame(calibration, [SyntheticPlayer((2.0, 5.0), 0), SyntheticPlayer((8.0, 15.0), 1)])
    # A third box far outside the court: rejected before anything is embedded.
    from app.ml.detect import Detection
    detections.append(Detection(bbox=(10.0, 10.0, 60.0, 150.0), confidence=0.9))

    tracked = tracker.update(frame_index=0, timestamp_s=0.0, detections=detections, frame=frame)

    assert calls == [2]
    assert all(obs.embedding is not None for _, obs in tracked)


def test_embeddings_survive_the_artifact_round_trip(tmp_path, calibration):
    from app.ml.artifacts import ArtifactWriter, read_observations
    from app.ml.tracking import Observation

    rng = np.random.default_rng(0)
    vector = rng.normal(size=512).astype(np.float32)
    vector /= np.linalg.norm(vector)
    obs = Observation(0, 0.0, (1, 2, 3, 4), (2, 4), (5.0, 5.0), 0.9,
                      np.full(72, 1 / 72, np.float32), embedding=vector)
    plain = Observation(1, 0.2, (1, 2, 3, 4), (2, 4), (5.0, 5.0), 0.9, np.full(72, 1 / 72, np.float32))
    with ArtifactWriter(tmp_path) as writer:
        writer.write_observations([(1, obs), (1, plain)])

    (_, restored), (_, without) = read_observations(tmp_path)
    assert float(restored.embedding @ vector) > 0.9999
    assert without.embedding is None


def test_the_pipeline_runs_without_a_reid_model(tmp_path, caplog):
    from app.ml.pipeline import AnalysisPipeline, PipelineConfig

    pipeline = AnalysisPipeline(PipelineConfig(detector_model="x.onnx", reid_model=str(tmp_path / "none.onnx")))
    with caplog.at_level("WARNING"):
        assert pipeline._embedder() is None
    assert "re-ID assente" in caplog.text
    assert AnalysisPipeline(PipelineConfig(detector_model="x.onnx", reid_model=None))._embedder() is None
    assert AnalysisPipeline(PipelineConfig(detector_model="x.onnx", reid_model=str(MODEL)))._embedder() is not None
