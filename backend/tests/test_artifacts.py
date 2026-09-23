"""Artifacts must be a faithful, replayable record of a run.

The whole value of storing them is that re-running the cheap stages on the
stored observations gives *the same answer* as the original analysis. If the
replay drifts, every threshold tuned with scripts/retune.py is tuned against
a system that does not exist.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.ml.artifacts import ArtifactWriter, load_artifacts, resolve_artifacts_dir
from app.ml.tracking import Tracklet
from tests.synthetic import walk


@pytest.fixture
def written(tmp_path, calibration):
    """Two tracklets written out exactly as the pipeline would write them."""
    tracklets = [
        Tracklet(id=3, observations=walk(calibration, 0, (2.0, 4.0), (3.0, 6.0), 0.0, 10.0)),
        Tracklet(id=7, observations=walk(calibration, 1, (8.0, 4.0), (7.0, 6.0), 0.0, 10.0)),
    ]
    directory = tmp_path / "artifacts"
    with ArtifactWriter(directory) as writer:
        # Interleaved by frame, the way the tracker emits them.
        pairs = [(t.id, o) for t in tracklets for o in t.observations]
        pairs.sort(key=lambda p: p[1].timestamp_s)
        writer.write_observations(pairs)
        writer.write_meta(
            {
                "calibration": calibration.to_dict(),
                "sample_hz": 5.0,
                "frames_sampled": len(pairs) // 2,
                "analysed_s": 10.0,
                "min_observations": 3,
            }
        )
    return directory, tracklets


def test_observations_survive_the_round_trip(written):
    directory, original = written
    loaded = load_artifacts(directory)

    assert {t.id for t in loaded.tracklets} == {3, 7}
    by_id = {t.id: t for t in loaded.tracklets}
    for source in original:
        restored = by_id[source.id]
        assert len(restored) == len(source)
        for a, b in zip(source.observations, restored.observations):
            assert a.frame_index == b.frame_index
            assert b.timestamp_s == pytest.approx(a.timestamp_s, abs=1e-3)
            assert b.foot_court == pytest.approx(a.foot_court, abs=1e-3)
            assert b.foot_px == pytest.approx(a.foot_px, abs=0.1)


def test_colour_signatures_survive_precisely_enough_to_link(written):
    """Identity linking compares histograms against a 0.45 veto, so the
    stored precision has to be far finer than that."""
    from app.ml.tracking import histogram_distance

    directory, original = written
    loaded = load_artifacts(directory)
    by_id = {t.id: t for t in loaded.tracklets}

    for source in original:
        distance = histogram_distance(source.color_signature(), by_id[source.id].color_signature())
        assert distance < 1e-3


def test_calibration_is_restored_identically(written, calibration):
    directory, _ = written
    loaded = load_artifacts(directory)
    assert np.allclose(loaded.calibration.homography, calibration.homography, atol=1e-9)
    assert loaded.calibration.source == calibration.source


def test_short_tracklets_are_dropped_the_same_way_the_pipeline_drops_them(tmp_path, calibration):
    directory = tmp_path / "artifacts"
    keep = Tracklet(id=1, observations=walk(calibration, 0, (2.0, 4.0), (3.0, 6.0), 0.0, 5.0))
    drop = Tracklet(id=2, observations=walk(calibration, 1, (8.0, 4.0), (8.1, 4.1), 0.0, 0.2))
    assert len(drop) < 3

    with ArtifactWriter(directory) as writer:
        writer.write_observations([(t.id, o) for t in (keep, drop) for o in t.observations])
        writer.write_meta(
            {
                "calibration": calibration.to_dict(),
                "sample_hz": 5.0,
                "frames_sampled": 26,
                "analysed_s": 5.0,
                "min_observations": 3,
            }
        )

    loaded = load_artifacts(directory)
    assert [t.id for t in loaded.tracklets] == [1]


def test_a_future_schema_version_is_refused(written):
    directory, _ = written
    meta_file = directory / "meta.json"
    meta = json.loads(meta_file.read_text())
    meta["schema_version"] = 99
    meta_file.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="rianalizza"):
        load_artifacts(directory)


def test_a_corrupt_line_names_its_line_number(written):
    directory, _ = written
    tracks = directory / "tracks.jsonl"
    lines = tracks.read_text().splitlines()
    lines[2] = '{"f": 1}'
    tracks.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match=r"tracks\.jsonl:3"):
        load_artifacts(directory)


def test_missing_artifacts_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="meta.json"):
        load_artifacts(tmp_path / "nothing")


def test_result_is_stored_without_the_raw_image_bytes(tmp_path, calibration):
    """player_crops_data holds JPEG bytes, which are neither JSON-serialisable
    nor useful here — the thumbnails are saved as files by the worker."""
    directory = tmp_path / "artifacts"
    writer = ArtifactWriter(directory)
    writer.write_result({"summary": {"rallies_count": 3}, "player_crops_data": {0: b"\xff\xd8"}})

    stored = json.loads((directory / "result.json").read_text())
    assert stored["summary"]["rallies_count"] == 3
    assert "player_crops_data" not in stored


# ── Rianalisi: la cartella non deve mai mescolare due generazioni ────────────

def test_reopening_clears_the_previous_run(written, calibration):
    """Una rianalisi tronca tracks.jsonl e lo riempie da capo. Se meta.json e
    result.json della run precedente restassero lì, chi legge la cartella a
    metà lavoro accosterebbe i totali di una run al file tracce di un'altra —
    esattamente come è successo leggendo 12.765 frame accanto a 664."""
    directory, _ = written
    (directory / "result.json").write_text('{"summary": {"rallies_count": 7}}')
    assert (directory / "meta.json").exists()

    with ArtifactWriter(directory):
        assert not (directory / "meta.json").exists()
        assert not (directory / "result.json").exists()
        assert (directory / "tracks.jsonl").read_text() == ""


def test_a_run_without_the_flag_counts_as_complete(written):
    """Gli artefatti scritti prima che il flag esistesse venivano salvati solo
    a fine passata: la sua assenza significa concluso."""
    directory, _ = written
    meta_file = directory / "meta.json"
    meta = json.loads(meta_file.read_text())
    meta.pop("complete", None)
    meta_file.write_text(json.dumps(meta))

    assert load_artifacts(directory).complete is True


def test_an_incomplete_run_is_reported_as_such(written, calibration):
    directory, _ = written
    meta = json.loads((directory / "meta.json").read_text())
    meta["complete"] = False
    (directory / "meta.json").write_text(json.dumps(meta))

    assert load_artifacts(directory).complete is False


# ── Individuare una partita senza conoscerne l'UUID ──────────────────────────

def _match_dirs(root, *names):
    for name in names:
        (root / name).mkdir(parents=True)
    return root


def test_latest_picks_the_most_recent_run(tmp_path):
    import os
    import time

    root = _match_dirs(tmp_path, "aaaa1111-old", "bbbb2222-new")
    old = time.time() - 3600
    os.utime(root / "aaaa1111-old", (old, old))

    assert resolve_artifacts_dir("latest", root).name == "bbbb2222-new"


def test_an_unambiguous_prefix_is_enough(tmp_path):
    """Gli id sono UUID e l'interfaccia non li mostrava: chiedere di digitarne
    uno per intero equivale a chiedere di non usare lo strumento."""
    root = _match_dirs(tmp_path, "aaaa1111-2222-3333", "bbbb4444-5555-6666")
    assert resolve_artifacts_dir("aaaa", root).name == "aaaa1111-2222-3333"
    assert resolve_artifacts_dir("aaaa1111-2222-3333", root).name == "aaaa1111-2222-3333"


def test_an_ambiguous_prefix_is_refused_with_the_candidates(tmp_path):
    root = _match_dirs(tmp_path, "aaaa1111", "aaaa2222")
    with pytest.raises(ValueError, match="più partite"):
        resolve_artifacts_dir("aaaa", root)


def test_an_unknown_id_lists_what_is_available(tmp_path):
    root = _match_dirs(tmp_path, "aaaa1111", "bbbb2222")
    with pytest.raises(FileNotFoundError, match="aaaa1111|bbbb2222"):
        resolve_artifacts_dir("zzzz", root)


def test_an_empty_or_missing_root_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Nessuna cartella"):
        resolve_artifacts_dir("latest", tmp_path / "assente")
    (tmp_path / "vuota").mkdir()
    with pytest.raises(FileNotFoundError, match="Nessun artefatto"):
        resolve_artifacts_dir("latest", tmp_path / "vuota")
