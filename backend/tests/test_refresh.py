"""Aggiorna statistiche: an analysed match recomputed from its stored run."""
from __future__ import annotations

import math

import pytest_asyncio

from tests.synthetic import SyntheticPlayer

FPS, STEP = 30.0, 6          # 5 Hz, as in production


def _store_run(calibration, directory, seconds: float = 40.0) -> None:
    """Four players moving about their places, tracked and stored as the
    pipeline does."""
    from tests.test_replay import _live_run

    places = [(2.5, 3.0), (7.5, 5.0), (2.5, 15.0), (7.5, 17.0)]
    frames = []
    for i in range(int(seconds * FPS / STEP)):
        t = i * STEP / FPS
        # Play for 12 s, 8 s of pause, again.
        amplitude = 1.5 if (t % 20.0) < 12.0 else 0.0
        frames.append([
            SyntheticPlayer((x + amplitude * math.sin(1.3 * t + k), y + amplitude * math.cos(1.1 * t + k)), k)
            for k, (x, y) in enumerate(places)
        ])
    _live_run(calibration, directory, frames)


@pytest_asyncio.fixture
async def analysed_match(calibration):
    """A completed match of the signed-in user, its run stored, its
    statistics computed and its players named."""
    from conftest import signed_in

    from app.core.config import get_settings
    from app.core.database import sync_session
    from app.core.storage import artifacts_dir
    from app.ml.reprocess import pipeline_config, recompute
    from app.models import Match, MatchStats, MatchStatus

    owner = await signed_in("refresher")
    match_id = (await owner.post("/api/matches", json={"title": "Set"})).json()["match_id"]
    directory = artifacts_dir(match_id)
    _store_run(calibration, directory)
    from app.ml.artifacts import load_artifacts

    result, _, _, _ = recompute(directory, None, pipeline_config(get_settings(), load_artifacts(directory).meta))
    for rally in result["rallies"]:
        rally.pop("serve")                  # computed before the serve check
    result["data_quality"]["verification"] = {"at": "2026-10-01T10:00:00+00:00", "refined": False}
    with sync_session() as session:
        match = session.get(Match, match_id)
        match.status = MatchStatus.COMPLETED
        match.calibration = calibration.to_dict()
        match.width, match.height = calibration.frame_size
        match.player_names = ["A", "B", "C", "D"]
        session.add(MatchStats(
            match_id=match_id, per_player=result["per_player"], heatmaps=result["heatmaps"],
            rallies=result["rallies"], summary=result["summary"], data_quality=result["data_quality"],
        ))
    try:
        yield owner, match_id
    finally:
        await owner.aclose()


async def test_statistics_are_recomputed_from_the_stored_run(analysed_match):
    owner, match_id = analysed_match
    response = await owner.post(f"/api/matches/{match_id}/refresh")
    assert response.status_code == 200, response.text
    assert response.json()["names_cleared"] is False

    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    assert stats["rallies"] and all(r["serve"] is not None for r in stats["rallies"])
    # Same people under the same numbers: names kept; the last accuracy
    # check kept with them, the calibration it measured unchanged.
    assert stats["player_names"] == ["A", "B", "C", "D"]
    assert stats["data_quality"]["verification"]["at"] == "2026-10-01T10:00:00+00:00"
    # How the tracker got its players reaches the reliability panel (the
    # response schema used to drop it).
    assert set(stats["data_quality"]["tracking"]) >= {"weak_detections", "feet_below_net"}


async def test_read_only_accounts_cannot_refresh(analysed_match):
    from conftest import signed_in

    _, match_id = analysed_match
    viewer = await signed_in("refresh-viewer", "visualizzatore")
    async with viewer:
        assert (await viewer.post(f"/api/matches/{match_id}/refresh")).status_code == 403


async def test_a_match_without_its_stored_run_must_be_analysed_again(analysed_match):
    import shutil

    from app.core.storage import artifacts_dir

    owner, match_id = analysed_match
    shutil.rmtree(artifacts_dir(match_id))
    response = await owner.post(f"/api/matches/{match_id}/refresh")
    assert response.status_code == 409
    assert "Rianalizza" in response.json()["detail"]


async def test_only_an_analysed_match_can_be_refreshed():
    from conftest import signed_in

    owner = await signed_in("refresh-new")
    async with owner:
        match_id = (await owner.post("/api/matches", json={"title": "Nuova"})).json()["match_id"]
        assert (await owner.post(f"/api/matches/{match_id}/refresh")).status_code == 409


async def test_the_hands_are_kept_with_the_names(analysed_match):
    owner, match_id = analysed_match
    response = await owner.patch(f"/api/matches/{match_id}", json={"player_hands": ["R", "R", "L", None]})
    assert response.status_code == 200, response.text
    assert response.json()["player_hands"] == ["R", "R", "L", None]
    assert (await owner.patch(f"/api/matches/{match_id}", json={"player_hands": ["X"]})).status_code == 422

    assert (await owner.post(f"/api/matches/{match_id}/refresh")).status_code == 200
    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    assert stats["player_hands"] == ["R", "R", "L", None]


async def test_strokes_are_labelled_as_seen_in_the_video(analysed_match):
    from conftest import signed_in

    owner, match_id = analysed_match
    url = f"/api/matches/{match_id}/stroke-labels"
    response = await owner.patch(url, json={"labels": {"1:83.4": "smash", "2:129.0": "nulla"}})
    assert response.status_code == 200, response.text
    response = await owner.patch(url, json={"labels": {"2:129.0": None, "0:5": "lob"}})
    assert response.json()["stroke_labels"] == {"1:83.4": "smash", "0:5": "lob"}
    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    assert stats["stroke_labels"] == {"1:83.4": "smash", "0:5": "lob"}

    assert (await owner.patch(url, json={"labels": {"1:83.4": "servizio"}})).status_code == 422
    assert (await owner.patch(url, json={"labels": {"chi": "lob"}})).status_code == 422
    viewer = await signed_in("labels-viewer", "visualizzatore")
    async with viewer:
        assert (await viewer.patch(url, json={"labels": {"1:83.4": "lob"}})).status_code == 403


async def test_strokes_are_classified_with_the_racket_hand_when_read(analysed_match):
    from app.core.database import sync_session
    from app.models import Match, MatchStats

    owner, match_id = analysed_match
    candidate = {"t": 50.0, "net_m": 4.0, "speed": 1.0, "up": "LR", "high": {"L": 0.6, "R": -0.8}}
    with sync_session() as session:
        stats = session.get(MatchStats, match_id)
        per_player = {k: dict(v) for k, v in stats.per_player.items()}
        per_player["0"]["strokes"] = [candidate]
        per_player["1"]["strokes"] = [candidate]
        stats.per_player = per_player
        session.get(Match, match_id).player_hands = ["R", "L", None, None]

    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    # Right-handed: the right arm up with the left: a smash. Left-handed:
    # the left up but the right hanging: not one.
    assert stats["per_player"]["0"]["strokes"] == [{"t": 50.0, "kind": "overhead"}]
    assert stats["per_player"]["1"]["strokes"] == []


async def test_the_ball_is_marked_frame_by_frame(analysed_match):
    from conftest import signed_in

    owner, match_id = analysed_match
    clips = await owner.get(f"/api/matches/{match_id}/ball/clips")
    assert clips.status_code == 200, clips.text
    body = clips.json()
    assert body["clips"] and body["labelled"] == 0
    frame = body["clips"][0]["start_frame"]

    url = f"/api/matches/{match_id}/ball/labels"
    response = await owner.patch(url, json={"labels": {str(frame): [100.5, 50.0], str(frame + 1): "none"}})
    assert response.json() == {"labelled": 2, "visible": 1}
    assert (await owner.get(url)).json()["labels"] == {str(frame): [100.5, 50.0], str(frame + 1): "none"}
    assert (await owner.patch(url, json={"labels": {"abc": [1, 1]}})).status_code == 422
    assert (await owner.patch(url, json={"labels": {str(frame): [99999, 1]}})).status_code == 422
    assert (await owner.get(f"/api/matches/{match_id}/ball/clips")).json()["clips"][0]["labelled"] == 2

    viewer = await signed_in("ball-viewer", "visualizzatore")
    async with viewer:
        assert (await viewer.get(f"/api/matches/{match_id}/ball/clips")).status_code == 403


async def test_the_marked_frames_are_exported_for_training(analysed_match):
    import io
    import zipfile

    owner, match_id = analysed_match
    assert (await owner.get(f"/api/matches/{match_id}/ball/export.zip")).status_code == 409   # nothing marked
    clips = (await owner.get(f"/api/matches/{match_id}/ball/clips")).json()["clips"]
    await owner.patch(f"/api/matches/{match_id}/ball/labels",
                      json={"labels": {str(clips[0]["start_frame"]): [10, 10]}})
    response = await owner.get(f"/api/matches/{match_id}/ball/export.zip")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert {"labels.csv", "meta.json"} <= set(archive.namelist())
    # The far half at full resolution, where the framing allows it.
    crop = (await owner.get(f"/api/matches/{match_id}/ball/clips")).json()["far_crop"]
    far = await owner.get(f"/api/matches/{match_id}/ball/export.zip?far=1")
    if crop is None:
        assert far.status_code == 409
    else:
        assert far.status_code == 200 and "lontana_" in far.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(far.content)) as archive:
            import json
            assert json.loads(archive.read("meta.json"))["offset"] == crop[:2]


async def test_the_ball_found_by_the_analysis_is_served_as_flights(analysed_match):
    from conftest import signed_in

    from app.core.storage import artifacts_dir

    owner, match_id = analysed_match
    url = f"/api/matches/{match_id}/ball/track"
    assert (await owner.get(url)).status_code == 404          # no ball in this run
    lines = [f"[{f},{f / 30:.3f},{100 + 10 * f},{500 - 5 * f},0.8]" for f in range(2, 40)]
    lines += ["[50,1.667,1700.0,200.0,0.6]"]                    # a stray answer
    (artifacts_dir(match_id) / "ball.jsonl").write_text("\n".join(lines) + "\n")

    body = (await owner.get(url)).json()
    assert body["raw"] == 39 and body["seen"] == 38 and len(body["flights"]) == 1
    assert body["flights"][0]["x"][0] == 120
    viewer = await signed_in("ball-track-viewer", "visualizzatore")
    async with viewer:
        assert (await viewer.get(url)).status_code == 200      # viewers see it too


async def test_the_time_the_analysis_took_is_shown_with_the_statistics(analysed_match):
    from datetime import datetime, timedelta

    from app.core.database import sync_session
    from app.models import Job, JobState

    owner, match_id = analysed_match
    assert (await owner.get(f"/api/matches/{match_id}/stats")).json()["analysis_time"] is None
    start = datetime(2026, 10, 31, 18, 0, 0)
    with sync_session() as session:
        # An older run, then the last one: the last one is shown.
        session.add(Job(match_id=match_id, state=JobState.DONE, started_at=start - timedelta(days=1),
                        finished_at=start - timedelta(days=1) + timedelta(minutes=30)))
        session.add(Job(match_id=match_id, state=JobState.DONE, started_at=start,
                        finished_at=start + timedelta(minutes=52, seconds=10)))
        session.add(Job(match_id=match_id, state=JobState.FAILED, started_at=start + timedelta(hours=2),
                        finished_at=start + timedelta(hours=2, minutes=1)))
    timing = (await owner.get(f"/api/matches/{match_id}/stats")).json()["analysis_time"]
    assert timing["seconds"] == 52 * 60 + 10
    assert timing["finished_at"].startswith("2026-10-31T18:52:10") and timing["finished_at"].endswith(("Z", "+00:00"))


async def test_a_refresh_reads_the_ball_and_says_when_it_shows_no_hits(analysed_match):
    from app.core.storage import artifacts_dir

    owner, match_id = analysed_match
    stats = (await owner.post(f"/api/matches/{match_id}/refresh")) and \
        (await owner.get(f"/api/matches/{match_id}/stats")).json()
    assert all(r.get("hits") is None for r in stats["rallies"])          # no ball, no hits
    (artifacts_dir(match_id) / "ball.jsonl").write_text(
        "\n".join(f"[{f},{f / 30:.3f},{400 + 3 * f},{300 + 2 * f},0.8]" for f in range(30, 90)) + "\n")
    assert (await owner.post(f"/api/matches/{match_id}/refresh")).status_code == 200
    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    # A ball going straight for two seconds: no hits in the rallies, so the
    # ball is not used for them, and the panel says so.
    assert all(r.get("hits") is None for r in stats["rallies"])
    assert any("non sono stati ricavati dalla palla" in w for w in stats["data_quality"]["warnings"])


async def test_the_hits_seen_in_the_video_are_marked(analysed_match):
    from conftest import signed_in

    owner, match_id = analysed_match
    url = f"/api/matches/{match_id}/hit-labels"
    response = await owner.put(url, json={"hits": [[12.345, 2], [10.0, 0]]})
    assert response.status_code == 200, response.text
    assert response.json()["hit_labels"] == [[10.0, 0], [12.35, 2]]
    assert (await owner.get(f"/api/matches/{match_id}/stats")).json()["hit_labels"] == [[10.0, 0], [12.35, 2]]
    assert (await owner.put(url, json={"hits": [[1.0, 4]]})).status_code == 422
    assert (await owner.put(url, json={"hits": []})).json()["hit_labels"] is None
    viewer = await signed_in("hits-viewer", "visualizzatore")
    async with viewer:
        assert (await viewer.put(url, json={"hits": [[1.0, 1]]})).status_code == 403


async def test_the_kind_of_each_marked_hit_is_kept(analysed_match):
    owner, match_id = analysed_match
    url = f"/api/matches/{match_id}/hit-labels"
    response = await owner.put(url, json={"hits": [[12.0, 2, "volee_rovescio"], [10.0, 0], [11.0, 1, "battuta"]]})
    assert response.status_code == 200, response.text
    # Hits with and without a kind together, in time order.
    assert response.json()["hit_labels"] == [[10.0, 0], [11.0, 1, "battuta"], [12.0, 2, "volee_rovescio"]]
    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    assert stats["hit_labels"] == [[10.0, 0], [11.0, 1, "battuta"], [12.0, 2, "volee_rovescio"]]
    assert (await owner.put(url, json={"hits": [[10.0, 0, "veronica"]]})).status_code == 422
    # The kind follows its hit when the names move it to another number.
    from app.core.database import sync_session
    from app.models import Match

    with sync_session() as session:
        session.get(Match, match_id).player_names = None
    await owner.patch(f"/api/matches/{match_id}", json={"player_names": ["c", "A ", "D", "B"]})
    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    assert stats["hit_labels"] == [[10.0, 1], [11.0, 3, "battuta"], [12.0, 0, "volee_rovescio"]]


async def test_the_marked_hits_follow_the_names_through_a_new_numbering(analysed_match):
    from app.core.database import sync_session
    from app.models import Match

    owner, match_id = analysed_match                     # named A, B, C, D
    url = f"/api/matches/{match_id}/hit-labels"
    await owner.put(url, json={"hits": [[10.0, 0], [11.0, 2], [12.0, 3]]})
    # A new analysis: the names go, the marks stay.
    with sync_session() as session:
        match = session.get(Match, match_id)
        match.player_names = None
        assert match.hit_labels == [[10.0, 0], [11.0, 2], [12.0, 3]]
    # The same people named again under other numbers: the marks move with them.
    response = await owner.patch(f"/api/matches/{match_id}", json={"player_names": ["c", "A ", "D", "B"]})
    assert response.status_code == 200, response.text
    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    assert stats["hit_labels"] == [[10.0, 1], [11.0, 0], [12.0, 2]]
    # Other people: nothing is guessed.
    await owner.patch(f"/api/matches/{match_id}", json={"player_names": ["X", "Y", "Z", "W"]})
    assert (await owner.get(f"/api/matches/{match_id}/stats")).json()["hit_labels"] == [[10.0, 1], [11.0, 0], [12.0, 2]]


async def test_the_points_seen_in_the_video_are_marked(analysed_match):
    from conftest import signed_in

    owner, match_id = analysed_match
    url = f"/api/matches/{match_id}/point-labels"
    response = await owner.put(url, json={"points": [[30.0, 41.5, [3, 1]], [10.004, 20.0, None]]})
    assert response.status_code == 200, response.text
    # In time order, rounded to the hundredth, the winners sorted.
    assert response.json()["point_labels"] == [[10.0, 20.0, None], [30.0, 41.5, [1, 3]]]
    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    assert stats["point_labels"] == [[10.0, 20.0, None], [30.0, 41.5, [1, 3]]]
    # Ending before it starts, overlapping, one player twice, a fifth player.
    for bad in ([[20.0, 10.0, None]], [[10.0, 20.0, None], [19.0, 25.0, None]],
                [[10.0, 20.0, [1, 1]]], [[10.0, 20.0, [0, 4]]], [[10.0, 20.0, [0]]]):
        assert (await owner.put(url, json={"points": bad})).status_code == 422, bad
    assert (await owner.get(f"/api/matches/{match_id}/stats")).json()["point_labels"] == stats["point_labels"]
    assert (await owner.put(url, json={"points": []})).json()["point_labels"] is None
    viewer = await signed_in("points-viewer", "visualizzatore")
    async with viewer:
        assert (await viewer.put(url, json={"points": [[1.0, 2.0, None]]})).status_code == 403


async def test_the_marked_points_follow_the_names_through_a_new_numbering(analysed_match):
    from app.core.database import sync_session
    from app.models import Match

    owner, match_id = analysed_match                     # named A, B, C, D
    await owner.put(f"/api/matches/{match_id}/point-labels",
                    json={"points": [[10.0, 20.0, [0, 2]], [30.0, 40.0, None]]})
    await owner.put(f"/api/matches/{match_id}/hit-labels", json={"hits": [[11.0, 2]]})
    with sync_session() as session:
        session.get(Match, match_id).player_names = None
    response = await owner.patch(f"/api/matches/{match_id}", json={"player_names": ["c", "A ", "D", "B"]})
    assert response.status_code == 200, response.text
    stats = (await owner.get(f"/api/matches/{match_id}/stats")).json()
    assert stats["point_labels"] == [[10.0, 20.0, [0, 1]], [30.0, 40.0, None]]
    assert stats["hit_labels"] == [[11.0, 0]]
    await owner.patch(f"/api/matches/{match_id}", json={"player_names": ["X", "Y", "Z", "W"]})
    assert (await owner.get(f"/api/matches/{match_id}/stats")).json()["point_labels"] == stats["point_labels"]
