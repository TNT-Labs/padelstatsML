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
