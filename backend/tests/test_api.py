"""End-to-end API flow: create -> upload -> calibrate -> queue."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.asyncio

W, H = 640, 360
GOOD_CORNERS = [[200, 100], [440, 100], [600, 330], [40, 330]]


def _make_video(path: Path) -> Path:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    if not writer.isOpened():
        pytest.skip("Questa build di OpenCV non sa scrivere file MP4")
    for i in range(60):
        writer.write(np.full((H, W, 3), (i * 3) % 255, dtype=np.uint8))
    writer.release()
    return path


@pytest_asyncio.fixture
async def client():
    from app.core.database import init_schema
    from app.main import app

    init_schema()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest_asyncio.fixture
async def uploaded_match(client, tmp_path):
    response = await client.post("/api/matches", json={"title": "Test"})
    assert response.status_code == 201
    match_id = response.json()["match_id"]

    video = _make_video(tmp_path / "clip.mp4")
    response = await client.put(
        f"/api/matches/{match_id}/video", content=video.read_bytes()
    )
    assert response.status_code == 200, response.text
    return match_id, response.json()


async def test_upload_records_video_properties(uploaded_match):
    _, match = uploaded_match
    assert match["status"] == "needs_calibration"
    assert match["width"] == W and match["height"] == H
    assert match["fps"] == pytest.approx(30.0, abs=0.5)
    assert match["duration_seconds"] == pytest.approx(2.0, abs=0.3)
    assert match["calibrated"] is False


async def test_unreadable_upload_is_rejected(client):
    match_id = (await client.post("/api/matches", json={"title": "Rotto"})).json()["match_id"]
    response = await client.put(f"/api/matches/{match_id}/video", content=b"not a video")
    assert response.status_code == 422
    # And the match must not be left claiming it has a video.
    assert (await client.get(f"/api/matches/{match_id}")).json()["status"] == "uploading"


async def test_analysis_cannot_start_without_calibration(uploaded_match, client):
    """The central guarantee of the rewrite: no homography, no metrics."""
    match_id, _ = uploaded_match
    response = await client.post(f"/api/matches/{match_id}/start")
    assert response.status_code == 409
    assert "alibra" in response.json()["detail"]


async def test_suggestion_is_returned_for_the_calibration_ui(uploaded_match, client):
    match_id, _ = uploaded_match
    response = await client.get(f"/api/matches/{match_id}/calibration/suggestion")
    assert response.status_code == 200
    body = response.json()
    assert len(body["corners_px"]) == 4
    assert len(body["corner_labels"]) == 4
    assert body["frame_size"] == [W, H]


async def test_invalid_corners_are_refused_with_a_reason(uploaded_match, client):
    match_id, _ = uploaded_match
    response = await client.post(
        f"/api/matches/{match_id}/calibration",
        # 60x50 px on a 640x360 frame: well under the 2% minimum court area.
        json={"corners_px": [[100, 100], [160, 100], [160, 150], [100, 150]]},
    )
    assert response.status_code == 422
    assert "piccolo" in response.json()["detail"]


async def test_mirrored_corner_order_is_refused(uploaded_match, client):
    match_id, _ = uploaded_match
    mirrored = [GOOD_CORNERS[0], GOOD_CORNERS[3], GOOD_CORNERS[2], GOOD_CORNERS[1]]
    response = await client.post(
        f"/api/matches/{match_id}/calibration", json={"corners_px": mirrored}
    )
    assert response.status_code == 422
    assert "specchiato" in response.json()["detail"]


async def test_full_flow_to_queued(uploaded_match, client):
    match_id, _ = uploaded_match

    response = await client.post(
        f"/api/matches/{match_id}/calibration", json={"corners_px": GOOD_CORNERS}
    )
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True

    match = (await client.get(f"/api/matches/{match_id}")).json()
    assert match["status"] == "ready"
    assert match["calibrated"] is True

    response = await client.post(f"/api/matches/{match_id}/start")
    assert response.status_code == 200
    assert response.json()["status"] == "queued"

    # A job row is what the worker picks up; there must be exactly one.
    from sqlalchemy import select

    from app.core.database import sync_session
    from app.models import Job

    with sync_session() as session:
        jobs = session.scalars(select(Job).where(Job.match_id == match_id)).all()
    assert len(jobs) == 1


async def test_starting_twice_does_not_queue_two_jobs(uploaded_match, client):
    match_id, _ = uploaded_match
    await client.post(f"/api/matches/{match_id}/calibration", json={"corners_px": GOOD_CORNERS})
    await client.post(f"/api/matches/{match_id}/start")
    await client.post(f"/api/matches/{match_id}/start")

    from sqlalchemy import select

    from app.core.database import sync_session
    from app.models import Job

    with sync_session() as session:
        jobs = session.scalars(select(Job).where(Job.match_id == match_id)).all()
    assert len(jobs) == 1


async def test_net_marker_reports_its_own_error(uploaded_match, client):
    match_id, _ = uploaded_match
    # Net drawn halfway up the quad: close to, but not exactly, Y = 10 m.
    net = [[120, 215], [520, 215]]
    response = await client.post(
        f"/api/matches/{match_id}/calibration",
        json={"corners_px": GOOD_CORNERS, "net_px": net},
    )
    assert response.status_code == 200
    assert response.json()["net_error_m"] is not None


async def test_player_names_are_persisted(uploaded_match, client):
    """They used to live only in browser state and were lost on reload."""
    match_id, _ = uploaded_match
    response = await client.patch(
        f"/api/matches/{match_id}", json={"player_names": ["Ana", "Bea", "Carlo", "Dino"]}
    )
    assert response.status_code == 200
    assert response.json()["player_names"] == ["Ana", "Bea", "Carlo", "Dino"]
    assert (await client.get(f"/api/matches/{match_id}")).json()["player_names"][0] == "Ana"


async def test_stats_are_absent_until_the_analysis_runs(uploaded_match, client):
    match_id, _ = uploaded_match
    assert (await client.get(f"/api/matches/{match_id}/stats")).status_code == 404


async def test_delete_removes_the_match_and_its_files(uploaded_match, client):
    from app.core.storage import keyframe_path, video_path

    match_id, _ = uploaded_match
    assert video_path(match_id).exists()

    assert (await client.delete(f"/api/matches/{match_id}")).status_code == 204
    assert (await client.get(f"/api/matches/{match_id}")).status_code == 404
    assert not video_path(match_id).exists()
    assert not keyframe_path(match_id).exists()


async def test_camera_preset_round_trip(uploaded_match, client):
    """Calibrate once, reuse on every later match from the same camera."""
    match_id, _ = uploaded_match

    response = await client.post(
        "/api/presets",
        json={
            "name": "Campo 1 - tripode angolo",
            "corners_px": GOOD_CORNERS,
            "frame_width": W,
            "frame_height": H,
        },
    )
    assert response.status_code == 201, response.text
    preset_id = response.json()["id"]

    response = await client.post(
        f"/api/matches/{match_id}/calibration", json={"preset_id": preset_id}
    )
    assert response.status_code == 200
    assert response.json()["corners_px"] == [list(map(float, c)) for c in GOOD_CORNERS]

    presets = (await client.get("/api/presets")).json()
    assert presets[0]["times_used"] == 1


async def test_preset_from_a_different_resolution_is_rescaled(uploaded_match, client):
    match_id, _ = uploaded_match
    response = await client.post(
        "/api/presets",
        json={
            "name": "Camera 4K",
            "corners_px": [[c[0] * 3, c[1] * 3] for c in GOOD_CORNERS],
            "frame_width": W * 3,
            "frame_height": H * 3,
        },
    )
    preset_id = response.json()["id"]

    response = await client.post(
        f"/api/matches/{match_id}/calibration", json={"preset_id": preset_id}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["corners_px"][0] == pytest.approx(GOOD_CORNERS[0], abs=0.5)
    assert "riscalato" in body["message"]


async def test_invalid_preset_is_refused_at_creation(client):
    response = await client.post(
        "/api/presets",
        json={
            "name": "Rotto",
            "corners_px": [[100, 100], [160, 100], [160, 150], [100, 150]],
            "frame_width": W,
            "frame_height": H,
        },
    )
    assert response.status_code == 422


async def test_health_reports_the_missing_detector(client):
    """The Pi cannot analyse anything without the exported ONNX model, so a
    missing model is a degraded service, not a silent fallback."""
    response = await client.get("/api/health")
    assert response.status_code == 503
    assert "mancante" in response.json()["checks"]["detector"]


async def test_unknown_api_path_returns_json_404(client):
    """The SPA fallback must not swallow API mistakes: returning index.html
    for /api/typo surfaces as an HTML parse error in the browser instead of
    the actual problem."""
    response = await client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


async def test_no_content_routes_declare_no_response_model(client):
    """A 204 route must not carry a response model.

    FastAPI asserts this at import time, so getting it wrong takes the whole
    application down rather than failing one endpoint. The trap is that these
    modules use `from __future__ import annotations`: a `-> None` return
    annotation is then resolved through `get_type_hints`, which normalises it
    to `NoneType` — truthy — and FastAPI reads it as a response model.

    This check is independent of FastAPI's version-specific behaviour, so it
    catches the next 204 route added without `response_model=None`.
    """
    from fastapi.routing import APIRoute

    from app.main import app

    def routes(router):
        for route in router.routes:
            if isinstance(route, APIRoute):
                yield route
            elif hasattr(route, "routes"):
                yield from routes(route)

    no_content = [r for r in routes(app.router) if r.status_code == 204]
    assert no_content, "nessuna rotta 204 trovata: il controllo non sta verificando nulla"
    for route in no_content:
        assert route.response_model is None, (
            f"{route.path} restituisce 204 ma dichiara un response model "
            f"({route.response_model}): aggiungi response_model=None"
        )


async def test_a_completed_match_can_be_analysed_again(uploaded_match, client):
    """Re-analysis re-queues a finished match, and forgets the player names:
    the new run may number the people differently, and old names would then
    sit on the wrong players."""
    match_id, _ = uploaded_match
    await client.post(f"/api/matches/{match_id}/calibration", json={"corners_px": GOOD_CORNERS})
    await client.patch(
        f"/api/matches/{match_id}", json={"player_names": ["Ana", "Bea", "Carlo", "Dino"]}
    )

    from app.core.database import sync_session
    from app.models import Match, MatchStatus

    with sync_session() as session:
        session.get(Match, match_id).status = MatchStatus.COMPLETED

    response = await client.post(f"/api/matches/{match_id}/start")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "queued"
    assert response.json()["player_names"] is None


async def test_the_identity_cue_reaches_the_client():
    """The schema used to list data_quality fields one by one; one it did not
    list was silently dropped before reaching the reliability panel."""
    from app.schemas.match import DataQuality

    cue = {"cue": "reid", "same": 0.03, "different": 0.39, "veto": 0.21, "pairs": [900, 2900]}
    quality = DataQuality(calibration_source="manual", sample_hz=5.0, frames_sampled=10,
                          players_found=4, identity_cue=cue)
    assert quality.model_dump()["identity_cue"] == cue
    legacy = DataQuality(calibration_source="manual", sample_hz=5.0, frames_sampled=10, players_found=4)
    assert legacy.identity_cue is None


async def test_player_thumbnails_are_served_from_a_relative_url(uploaded_match, client):
    """The URL used to be absolute, built on API_BASE_URL, and every
    thumbnail broke when that setting did not match the address the browser
    used to reach the Pi. Relative, it works from any address."""
    match_id, _ = uploaded_match

    from app.core.database import sync_session
    from app.core.storage import save_crop
    from app.models import MatchStats

    player = {
        "team": 0, "samples": 10, "tracked_ratio": 0.8, "distance_m": 100.0,
        "distance_rally_m": 60.0, "avg_speed_ms": 1.5, "peak_speed_ms": 5.0,
        "coverage_m2": 30.0, "zone_pct": {"net": 0.2, "mid": 0.3, "back": 0.5},
    }
    key = save_crop(match_id, 0, b"\xff\xd8\xff\xe0 jpeg")
    with sync_session() as session:
        session.add(MatchStats(
            match_id=match_id, per_player={"0": player}, heatmaps={}, rallies=[],
            summary={"analysed_s": 60, "rallies_count": 0, "total_rally_s": 0, "avg_rally_s": 0,
                     "median_rally_s": 0, "longest_rally_s": 0, "active_ratio": 0, "players_found": 1},
            data_quality={"calibration_source": "manual", "sample_hz": 5.0, "frames_sampled": 300,
                          "players_found": 1},
            player_crops={"0": key},
        ))

    stats = (await client.get(f"/api/matches/{match_id}/stats")).json()
    url = stats["per_player"]["0"]["crop_url"]
    assert url.startswith(f"/api/matches/{match_id}/crops/0?v=")

    image = await client.get(url)
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
