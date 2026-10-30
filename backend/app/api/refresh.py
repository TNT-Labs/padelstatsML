"""Aggiorna statistiche: recompute an analysed match from its stored run.

    POST /api/matches/{id}/refresh

A fix to identity, rallies or metrics reaches an analysed match without
running the detector again (Rianalizza, about as long as the video): the
stages after it run on the detections stored in the artifacts, through the
match's current calibration — seconds. The same as scripts/refresh.py.

The players' pictures are cut again from the video afterwards, in the
background: reading the video up to the last picture takes minutes on the
Pi for a long match, far beyond what a request may last behind the tunnel.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_editor
from app.api.verification import _CACHE, _LOCK, _completed
from app.core.config import get_settings
from app.core.database import get_db
from app.core.storage import artifacts_dir, save_crop, video_path
from app.ml.artifacts import load_artifacts
from app.ml.identity import PlayerTrack
from app.ml.reprocess import pipeline_config, refresh, same_slots
from app.models import Match, MatchStats, MatchStatus, User

logger = logging.getLogger("padel.refresh")

router = APIRouter(prefix="/api/matches", tags=["refresh"])

# One picture job at a time, in the order asked: the last refresh's
# pictures are the ones left.
_PICTURES = asyncio.Lock()


@router.post("/{match_id}/refresh")
async def refresh_stats(
    match_id: str, background: BackgroundTasks,
    db: AsyncSession = Depends(get_db), user: User = Depends(require_editor),
) -> dict:
    match, stats = await _completed(db, match_id, user)
    directory = artifacts_dir(match_id)
    if not (directory / "tracks.jsonl").exists():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Le rilevazioni di questa analisi non sono state conservate: usa Rianalizza.",
        )
    calibration_data = match.calibration
    previous_quality = dict(stats.data_quality or {})
    settings = get_settings()

    def run():
        artifacts = load_artifacts(directory)
        if not artifacts.complete:
            raise ValueError("l'analisi conservata non è completa, usa Rianalizza")
        result, identity, _, _ = refresh(
            directory, calibration_data, pipeline_config(settings, artifacts.meta), previous_quality)
        return result, identity

    async with _LOCK:
        try:
            result, identity = await run_in_threadpool(run)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, f"Aggiornamento non riuscito: {exc}") from exc
        _CACHE.clear()

    names_cleared = bool(match.player_names) and not same_slots(stats.per_player, result["per_player"])
    if names_cleared:
        match.player_names = None
        match.player_hands = None
    if not same_slots(stats.per_player, result["per_player"]):
        # Other people under the same numbers: no picture until the new
        # ones are cut, rather than a face that names the wrong player.
        stats.player_crops = None
    stats.per_player = result["per_player"]
    stats.heatmaps = result["heatmaps"]
    stats.rallies = result["rallies"]
    stats.summary = result["summary"]
    stats.data_quality = result["data_quality"]
    await db.flush()
    background.add_task(_pictures, match_id, identity.players, result["summary"])
    return {"summary": result["summary"], "names_cleared": names_cleared}


async def _pictures(match_id: str, players: list[PlayerTrack], summary: dict) -> None:
    async with _PICTURES:
        try:
            await run_in_threadpool(_cut_pictures, match_id, players, summary)
        except Exception:                    # noqa: BLE001 - the statistics are saved already
            logger.exception("Foto dei giocatori non aggiornate per la partita %s", match_id)


def _cut_pictures(match_id: str, players: list[PlayerTrack], summary: dict) -> None:
    """The players' pictures for the statistics just saved — unless they
    were replaced meanwhile (another refresh, a new analysis)."""
    from app.core.database import sync_session
    from app.ml.thumbnails import choose_pictures, pictures_from_video

    source = video_path(match_id)
    picks = choose_pictures(players)
    if not picks or not source.exists():
        return
    pictures = pictures_from_video(source, picks)

    def current(session) -> MatchStats | None:
        match = session.get(Match, match_id)
        stats = session.get(MatchStats, match_id)
        if match is None or match.status != MatchStatus.COMPLETED or stats is None:
            return None
        return stats if stats.summary == summary else None

    with sync_session() as session:
        if current(session) is None:
            return
    crops = {str(pid): save_crop(match_id, pid, image) for pid, image in pictures.items()}
    with sync_session() as session:
        stats = current(session)
        if stats is not None:
            stats.player_crops = crops or None
