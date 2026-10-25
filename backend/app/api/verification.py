"""Accuracy check of an analysed match against points marked by hand.

    GET  /api/matches/{id}/verification   landmarks, frames to mark, last result
    POST /api/matches/{id}/verification   the marks: returns and stores the errors

See app/ml/verification.py for what is measured.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_editor
from app.api.matches import _get_match
from app.core.config import get_settings
from app.core.database import get_db
from app.core.storage import artifacts_dir
from app.ml.artifacts import load_artifacts
from app.ml.court import CourtCalibration, calibration_from_dict
from app.ml.identity import PlayerTrack, resolve_players
from app.ml.player_boxes import matches_stats
from app.ml.rallies import Rally
from app.ml.smoothing import SmoothedPath, smooth_path
from app.ml.verification import (
    calibration_errors,
    choose_frames,
    landmarks_payload,
    player_errors,
)
from app.models import MatchStats, MatchStatus, User

router = APIRouter(prefix="/api/matches", tags=["verification"])

Point = tuple[float, float]

# Rebuilding the players takes seconds on the Pi; both requests of one
# check need them. One match at a time, the last one kept.
_LOCK = asyncio.Lock()
_CACHE: dict[tuple, tuple[list[PlayerTrack], dict[int, SmoothedPath], CourtCalibration]] = {}


class FrameMarks(BaseModel):
    frame_index: int = Field(ge=0)
    feet: list[Point] = Field(default_factory=list, max_length=8)


class VerificationSubmit(BaseModel):
    landmarks: dict[str, Point] = Field(default_factory=dict)
    frames: list[FrameMarks] = Field(default_factory=list, max_length=30)


async def _completed(db: AsyncSession, match_id: str, user: User):
    match = await _get_match(db, match_id, user)
    if match.status != MatchStatus.COMPLETED or not match.calibration:
        raise HTTPException(status.HTTP_409_CONFLICT, "La verifica richiede un'analisi completata.")
    stats = await db.scalar(select(MatchStats).where(MatchStats.match_id == match_id))
    if stats is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Statistiche non ancora disponibili.")
    return match, stats


def _players(match_id: str, per_player: dict, calibration_data: dict):
    """The players of the statistics on screen, with their filtered paths."""
    directory = artifacts_dir(match_id)
    tracks = directory / "tracks.jsonl"
    if not tracks.exists():
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Rilevazioni non salvate per questa partita (KEEP_ARTIFACTS): verifica non possibile.",
        )
    key = (match_id, tracks.stat().st_mtime_ns,
           tuple(sorted((str(k), (v or {}).get("samples")) for k, v in per_player.items())))
    if key in _CACHE:
        return _CACHE[key]
    artifacts = load_artifacts(directory)
    config = artifacts.meta.get("config") or {}
    speed = float(config.get("max_player_speed_ms") or get_settings().max_player_speed_ms)
    identity = resolve_players(artifacts.tracklets, max_speed_ms=speed)
    if not matches_stats(identity.players, per_player):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Le statistiche sono di una versione precedente del programma: aggiornale "
            "(scripts/refresh.py o Rianalizza) e ripeti la verifica.",
        )
    calibration = calibration_from_dict(calibration_data)
    paths = {p.player_id: smooth_path(p.observations, calibration) for p in identity.players}
    _CACHE.clear()
    _CACHE[key] = (identity.players, paths, calibration)
    return _CACHE[key]


@router.get("/{match_id}/verification")
async def verification_plan(
    match_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_editor),
) -> dict:
    match, stats = await _completed(db, match_id, user)
    async with _LOCK:
        players, _, _ = await run_in_threadpool(_players, match_id, stats.per_player, match.calibration)
    rallies = [Rally(r["index"], r["start_s"], r["end_s"], r.get("players_tracked", 0))
               for r in (stats.rallies or [])]
    return {
        "landmarks": landmarks_payload(),
        "frames": choose_frames(players, rallies),
        "frame_size": [match.width, match.height],
        "last": (stats.data_quality or {}).get("verification"),
    }


@router.post("/{match_id}/verification")
async def submit_verification(
    match_id: str, payload: VerificationSubmit,
    db: AsyncSession = Depends(get_db), user: User = Depends(require_editor),
) -> dict:
    match, stats = await _completed(db, match_id, user)
    if not payload.landmarks and not any(f.feet for f in payload.frames):
        raise HTTPException(422, "Nessun punto segnato.")
    async with _LOCK:
        players, paths, calibration = await run_in_threadpool(
            _players, match_id, stats.per_player, match.calibration)
    result = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calibration": calibration_errors(calibration, dict(payload.landmarks)),
        "players": player_errors(
            calibration, players, paths, [(f.frame_index, list(f.feet)) for f in payload.frames]
        ) if payload.frames else None,
    }
    # A new dict, not an edit in place: the JSON column must see the change.
    stats.data_quality = {**(stats.data_quality or {}), "verification": result}
    await db.flush()
    return result
