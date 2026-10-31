"""Accuracy check of an analysed match against points marked by hand.

    GET  /api/matches/{id}/verification          landmarks, frames to mark, last result
    POST /api/matches/{id}/verification          the marks: returns and stores the errors
    POST /api/matches/{id}/verification/refine   a calibration fitted on the landmarks too
    POST /api/matches/{id}/verification/apply    use it: statistics recomputed in seconds

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
from app.ml.court import (
    DERIVED_MAX_OFF_FRAME,
    CalibrationError,
    CourtCalibration,
    build_calibration,
    calibration_from_dict,
)
from app.ml.identity import PlayerTrack, resolve_players
from app.ml.player_boxes import matches_stats, save_player_boxes
from app.ml.reprocess import pipeline_config, recompute, same_slots, with_calibration
from app.ml.rallies import Rally
from app.ml.smoothing import SmoothedPath, smooth_path
from app.ml.verification import (
    calibration_errors,
    choose_frames,
    landmarks_payload,
    player_errors,
    refine_calibration,
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
    key = (match_id, tracks.stat().st_mtime_ns, str(calibration_data.get("corners_px")),
           tuple(sorted((str(k), (v or {}).get("samples")) for k, v in per_player.items())))
    if key in _CACHE:
        return _CACHE[key]
    artifacts = with_calibration(load_artifacts(directory), calibration_data)
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
        ) if any(f.feet for f in payload.frames) else None,
    }
    # A new dict, not an edit in place: the JSON column must see the change.
    stats.data_quality = {**(stats.data_quality or {}), "verification": result}
    await db.flush()
    return result


class RefineSubmit(BaseModel):
    landmarks: dict[str, Point]


class ApplySubmit(RefineSubmit):
    corners_px: list[Point] = Field(min_length=4, max_length=4)
    k1: float = Field(default=0.0, ge=0, le=0.5)


@router.post("/{match_id}/verification/refine")
async def propose_refinement(
    match_id: str, payload: RefineSubmit,
    db: AsyncSession = Depends(get_db), user: User = Depends(require_editor),
) -> dict:
    """A calibration fitted on the corners and the landmarks; nothing saved."""
    match, _ = await _completed(db, match_id, user)
    try:
        return refine_calibration(calibration_from_dict(match.calibration), dict(payload.landmarks))
    except CalibrationError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/{match_id}/verification/apply")
async def apply_refinement(
    match_id: str, payload: ApplySubmit,
    db: AsyncSession = Depends(get_db), user: User = Depends(require_editor),
) -> dict:
    """Make the corrected calibration the match's, and recompute its
    statistics from the stored detections — projected again through it."""
    match, stats = await _completed(db, match_id, user)
    try:
        # Corners of a proposal (refine): computed, possibly far out.
        calibration = build_calibration(
            [tuple(p) for p in payload.corners_px], (match.width, match.height), k1=payload.k1,
            max_off_frame=DERIVED_MAX_OFF_FRAME)
    except CalibrationError as exc:
        raise HTTPException(422, str(exc)) from exc
    calibration_data = {
        **(match.calibration or {}),
        "corners_px": calibration.corners_px.tolist(),
        "frame_size": [match.width, match.height],
        # The net check and the preset were for the corners placed by hand.
        "net_px": None,
        "preset_id": None,
        "refined_from_landmarks": len(payload.landmarks),
        "k1": payload.k1,
    }
    directory = artifacts_dir(match_id)
    settings = get_settings()

    def run():
        artifacts = load_artifacts(directory)
        result, identity, _, used = recompute(directory, calibration_data, pipeline_config(settings, artifacts.meta))
        video = used.meta.get("video") or {}
        save_player_boxes(directory, identity, used.sample_hz,
                          {"width": video.get("width"), "height": video.get("height")}, result["per_player"])
        return result

    async with _LOCK:
        try:
            result = await run_in_threadpool(run)
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, f"Ricalcolo non riuscito: {exc}") from exc
        _CACHE.clear()

    result["data_quality"]["verification"] = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Measured on the landmarks the calibration was just fitted on: an
        # optimistic figure, said so on screen; the expected error is the
        # one measured leaving each out (refine).
        "calibration": calibration_errors(calibration, dict(payload.landmarks)),
        "players": None,
        "refined": True,
    }
    names_cleared = bool(match.player_names) and not same_slots(stats.per_player, result["per_player"])
    if names_cleared:
        match.player_names = None
        match.player_hands = None
    if not same_slots(stats.per_player, result["per_player"]):
        match.stroke_labels = None
        match.hit_labels = None
    match.calibration = calibration_data
    stats.per_player = result["per_player"]
    stats.heatmaps = result["heatmaps"]
    stats.rallies = result["rallies"]
    stats.summary = result["summary"]
    stats.data_quality = result["data_quality"]
    await db.flush()
    return {"summary": result["summary"], "verification": result["data_quality"]["verification"],
            "names_cleared": names_cleared}
