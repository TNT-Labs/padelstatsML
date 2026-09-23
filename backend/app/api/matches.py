"""Match lifecycle endpoints.

    POST   /api/matches                          create, get an upload URL
    PUT    /api/matches/{id}/video               stream the video to the SSD
    GET    /api/matches/{id}/keyframe            frame used for calibration
    GET    /api/matches/{id}/calibration/suggestion
    POST   /api/matches/{id}/calibration         validate and store the corners
    POST   /api/matches/{id}/start               queue the analysis
    GET    /api/matches/{id}                     poll status
    GET    /api/matches/{id}/stats               results
    PATCH  /api/matches/{id}                     rename / assign player names
    DELETE /api/matches/{id}

The calibration step is mandatory and is the main structural change: an
analysis cannot be queued before a human has confirmed where the court is.
Every metric this system returns is in metres, and an unvalidated homography
produces confident, wrong numbers.
"""
from __future__ import annotations

import logging

import cv2
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.storage import (
    crop_path,
    crop_url,
    delete_match_files,
    free_space_bytes,
    keyframe_path,
    upload_url,
    video_path,
)
from app.ml.court import (
    CORNER_LABELS,
    CalibrationError,
    build_calibration,
    suggest_corners,
)
from app.ml.video import VideoError, extract_keyframe, probe
from app.models import CameraPreset, Job, JobState, Match, MatchStats, MatchStatus
from app.schemas.match import (
    CalibrationResult,
    CalibrationSubmit,
    CalibrationSuggestion,
    MatchCreate,
    MatchRead,
    MatchStatsRead,
    MatchUpdate,
    UploadInitResponse,
)

logger = logging.getLogger("padel.api")
# Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY; the literal is
# stable across versions.
HTTP_UNPROCESSABLE = 422

router = APIRouter(prefix="/api/matches", tags=["matches"])

_UPLOAD_CHUNK_GUARD_BYTES = 8 * 1024 * 1024   # keep this much headroom free


# ── Creation and upload ──────────────────────────────────────────────────────

@router.post("", response_model=UploadInitResponse, status_code=status.HTTP_201_CREATED)
async def create_match(
    payload: MatchCreate, db: AsyncSession = Depends(get_db)
) -> UploadInitResponse:
    settings = get_settings()
    max_bytes = settings.max_video_size_mb * 1024 * 1024

    if payload.file_size_bytes and payload.file_size_bytes > max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Video troppo grande: {payload.file_size_bytes / 1_048_576:.0f} MB "
            f"(limite {settings.max_video_size_mb} MB).",
        )
    if payload.file_size_bytes and free_space_bytes() < payload.file_size_bytes * 1.2:
        raise HTTPException(
            status.HTTP_507_INSUFFICIENT_STORAGE,
            "Spazio su disco insufficiente per questo video.",
        )

    match = Match(
        title=payload.title,
        player_names=payload.player_names,
        status=MatchStatus.UPLOADING,
        video_filename="",
    )
    db.add(match)
    await db.flush()

    match.video_filename = f"{match.id}.mp4"
    await db.flush()

    return UploadInitResponse(match_id=match.id, upload_url=upload_url(match.id))


@router.put("/{match_id}/video", response_model=MatchRead)
async def upload_video(
    match_id: str, request: Request, db: AsyncSession = Depends(get_db)
) -> MatchRead:
    """Stream the video to disk, then probe it and extract the keyframe.

    Probing here — rather than in the worker — means the API can reject an
    unreadable file while the user is still looking at the upload screen, and
    guarantees that a match reaching the queue has a real video behind it.
    """
    match = await _get_match(db, match_id)
    if match.status not in (MatchStatus.UPLOADING, MatchStatus.FAILED):
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Upload non consentito nello stato '{match.status.value}'."
        )

    settings = get_settings()
    max_bytes = settings.max_video_size_mb * 1024 * 1024
    target = video_path(match_id)
    tmp = target.with_suffix(".part")
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    try:
        with open(tmp, "wb") as handle:
            async for chunk in request.stream():
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"Video oltre il limite di {settings.max_video_size_mb} MB.",
                    )
                handle.write(chunk)
        if written == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Nessun dato ricevuto.")
        tmp.replace(target)
    except HTTPException:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Scrittura su disco fallita: {exc}"
        ) from exc

    try:
        info = probe(target)
        extract_keyframe(target, keyframe_path(match_id), at_second=5.0)
    except VideoError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(HTTP_UNPROCESSABLE, str(exc)) from exc

    match.video_size_bytes = written
    match.fps = info.fps
    match.width = info.width
    match.height = info.height
    match.duration_seconds = info.duration_s
    match.status = MatchStatus.NEEDS_CALIBRATION
    match.progress = 0
    match.progress_message = "In attesa di calibrazione del campo"
    match.error_message = None
    await db.flush()
    return MatchRead.from_match(match)


@router.get("/{match_id}/keyframe")
async def get_keyframe(match_id: str, db: AsyncSession = Depends(get_db)) -> FileResponse:
    await _get_match(db, match_id)
    path = keyframe_path(match_id)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Keyframe non disponibile.")
    return FileResponse(str(path), media_type="image/jpeg")


# ── Calibration ──────────────────────────────────────────────────────────────

@router.get("/{match_id}/calibration/suggestion", response_model=CalibrationSuggestion)
async def calibration_suggestion(
    match_id: str, db: AsyncSession = Depends(get_db)
) -> CalibrationSuggestion:
    """Pre-fill the four handles. A suggestion, never a calibration."""
    match = await _get_match(db, match_id)
    path = keyframe_path(match_id)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Keyframe non disponibile.")

    frame = cv2.imread(str(path))
    if frame is None:
        raise HTTPException(HTTP_UNPROCESSABLE, "Keyframe illeggibile.")

    # An existing calibration is the best possible suggestion.
    if match.calibration and match.calibration.get("corners_px"):
        return CalibrationSuggestion(
            corners_px=[tuple(p) for p in match.calibration["corners_px"]],
            method="saved",
            confidence=1.0,
            frame_size=(frame.shape[1], frame.shape[0]),
            corner_labels=list(CORNER_LABELS),
        )

    suggestion = suggest_corners(frame)
    return CalibrationSuggestion(
        corners_px=[tuple(p) for p in suggestion.to_list()],
        method=suggestion.method,
        confidence=suggestion.confidence,
        frame_size=(frame.shape[1], frame.shape[0]),
        corner_labels=list(CORNER_LABELS),
    )


@router.post("/{match_id}/calibration", response_model=CalibrationResult)
async def submit_calibration(
    match_id: str, payload: CalibrationSubmit, db: AsyncSession = Depends(get_db)
) -> CalibrationResult:
    match = await _get_match(db, match_id)
    if match.status in (MatchStatus.QUEUED, MatchStatus.ANALYZING):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Analisi in corso: impossibile ricalibrare ora."
        )
    if not match.width or not match.height:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Carica prima il video: le dimensioni non sono note."
        )

    frame_size = (match.width, match.height)
    corners, net_px, source, preset_id, note = await _resolve_corners(db, payload, frame_size)

    try:
        calibration = build_calibration(
            corners_px=corners, frame_size=frame_size, source=source, net_px=net_px
        )
    except CalibrationError as exc:
        raise HTTPException(HTTP_UNPROCESSABLE, str(exc)) from exc

    match.calibration = {
        "corners_px": calibration.corners_px.tolist(),
        "frame_size": list(frame_size),
        "source": source.value,
        "net_px": [list(p) for p in net_px] if net_px is not None else None,
        "preset_id": preset_id,
    }
    if match.status in (MatchStatus.NEEDS_CALIBRATION, MatchStatus.FAILED, MatchStatus.COMPLETED):
        match.status = MatchStatus.READY
        match.progress_message = "Pronto per l'analisi"
        match.error_message = None

    saved_preset_id = preset_id
    if payload.save_as_preset:
        saved_preset_id = await _save_preset(
            db, payload.save_as_preset, calibration.corners_px.tolist(), frame_size, net_px
        )

    await db.flush()
    return CalibrationResult(
        ok=True,
        corners_px=[tuple(p) for p in calibration.corners_px.tolist()],
        net_error_m=calibration.net_error_m,
        preset_id=saved_preset_id,
        message=note,
    )


async def _resolve_corners(
    db: AsyncSession,
    payload: CalibrationSubmit,
    frame_size: tuple[int, int],
):
    """Corners come either from a saved preset (rescaled) or from the UI."""
    from app.ml.court import CalibrationSource

    if payload.preset_id:
        preset = await db.get(CameraPreset, payload.preset_id)
        if preset is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Preset camera non trovato.")

        corners, note = _rescale_preset(preset, frame_size)
        net = _rescale_points(preset.net_px, preset, frame_size) if preset.net_px else None
        preset.times_used += 1
        return corners, net, CalibrationSource.PRESET, preset.id, note

    if not payload.corners_px:
        raise HTTPException(
            HTTP_UNPROCESSABLE,
            "Servono le coordinate dei 4 angoli oppure un preset camera.",
        )
    source = CalibrationSource(payload.source.value)
    net = [tuple(p) for p in payload.net_px] if payload.net_px else None
    return [tuple(p) for p in payload.corners_px], net, source, None, None


def _rescale_preset(preset: CameraPreset, frame_size: tuple[int, int]):
    if (preset.frame_width, preset.frame_height) == frame_size:
        return [tuple(p) for p in preset.corners_px], None
    note = (
        f"Preset registrato a {preset.frame_width}x{preset.frame_height}, "
        f"riscalato a {frame_size[0]}x{frame_size[1]}. Verifica gli angoli."
    )
    return _rescale_points(preset.corners_px, preset, frame_size), note


def _rescale_points(points, preset: CameraPreset, frame_size: tuple[int, int]):
    sx = frame_size[0] / preset.frame_width
    sy = frame_size[1] / preset.frame_height
    return [(float(p[0]) * sx, float(p[1]) * sy) for p in points]


async def _save_preset(
    db: AsyncSession,
    name: str,
    corners: list,
    frame_size: tuple[int, int],
    net_px,
) -> str:
    existing = await db.scalar(select(CameraPreset).where(CameraPreset.name == name))
    if existing is not None:
        existing.corners_px = corners
        existing.frame_width, existing.frame_height = frame_size
        existing.net_px = [list(p) for p in net_px] if net_px else None
        return existing.id

    preset = CameraPreset(
        name=name,
        corners_px=corners,
        frame_width=frame_size[0],
        frame_height=frame_size[1],
        net_px=[list(p) for p in net_px] if net_px else None,
    )
    db.add(preset)
    await db.flush()
    return preset.id


# ── Analysis ─────────────────────────────────────────────────────────────────

@router.post("/{match_id}/start", response_model=MatchRead)
async def start_analysis(match_id: str, db: AsyncSession = Depends(get_db)) -> MatchRead:
    match = await _get_match(db, match_id)

    if not match.calibration:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Calibra il campo prima di avviare l'analisi.",
        )
    if match.status in (MatchStatus.QUEUED, MatchStatus.ANALYZING):
        raise HTTPException(status.HTTP_409_CONFLICT, "Analisi già in corso.")
    if not video_path(match_id).exists():
        raise HTTPException(status.HTTP_409_CONFLICT, "Il file video non è presente sul disco.")

    # Drop any stale queued job for this match so re-analysis cannot queue twice.
    stale = await db.scalars(
        select(Job).where(Job.match_id == match_id, Job.state == JobState.QUEUED)
    )
    for job in stale:
        await db.delete(job)

    db.add(Job(match_id=match_id))
    match.status = MatchStatus.QUEUED
    match.progress = 0
    match.progress_message = "In coda"
    match.error_message = None
    # Names are attached to player ids, and a new analysis may number the
    # people differently — the identity stage is exactly what re-analysis
    # improves. Kept, they would silently label the wrong players; cleared,
    # the app asks for them again when the analysis completes.
    match.player_names = None
    await db.flush()
    return MatchRead.from_match(match)


# ── Reads ────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[MatchRead])
async def list_matches(db: AsyncSession = Depends(get_db), limit: int = 100) -> list[MatchRead]:
    result = await db.execute(
        select(Match).order_by(Match.created_at.desc()).limit(min(limit, 500))
    )
    return [MatchRead.from_match(m) for m in result.scalars().all()]


@router.get("/{match_id}", response_model=MatchRead)
async def get_match(match_id: str, db: AsyncSession = Depends(get_db)) -> MatchRead:
    return MatchRead.from_match(await _get_match(db, match_id))


@router.patch("/{match_id}", response_model=MatchRead)
async def update_match(
    match_id: str, payload: MatchUpdate, db: AsyncSession = Depends(get_db)
) -> MatchRead:
    """Rename a match or assign player names.

    Player names used to live only in browser state, so they were lost on
    reload and never reached a second device.
    """
    match = await _get_match(db, match_id)
    if payload.title is not None:
        match.title = payload.title
    if payload.player_names is not None:
        match.player_names = [n.strip() for n in payload.player_names]
    await db.flush()
    return MatchRead.from_match(match)


@router.get("/{match_id}/stats", response_model=MatchStatsRead)
async def get_stats(match_id: str, db: AsyncSession = Depends(get_db)) -> MatchStatsRead:
    match = await _get_match(db, match_id)
    stats = await db.scalar(select(MatchStats).where(MatchStats.match_id == match_id))
    if stats is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Statistiche non ancora disponibili.")

    crops = stats.player_crops or {}
    per_player = {}
    for player_id, values in stats.per_player.items():
        entry = dict(values)
        if player_id in crops:
            entry["crop_url"] = crop_url(match_id, int(player_id))
        per_player[player_id] = entry

    return MatchStatsRead(
        match_id=match_id,
        title=match.title,
        player_names=match.player_names,
        per_player=per_player,
        heatmaps=stats.heatmaps,
        rallies=stats.rallies,
        summary=stats.summary,
        data_quality=stats.data_quality,
    )


@router.get("/{match_id}/crops/{player_id}")
async def get_crop(
    match_id: str, player_id: int, db: AsyncSession = Depends(get_db)
) -> FileResponse:
    await _get_match(db, match_id)
    path = crop_path(match_id, player_id)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Anteprima non disponibile.")
    return FileResponse(str(path), media_type="image/jpeg")


# A 204 route must declare `response_model=None`. These modules use
# `from __future__ import annotations`, so FastAPI resolves the `-> None`
# return annotation through `get_type_hints`, which normalises it to
# `NoneType` — a truthy value. FastAPI then treats it as a response model and
# asserts that a 204 cannot have a body, failing at import time. Saying so
# explicitly is version-proof and keeps the annotation for readers.
@router.delete(
    "/{match_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_match(match_id: str, db: AsyncSession = Depends(get_db)) -> None:
    match = await _get_match(db, match_id)
    if match.status == MatchStatus.ANALYZING:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Analisi in corso: attendi il termine prima di eliminare la partita.",
        )
    await db.delete(match)
    await db.flush()
    try:
        delete_match_files(match_id)
    except OSError as exc:
        logger.warning("Pulizia file del match %s incompleta: %s", match_id, exc)


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _get_match(db: AsyncSession, match_id: str) -> Match:
    match = await db.get(Match, match_id)
    if match is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Partita non trovata.")
    return match
