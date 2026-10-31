"""Marking the ball by hand, frame by frame, to train a ball detector.

    GET  /api/matches/{id}/ball/clips                          clips to mark, and progress
    GET  /api/matches/{id}/ball/clips/{clip}/frames/{i}.jpg    one frame, full size
    GET  /api/matches/{id}/ball/labels                         every mark so far
    PATCH /api/matches/{id}/ball/labels                        set or clear marks
    GET  /api/matches/{id}/ball/export.zip                     the marked clips, for training

For whoever may change the match: marking is work on the match's data. See
app/ml/ball_data.py.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_editor
from app.api.matches import _get_match
from app.core.config import get_settings
from app.core.database import get_db
from app.core.storage import video_path
from app.ml.ball_data import NOT_VISIBLE, BallData, ball_dir, choose_clips
from app.models import Match, MatchStats, MatchStatus, User

router = APIRouter(prefix="/api/matches", tags=["ball"])

# Cutting a clip out of the video takes a second or two: one at a time.
_EXTRACT = asyncio.Lock()

FrameKey = Annotated[str, Field(pattern=r"^\d{1,8}$")]
Point = Annotated[list[Annotated[float, Field(ge=0, le=10000)]], Field(min_length=2, max_length=2)]


class LabelsUpdate(BaseModel):
    """Per frame number: [x, y] in video pixels, "none" when the ball is not
    in sight, null to clear."""
    labels: dict[FrameKey, Point | Literal["none"] | None] = Field(max_length=200)


async def _data(db: AsyncSession, match_id: str, user: User) -> tuple[Match, BallData]:
    match = await _get_match(db, match_id, user)
    if match.status != MatchStatus.COMPLETED:
        raise HTTPException(status.HTTP_409_CONFLICT, "Si annota la palla su una partita già analizzata.")
    return match, BallData(ball_dir(get_settings().data_path, match_id))


@router.get("/{match_id}/ball/clips")
async def ball_clips(match_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_editor)) -> dict:
    match, data = await _data(db, match_id, user)
    clips = data.clips()
    if clips is None:
        stats = await db.scalar(select(MatchStats).where(MatchStats.match_id == match_id))
        clips = choose_clips(stats.rallies if stats else [], float(match.fps or 30.0))
        if not clips:
            raise HTTPException(status.HTTP_409_CONFLICT, "Nessuno scambio abbastanza lungo da annotare.")
        data.save_clips(clips)
    labels = data.labels()
    return {
        "frame_size": [match.width, match.height],
        "fps": match.fps,
        "clips": [
            {"id": c.id, "start_frame": c.start_frame, "frames": c.frames, "start_s": c.start_s,
             "labelled": sum(str(c.start_frame + i) in labels for i in range(c.frames))}
            for c in clips
        ],
        "labelled": len(labels),
        "visible": sum(isinstance(v, list) for v in labels.values()),
    }


@router.get("/{match_id}/ball/clips/{clip_id}/frames/{offset}.jpg")
async def ball_frame(
    match_id: str, clip_id: int, offset: int,
    db: AsyncSession = Depends(get_db), user: User = Depends(require_editor),
) -> FileResponse:
    _, data = await _data(db, match_id, user)
    clip = data.clip(clip_id)
    if clip is None or not 0 <= offset < clip.frames:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Fotogramma non trovato.")
    path = data.frame_path(clip, offset)
    if not path.exists():
        async with _EXTRACT:
            if not path.exists():
                source = video_path(match_id)
                if not source.exists():
                    raise HTTPException(status.HTTP_404_NOT_FOUND, "Il video non è più sul Pi.")
                try:
                    await run_in_threadpool(data.extract, clip, source)
                except OSError as exc:
                    raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Fotogramma oltre la fine del video.")
    # The same frame never changes: the browser may keep it.
    return FileResponse(str(path), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@router.get("/{match_id}/ball/labels")
async def ball_labels(match_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_editor)) -> dict:
    _, data = await _data(db, match_id, user)
    return {"labels": data.labels()}


@router.patch("/{match_id}/ball/labels")
async def label_ball(
    match_id: str, payload: LabelsUpdate,
    db: AsyncSession = Depends(get_db), user: User = Depends(require_editor),
) -> dict:
    match, data = await _data(db, match_id, user)
    width, height = match.width or 10000, match.height or 10000
    for value in payload.labels.values():
        if isinstance(value, list) and not (value[0] <= width and value[1] <= height):
            raise HTTPException(422, "Punto fuori dal fotogramma.")
    labels = await run_in_threadpool(
        data.update_labels, {k: (NOT_VISIBLE if v == "none" else v) for k, v in payload.labels.items()})
    return {"labelled": len(labels), "visible": sum(isinstance(v, list) for v in labels.values())}


@router.get("/{match_id}/ball/export.zip")
async def ball_export(match_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(require_editor)):
    match, data = await _data(db, match_id, user)
    if not data.labels():
        raise HTTPException(status.HTTP_409_CONFLICT, "Nessun fotogramma annotato.")
    target = data.directory / "export.zip"
    async with _EXTRACT:
        await run_in_threadpool(data.export_zip, match_id, (match.width or 1920, match.height or 1080), target)
    return FileResponse(str(target), media_type="application/zip",
                        filename=f"palla_{match_id[:8]}.zip")
