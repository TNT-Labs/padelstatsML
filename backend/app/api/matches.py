"""REST endpoints for match lifecycle: create -> upload -> queue -> poll -> stats."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.storage import generate_upload_url
from app.models.match import Match, MatchStatus
from app.schemas.match import MatchCreate, MatchRead, MatchStatsRead, UploadInitResponse
from app.workers.tasks import analyze_match_task

router = APIRouter(prefix="/matches", tags=["matches"])


@router.post("", response_model=UploadInitResponse, status_code=status.HTTP_201_CREATED)
async def create_match(payload: MatchCreate, db: AsyncSession = Depends(get_db)) -> UploadInitResponse:
    """Step 1: client crea match, riceve presigned URL per upload diretto S3."""
    settings = get_settings()

    if payload.file_size_bytes is not None:
        max_bytes = settings.max_video_size_mb * 1024 * 1024
        if payload.file_size_bytes > max_bytes:
            size_mb = payload.file_size_bytes / 1_048_576
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Video troppo grande: {size_mb:.0f} MB (limite {settings.max_video_size_mb} MB)",
            )

    match = Match(
        title=payload.title,
        player_names=payload.player_names,
        status=MatchStatus.UPLOADING,
        video_s3_key="",  # set below
    )
    db.add(match)
    await db.flush()  # get ID

    s3_key = f"raw/{match.id}.mp4"
    match.video_s3_key = s3_key
    upload_url = generate_upload_url(s3_key, file_size_bytes=payload.file_size_bytes)

    return UploadInitResponse(match_id=match.id, upload_url=upload_url, s3_key=s3_key)


@router.post("/{match_id}/start", response_model=MatchRead)
async def start_processing(match_id: UUID, db: AsyncSession = Depends(get_db)) -> Match:
    """Step 2: client conferma upload completato, accodiamo job di analisi."""
    match = await db.get(Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    if match.status != MatchStatus.UPLOADING:
        raise HTTPException(409, f"Cannot start: status is {match.status}")

    match.status = MatchStatus.QUEUED
    match.progress = 0
    await db.flush()

    analyze_match_task.delay(str(match.id))
    return match


@router.get("/{match_id}", response_model=MatchRead)
async def get_match(match_id: UUID, db: AsyncSession = Depends(get_db)) -> Match:
    """Step 3: polling status. Mobile chiama ogni 5-10s finché completed."""
    match = await db.get(Match, match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    return match


@router.get("/{match_id}/stats", response_model=MatchStatsRead)
async def get_stats(match_id: UUID, db: AsyncSession = Depends(get_db)) -> MatchStatsRead:
    """Step 4: stats finali quando completed."""
    from app.models.match import MatchStats

    stats = await db.scalar(select(MatchStats).where(MatchStats.match_id == match_id))
    if not stats:
        raise HTTPException(404, "Stats not available yet")

    total_shots = sum(
        sum(p.get("shots", {}).values()) for p in stats.per_player.values()
    )

    return MatchStatsRead(
        match_id=stats.match_id,
        per_player=stats.per_player,
        heatmaps=stats.heatmaps,
        rallies_count=len(stats.rallies),
        total_shots=total_shots,
    )


@router.put("/{match_id}/video", status_code=204)
async def upload_video_local(
    match_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Receive a raw video stream and write it to the local SSD.

    Used only when STORAGE_BACKEND=local.  With the S3 backend this endpoint
    is never called — clients upload directly to the presigned S3 URL instead.
    """
    settings = get_settings()
    if settings.storage_backend != "local":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Endpoint only active with local storage backend")

    match = await db.get(Match, match_id)
    if not match:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Match not found")
    if match.status != MatchStatus.UPLOADING:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Cannot upload: status is {match.status}")

    from app.core.storage import local_video_path
    video_path = local_video_path(match.video_s3_key)
    tmp_path = video_path.with_suffix(".tmp")

    try:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as fh:
            async for chunk in request.stream():
                fh.write(chunk)
        tmp_path.rename(video_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Upload failed: {exc}") from exc


@router.get("", response_model=list[MatchRead])
async def list_matches(db: AsyncSession = Depends(get_db), limit: int = 50) -> list[Match]:
    result = await db.execute(select(Match).order_by(Match.created_at.desc()).limit(limit))
    return list(result.scalars().all())
