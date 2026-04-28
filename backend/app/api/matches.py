"""REST endpoints for match lifecycle: create -> upload -> queue -> poll -> stats."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.storage import generate_upload_url
from app.models.match import Match, MatchStatus
from app.schemas.match import MatchCreate, MatchRead, MatchStatsRead, UploadInitResponse
from app.workers.tasks import analyze_match_task

router = APIRouter(prefix="/matches", tags=["matches"])


@router.post("", response_model=UploadInitResponse, status_code=status.HTTP_201_CREATED)
async def create_match(payload: MatchCreate, db: AsyncSession = Depends(get_db)) -> UploadInitResponse:
    """Step 1: client crea match, riceve presigned URL per upload diretto S3."""
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
    upload_url = generate_upload_url(s3_key)

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


@router.get("", response_model=list[MatchRead])
async def list_matches(db: AsyncSession = Depends(get_db), limit: int = 50) -> list[Match]:
    result = await db.execute(select(Match).order_by(Match.created_at.desc()).limit(limit))
    return list(result.scalars().all())
