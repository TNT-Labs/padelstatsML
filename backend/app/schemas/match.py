"""API request/response schemas. Decoupled from DB models."""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.match import MatchStatus


class MatchCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    player_names: list[str] | None = Field(default=None, max_length=4)
    # Optional: client declares file size so the API can reject oversized videos
    # before the upload starts, and sign the presigned URL with ContentLength.
    file_size_bytes: int | None = Field(default=None, gt=0)


class UploadInitResponse(BaseModel):
    match_id: UUID
    upload_url: str  # S3 presigned PUT
    s3_key: str


class MatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    status: MatchStatus
    progress: int
    error_message: str | None
    duration_seconds: float | None
    player_names: list[str] | None
    created_at: datetime
    updated_at: datetime


class PlayerStats(BaseModel):
    distance_m: float
    winners: int
    errors: int
    shots: dict[str, int]  # {"smash": n, "volley": n, "bandeja": n, "other": n}
    crop_url: str | None = None


class MatchStatsRead(BaseModel):
    match_id: UUID
    per_player: dict[str, PlayerStats]
    heatmaps: dict[str, list[list[float]]]  # player_id -> [[x, y, weight], ...]
    rallies_count: int
    total_shots: int
