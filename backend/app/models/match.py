"""Domain models: Match (upload + processing state) and MatchStats (results)."""
import enum
import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class MatchStatus(str, enum.Enum):
    UPLOADING = "uploading"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200))
    status: Mapped[MatchStatus] = mapped_column(
        Enum(MatchStatus, values_callable=lambda e: [x.value for x in e]),
        default=MatchStatus.UPLOADING,
        index=True,
    )
    progress: Mapped[int] = mapped_column(default=0)  # 0..100
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Storage
    video_s3_key: Mapped[str] = mapped_column(String(500))
    duration_seconds: Mapped[float | None] = mapped_column(nullable=True)

    # Players (semplificato per MVP: 4 nomi opzionali)
    player_names: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    stats: Mapped["MatchStats | None"] = relationship(
        back_populates="match", uselist=False, cascade="all, delete-orphan"
    )


class MatchStats(Base):
    __tablename__ = "match_stats"

    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id", ondelete="CASCADE"), primary_key=True
    )

    # Per-player stats: {player_id: {distance_m, winners, errors, shots: {smash, volley, bandeja}}}
    per_player: Mapped[dict] = mapped_column(JSON)

    # Heatmap: {player_id: [[x_m, y_m, weight], ...]} in court coordinates
    heatmaps: Mapped[dict] = mapped_column(JSON)

    # Rallies: list of {start_frame, end_frame, shots_count, winner_player}
    rallies: Mapped[list] = mapped_column(JSON)

    # Court calibration data (homography matrix etc) for replay overlay
    court_calibration: Mapped[dict] = mapped_column(JSON)

    # Storage keys for per-player crop images: {"0": s3_key, "1": s3_key, ...}
    player_crops: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    match: Mapped[Match] = relationship(back_populates="stats")
