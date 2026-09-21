"""Domain models.

Lifecycle of a match:

    UPLOADING ─upload─> NEEDS_CALIBRATION ─corners─> READY ─start─> QUEUED
                                                                      │
                                                            worker claims job
                                                                      │
                                                                  ANALYZING
                                                                   ┌──┴──┐
                                                              COMPLETED  FAILED

NEEDS_CALIBRATION is the important addition. Analysis cannot start without a
validated homography: every metric this system produces is in metres, and an
unvalidated homography yields plausible-looking numbers that are simply wrong.
Refusing to run is the honest behaviour.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
# Defined next to the geometry it describes; re-exported here so API and DB
# layers share one enum instead of two that must be kept in sync.
from app.ml.court import CalibrationSource  # noqa: F401


def _uuid_str() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Timestamps are generated in Python, not by the database.

    With a server-side default the value is unknown until the row is
    re-selected, so reading `created_at` right after a flush triggers lazy IO
    — which inside an async request raises MissingGreenlet. Generating the
    value here keeps the object complete after flush and keeps every
    timestamp in UTC.
    """
    return datetime.now(timezone.utc)


class MatchStatus(str, enum.Enum):
    UPLOADING         = "uploading"
    NEEDS_CALIBRATION = "needs_calibration"
    READY             = "ready"
    QUEUED            = "queued"
    ANALYZING         = "analyzing"
    COMPLETED         = "completed"
    FAILED            = "failed"


class JobState(str, enum.Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


def _enum_col(py_enum: type[enum.Enum]):
    """Store enum *values* (lowercase), never the member names."""
    return Enum(py_enum, values_callable=lambda e: [x.value for x in e], native_enum=False)


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    title: Mapped[str] = mapped_column(String(200))
    status: Mapped[MatchStatus] = mapped_column(
        _enum_col(MatchStatus), default=MatchStatus.UPLOADING, index=True
    )
    progress: Mapped[int] = mapped_column(Integer, default=0)          # 0..100
    progress_message: Mapped[str | None] = mapped_column(String(200), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Storage — relative to settings.videos_path
    video_filename: Mapped[str] = mapped_column(String(200))
    video_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Source video properties, filled once the upload lands
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Calibration — required before analysis can be queued.
    # {"corners_px": [[x,y] x4], "frame_size": [w,h], "source": "manual",
    #  "net_px": [[x,y],[x,y]] | null, "preset_id": "..." | null}
    calibration: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    player_names: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    stats: Mapped["MatchStats | None"] = relationship(
        back_populates="match", uselist=False, cascade="all, delete-orphan"
    )
    jobs: Mapped[list["Job"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )


class MatchStats(Base):
    """Analysis output. Every metric here is Tier 1 — derived from player
    positions only, on a validated homography. Ball-derived metrics (shot
    types, winners/errors, ball speed) are deliberately absent: they are not
    computable on a Pi 5 without a ball tracker, and fabricating them was the
    single largest source of wrong numbers in the previous version."""

    __tablename__ = "match_stats"

    match_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("matches.id", ondelete="CASCADE"), primary_key=True
    )

    # {player_id: {team, distance_m, distance_rally_m, avg_speed_ms,
    #              peak_speed_ms, zone_pct: {...}, coverage_m2,
    #              tracked_ratio, samples}}
    per_player: Mapped[dict] = mapped_column(JSON)

    # {player_id: [[x_m, y_m, weight], ...]} on a 0.5 m grid
    heatmaps: Mapped[dict] = mapped_column(JSON)

    # [{index, start_s, end_s, duration_s, players_tracked}]
    rallies: Mapped[list] = mapped_column(JSON)

    # {rallies_count, total_rally_s, avg_rally_s, median_rally_s,
    #  longest_rally_s, active_ratio, analysed_s}
    summary: Mapped[dict] = mapped_column(JSON)

    # Everything a reader needs to judge how much to trust the numbers above:
    # {calibration_source, calibration_rms_px, sample_hz, frames_sampled,
    #  players_found, side_changes, warnings: [...]}
    data_quality: Mapped[dict] = mapped_column(JSON)

    # {player_id: "crops/<match>/player_<n>.jpg"}
    player_crops: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    match: Mapped[Match] = relationship(back_populates="stats")


class CameraPreset(Base):
    """A saved camera position.

    Calibrating once per physical camera setup — not once per match — is what
    makes manual calibration acceptable in daily use: 30 seconds the first
    time, one click every time after that.
    """

    __tablename__ = "camera_presets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    corners_px: Mapped[list] = mapped_column(JSON)          # [[x,y] x4] TL,TR,BR,BL
    frame_width: Mapped[int] = mapped_column(Integer)
    frame_height: Mapped[int] = mapped_column(Integer)
    net_px: Mapped[list | None] = mapped_column(JSON, nullable=True)  # optional sanity check
    times_used: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class Job(Base):
    """Analysis queue. Replaces Celery+Redis.

    One row per attempt to analyse a match. A single worker process claims
    rows with a conditional UPDATE, so no broker is needed for a deployment
    that runs exactly one job at a time.
    """

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    match_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("matches.id", ondelete="CASCADE"), index=True
    )
    state: Mapped[JobState] = mapped_column(_enum_col(JobState), default=JobState.QUEUED, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at:   Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at:  Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    match: Mapped[Match] = relationship(back_populates="jobs")
