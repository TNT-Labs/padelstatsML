"""API request/response schemas, decoupled from the ORM models."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from app.models.match import CalibrationSource, MatchStatus


def _as_utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes. Everything is stored in UTC, so
    label it as such before serialising: without an offset the browser parses
    the timestamp as local time and every match looks hours old."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


# ── Matches ──────────────────────────────────────────────────────────────────

class MatchCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    player_names: list[str] | None = Field(default=None, max_length=4)
    file_size_bytes: int | None = Field(default=None, gt=0)


class MatchUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    player_names: list[str] | None = Field(default=None, max_length=4)


class UploadInitResponse(BaseModel):
    match_id: str
    upload_url: str


class MatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    status: MatchStatus
    progress: int
    progress_message: str | None
    error_message: str | None
    duration_seconds: float | None
    fps: float | None
    width: int | None
    height: int | None
    player_names: list[str] | None
    calibrated: bool
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def _serialise_timestamps(self, value: datetime) -> datetime:
        return _as_utc(value)

    @classmethod
    def from_match(cls, match) -> "MatchRead":
        return cls(
            id=match.id,
            title=match.title,
            status=match.status,
            progress=match.progress,
            progress_message=match.progress_message,
            error_message=match.error_message,
            duration_seconds=match.duration_seconds,
            fps=match.fps,
            width=match.width,
            height=match.height,
            player_names=match.player_names,
            calibrated=bool(match.calibration),
            created_at=match.created_at,
            updated_at=match.updated_at,
        )


# ── Calibration ──────────────────────────────────────────────────────────────

Point = tuple[float, float]


class CalibrationSuggestion(BaseModel):
    """Starting point for the calibration UI — never a calibration itself."""

    corners_px: list[Point]
    method: str
    confidence: float
    frame_size: tuple[int, int]
    corner_labels: list[str]


class CalibrationSubmit(BaseModel):
    """Either `corners_px` (manual placement) or `preset_id` (reuse a saved
    camera position) must be supplied. When both are present the preset wins:
    it is the authoritative copy and the server rescales it to this video."""

    # Order: far-left, far-right, near-right, near-left.
    corners_px: list[Point] | None = Field(default=None, min_length=4, max_length=4)
    # Optional: the two ends of the net at floor level. Used as an independent
    # accuracy check, since the net is the one landmark not used in the fit.
    net_px: list[Point] | None = Field(default=None, min_length=2, max_length=2)
    source: CalibrationSource = CalibrationSource.MANUAL
    preset_id: str | None = None
    save_as_preset: str | None = Field(default=None, min_length=1, max_length=100)


class CalibrationResult(BaseModel):
    ok: bool
    corners_px: list[Point]
    net_error_m: float | None = None
    preset_id: str | None = None
    message: str | None = None


# ── Stats ────────────────────────────────────────────────────────────────────

class ZoneShare(BaseModel):
    net: float
    mid: float
    back: float


class PlayerStats(BaseModel):
    team: int
    samples: int
    tracked_ratio: float
    distance_m: float
    distance_rally_m: float
    avg_speed_ms: float
    peak_speed_ms: float
    coverage_m2: float
    zone_pct: ZoneShare
    rejected_steps: int = 0
    source_tracklets: int = 0
    # "drive" or "reves" when players were found by their side of the pair.
    role: str | None = None
    crop_url: str | None = None


class MatchSummary(BaseModel):
    analysed_s: float
    rallies_count: int
    total_rally_s: float
    avg_rally_s: float
    median_rally_s: float
    longest_rally_s: float
    active_ratio: float
    players_found: int


class DataQuality(BaseModel):
    calibration_source: str
    net_error_m: float | None = None
    sample_hz: float
    frames_sampled: int
    players_found: int
    clusters_found: int = 0
    side_changes: int = 0
    detector_model: str = ""
    # {"cue": "reid", "same", "different", "veto", "pairs"} or {"cue": "colore",
    # "reason"}; None for results produced before re-identification existed.
    identity_cue: dict | None = None
    metrics_tier: int = 1
    excluded_metrics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RallyRead(BaseModel):
    index: int
    start_s: float
    end_s: float
    duration_s: float
    players_tracked: int


class MatchStatsRead(BaseModel):
    match_id: str
    title: str
    player_names: list[str] | None
    per_player: dict[str, PlayerStats]
    heatmaps: dict[str, list[list[float]]]
    rallies: list[RallyRead]
    summary: MatchSummary
    data_quality: DataQuality


# ── Camera presets ───────────────────────────────────────────────────────────

class PresetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    corners_px: list[Point] = Field(min_length=4, max_length=4)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    net_px: list[Point] | None = Field(default=None, min_length=2, max_length=2)


class PresetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    corners_px: list
    frame_width: int
    frame_height: int
    net_px: list | None
    times_used: int
    created_at: datetime

    @field_serializer("created_at")
    def _serialise_created_at(self, value: datetime) -> datetime:
        return _as_utc(value)
