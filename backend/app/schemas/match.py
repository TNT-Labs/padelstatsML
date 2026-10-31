"""API request/response schemas, decoupled from the ORM models."""
from __future__ import annotations

from datetime import datetime, timezone

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from app.models.match import CalibrationSource, MatchStatus


def _as_utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes. Everything is stored in UTC, so
    label it as such before serialising: without an offset the browser parses
    the timestamp as local time and every match looks hours old."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


# ── Matches ──────────────────────────────────────────────────────────────────

PlayerName = Annotated[str, Field(max_length=60)]
# "R" right-handed, "L" left-handed, None not said.
PlayerHand = Literal["R", "L"] | None
# What a high stroke found really was, seen in the video.
StrokeLabel = Literal["smash", "bandeja", "lob", "volee", "dritto", "rovescio", "altro", "nulla"]


class StrokeLabelsUpdate(BaseModel):
    """Labels to set ("<player id>:<seconds>" -> label) or clear (None)."""
    labels: dict[Annotated[str, Field(max_length=32, pattern=r"^\d+:\d+(\.\d+)?$")], StrokeLabel | None] = Field(
        max_length=500)

class HitLabelsUpdate(BaseModel):
    """All the hits marked on a match: [seconds, player id] each."""
    hits: list[tuple[Annotated[float, Field(ge=0, le=86400)], Annotated[int, Field(ge=0, le=3)]]] = Field(
        max_length=5000)


class MatchCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    player_names: list[PlayerName] | None = Field(default=None, max_length=4)
    file_size_bytes: int | None = Field(default=None, gt=0)


class MatchUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    player_names: list[PlayerName] | None = Field(default=None, max_length=4)
    player_hands: list[PlayerHand] | None = Field(default=None, max_length=4)


class UploadInitResponse(BaseModel):
    match_id: str
    # Whole-file PUT, kept for scripts and LAN clients.
    upload_url: str
    # The web app sends pieces of this size to /upload instead.
    chunk_size_bytes: int


class UploadStatus(BaseModel):
    received_bytes: int
    chunk_size_bytes: int


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
    player_hands: list[PlayerHand] | None = None
    calibrated: bool
    analysis_start_s: float | None = None
    # Only filled in for administrators, who see everyone's matches.
    owner: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at")
    def _serialise_timestamps(self, value: datetime) -> datetime:
        return _as_utc(value)

    @classmethod
    def from_match(cls, match, owner: str | None = None) -> "MatchRead":
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
            player_hands=match.player_hands,
            calibrated=bool(match.calibration),
            analysis_start_s=match.analysis_start_s,
            owner=owner,
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
    # Lens correction of the saved calibration (0: none).
    k1: float = 0.0
    # What the handles were placed on (see CalibrationSubmit.reference).
    reference: str = "corners"
    # Set when the calibration was applied by itself (camera recognised).
    note: str | None = None


class CalibrationSubmit(BaseModel):
    """Either `corners_px` (manual placement) or `preset_id` (reuse a saved
    camera position) must be supplied. When both are present the preset wins:
    it is the authoritative copy and the server rescales it to this video."""

    # The four handles, in the order of `reference`: the court's corners
    # (far-left, far-right, near-right, near-left), or with "net_service"
    # the foot of the left and right net posts, then the right and left ends
    # of the near service line. The corners are computed from them.
    corners_px: list[Point] | None = Field(default=None, min_length=4, max_length=4)
    reference: Literal["corners", "net_service"] = "corners"
    # Optional: the two ends of the net at floor level. Used as an independent
    # accuracy check, since the net is the one landmark not used in the fit.
    net_px: list[Point] | None = Field(default=None, min_length=2, max_length=2)
    source: CalibrationSource = CalibrationSource.MANUAL
    preset_id: str | None = None
    save_as_preset: str | None = Field(default=None, min_length=1, max_length=100)
    # Second of the video the analysis starts from; None keeps the current one.
    analysis_start_s: float | None = Field(default=None, ge=0)
    # Lens correction for a wide-angle lens (app.ml.court); None keeps the
    # current one.
    k1: float | None = Field(default=None, ge=0, le=0.5)


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
    # High strokes from the posture (app.ml.strokes): candidates as stored,
    # [{"t", "kind"}] as read; None without the posture.
    strokes: list[dict] | None = None
    # Hits of the ball, when it was followed.
    hits: int | None = None


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
    # Second of the video the analysis started from (0: the beginning).
    analysis_start_s: float = 0.0
    # Mean speed above which the players were taken to be in play, chosen on
    # the match; None for results from before it was.
    rally_speed_threshold_ms: float | None = None
    # The video's frame rate varied (phones): times were read from the video.
    variable_frame_rate: bool = False
    # Second detector look at the far half: {"active", "added"}, when enabled.
    far_pass: dict | None = None
    # {"weak_detections", "feet_below_net", "scenery"}: how the tracker got
    # its players. Listed here or the response drops it, as it did.
    tracking: dict | None = None
    # The players' posture on the accelerator: {"model", "frames"}.
    pose: dict | None = None
    # The ball, when followed: the network, frames looked at, frames with it.
    ball: dict | None = None
    # Last accuracy check against points marked by hand (app.ml.verification).
    verification: dict | None = None


class RallyRead(BaseModel):
    index: int
    start_s: float
    end_s: float
    duration_s: float
    players_tracked: int
    # Opened by a recognised serve; None for analyses older than the check.
    serve: bool | None = None
    # [time, player id] of each hit, from the ball; None without the ball.
    hits: list[list[float]] | None = None


class AnalysisTime(BaseModel):
    """How long the last analysis took, from the worker's own clock: when
    it took the job and when it finished it (the wait in the queue left out)."""
    seconds: float
    finished_at: datetime

    @field_serializer("finished_at")
    def _serialise_finished(self, value: datetime) -> datetime:
        return _as_utc(value)


class MatchStatsRead(BaseModel):
    match_id: str
    title: str
    player_names: list[str] | None
    player_hands: list[PlayerHand] | None = None
    stroke_labels: dict[str, str] | None = None
    hit_labels: list[list[float]] | None = None
    per_player: dict[str, PlayerStats]
    heatmaps: dict[str, list[list[float]]]
    rallies: list[RallyRead]
    summary: MatchSummary
    data_quality: DataQuality
    analysis_time: AnalysisTime | None = None


# ── Camera presets ───────────────────────────────────────────────────────────

class PresetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    corners_px: list[Point] = Field(min_length=4, max_length=4)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    net_px: list[Point] | None = Field(default=None, min_length=2, max_length=2)
    k1: float = Field(default=0.0, ge=0, le=0.5)


class PresetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    corners_px: list
    frame_width: int
    frame_height: int
    net_px: list | None
    k1: float | None = None
    times_used: int
    created_at: datetime

    @field_serializer("created_at")
    def _serialise_created_at(self, value: datetime) -> datetime:
        return _as_utc(value)
