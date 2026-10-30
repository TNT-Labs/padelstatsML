"""Centralised configuration for the single-node Raspberry Pi 5 deployment.

Design constraints (decided 2026-09, Pi-5-only target):
  * No GPU, no external inference host. An AI HAT+ (Hailo) is used when
    fitted, for the detector only; without it everything runs on the CPU.
  * A few users with their own login, one concurrent analysis job.
  * Everything on local disk: SQLite + filesystem. No Postgres/Redis/MinIO/S3.
  * Inference runs on onnxruntime (CPU). torch/ultralytics are NOT installed
    on the Pi — they are only needed once, off-device, to export the ONNX model.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: .../backend/app/core/config.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Two candidate .env files, because the API and the worker are started
    # from `backend/` while the configuration lives at the repository root
    # (that is the file docker compose reads too). Without the first entry a
    # native install silently runs on defaults — pointing at /data instead of
    # the SSD — and the only symptom is an empty match list.
    # Later entries win, so a `backend/.env` can override the shared one.
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", ".env"),
        extra="ignore",
    )

    # ── API ──────────────────────────────────────────────────────────────────
    # Comma-separated origins; empty (the default) means same-origin only,
    # which is what you want when the API serves the web UI itself. Set it
    # only when the Vite dev server runs on another port.
    #
    # Deliberately a plain string, not list[str]: pydantic-settings decodes
    # complex field types as JSON straight from the environment source,
    # before any validator can run. A field typed list[str] therefore rejects
    # both `CORS_ORIGINS=` and `CORS_ORIGINS=http://localhost:5173` with a
    # SettingsError at import time, and accepts only JSON array syntax.
    # Read it through `cors_origin_list`.
    cors_origins: str = ""

    # Path prefix the whole app is served under, e.g. "/padel" when it shares
    # a domain with other applications behind one Cloudflare tunnel. Empty
    # (the default) serves it at the root, as a standalone install does.
    base_path: str = ""

    # ── Accesso ──────────────────────────────────────────────────────────────
    # Sessions are server-side rows; the cookie only carries a random token.
    session_idle_hours: int = 12
    session_max_days: int = 7
    # Must be true behind HTTPS (the tunnel): the browser then sends the
    # cookie only over TLS. False for a plain-HTTP LAN install.
    cookie_secure: bool = False
    max_login_attempts: int = 5
    lock_minutes: int = 15
    # The access log holds IP addresses: kept only as long as it is useful.
    audit_keep_days: int = 365
    # Read the visitor's address from CF-Connecting-IP. Only safe when nothing
    # but cloudflared can reach the API (no published port), as in compose.
    trust_cloudflare: bool = False
    # First administrator, created at start-up only when no user exists yet.
    padel_admin_username: str = ""
    padel_admin_password: str = ""

    # ── Storage ──────────────────────────────────────────────────────────────
    # Mount the SSD here. Everything (DB, videos, crops) lives under this root.
    data_dir: str = "/data"
    max_video_size_mb: int = 4096
    # Videos travel in pieces of this size: Cloudflare refuses any single
    # request above 100 MB, and a dropped connection costs one piece, not the
    # whole upload.
    upload_chunk_mb: int = 32

    # ── Detector ─────────────────────────────────────────────────────────────
    # Path to the exported YOLOv8 ONNX model (see scripts/export_yolo_onnx.py).
    # The Pi never loads a .pt file — that would pull in torch.
    detector_model: str = "weights/yolov8n.onnx"
    # Inference input size. 480 is the sweet spot on a Cortex-A76: ~1.8x faster
    # than 640 with a small recall loss on distant players.
    detector_imgsz: int = 480
    detector_conf: float = 0.30
    # Detections between this and detector_conf never start a track, but
    # extend one that lost its player, close to where it expects them: the
    # far, small, half-hidden players the detector is least sure of. Set it
    # equal to detector_conf to turn it off.
    detector_conf_weak: float = 0.12
    detector_iou: float = 0.55
    # A second detector look at the far half of the court, where players are
    # smallest. More far detections; the analysis takes ~50-80% longer on a
    # Pi. "auto": when the far players reach the network too small to be
    # found reliably (see CourtCalibration.far_player_input_px); "true" or
    # "false" to force it.
    detector_far_pass: str = "auto"
    # onnxruntime intra-op threads. 0 = auto (physical cores, capped at 4).
    inference_threads: int = 0

    # ── AI accelerator (Raspberry Pi AI HAT+, Hailo) ─────────────────────────
    # "auto": the detector runs on the accelerator when it is fitted and
    # usable, on the CPU otherwise (app/ml/accel.py). "cpu": never. "hailo":
    # its absence is an error instead of a fallback.
    accelerator: str = "auto"
    # Compiled network (.hef). Empty: the first found among weights/ and the
    # hailo-models package (accel.DEFAULT_MODELS).
    hailo_model: str = ""
    # The players' posture (17 body points) on the accelerator, alongside the
    # detector: "auto" when a pose network is found, "off" never. Without
    # the accelerator there is none. HAILO_POSE_MODEL: a .hef of one's own.
    pose: str = "auto"
    hailo_pose_model: str = ""
    # Posture a second, between the samples too: the swing of a forehand or
    # a backhand lasts a third of a second. 0: only at the samples.
    pose_hz: float = 15.0

    # ── Re-identification ───────────────────────────────────────────────────
    # OSNet exported with scripts/export_reid_onnx.py. Optional: without it,
    # players are told apart by kit colour only — which fails when team-mates
    # dress alike. Empty string disables it explicitly.
    reid_model: str = "weights/osnet_x0_25_msmt17.onnx"

    # ── Sampling ─────────────────────────────────────────────────────────────
    # Player positions are sampled at this rate regardless of source fps.
    # 5 Hz keeps path length within ~10% of ground truth for padel movement
    # while costing 6x less than full-rate processing.
    #   3 Hz  -> ~40 min for a 60 min match, distance underestimated ~15%
    #   5 Hz  -> ~70 min for a 60 min match  (default)
    #   8 Hz  -> ~110 min, marginal accuracy gain
    sample_hz: float = 5.0
    # Hard ceiling on analysed duration; longer videos are truncated.
    max_analysis_minutes: int = 120

    # ── Tracking ─────────────────────────────────────────────────────────────
    # Maximum plausible player speed (m/s). Used to gate association and to
    # reject teleport artefacts when integrating distance.
    max_player_speed_ms: float = 8.0
    # A track is dropped after this many seconds without a matching detection.
    track_max_age_s: float = 1.2

    # ── Rally segmentation ───────────────────────────────────────────────────
    # Mean speed of the players above which a point is in play. With
    # rally_speed_adaptive (default) it is the ceiling and each match picks
    # its own threshold below it — slow amateur rallies included; set
    # RALLY_SPEED_ADAPTIVE=false to use exactly this value.
    rally_speed_threshold_ms: float = 1.1
    rally_speed_adaptive: bool = True
    rally_min_duration_s: float = 3.0
    rally_merge_gap_s: float = 3.0

    # ── Artifacts ────────────────────────────────────────────────────────────
    # Keep the raw per-frame observations of each analysis (~17 MB per hour of
    # video with four players at 5 Hz). They are what make scripts/overlay.py
    # and scripts/retune.py possible: without them, changing a threshold costs
    # a full re-analysis.
    keep_artifacts: bool = True

    # ── Worker ───────────────────────────────────────────────────────────────
    worker_poll_seconds: float = 2.0
    job_max_attempts: int = 2

    # ── Derived values ───────────────────────────────────────────────────────
    @property
    def far_pass_mode(self) -> bool | None:
        """detector_far_pass as the pipeline takes it: None for auto."""
        value = str(self.detector_far_pass).strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
        return None

    @property
    def cors_origin_list(self) -> list[str]:
        """Origins as a list. Empty means: do not enable CORS at all."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def base_prefix(self) -> str:
        """base_path normalised: "" or "/something" without a trailing slash."""
        value = self.base_path.strip().rstrip("/")
        if value and not value.startswith("/"):
            value = "/" + value
        return value

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def videos_path(self) -> Path:
        return self.data_path / "videos"

    @property
    def crops_path(self) -> Path:
        return self.data_path / "crops"

    @property
    def keyframes_path(self) -> Path:
        return self.data_path / "keyframes"

    @property
    def artifacts_path(self) -> Path:
        return self.data_path / "artifacts"

    @property
    def presets_path(self) -> Path:
        """The keyframe each camera preset was calibrated on."""
        return self.data_path / "presets"

    @property
    def db_path(self) -> Path:
        return self.data_path / "padel.db"

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path}"

    @property
    def sync_database_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_path,
            self.videos_path,
            self.crops_path,
            self.keyframes_path,
            self.artifacts_path,
            self.presets_path,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
