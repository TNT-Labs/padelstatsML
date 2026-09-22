"""Centralised configuration for the single-node Raspberry Pi 5 deployment.

Design constraints (decided 2026-09, Pi-5-only target):
  * No GPU, no external inference host, no NPU accelerator.
  * One user, one concurrent analysis job.
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
    api_host: str = "0.0.0.0"
    api_port: int = 8000
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
    # External base URL, used to build upload/crop URLs handed to the browser.
    api_base_url: str = "http://padelpi.local:8000"

    # ── Storage ──────────────────────────────────────────────────────────────
    # Mount the SSD here. Everything (DB, videos, crops) lives under this root.
    data_dir: str = "/data"
    max_video_size_mb: int = 4096

    # ── Detector ─────────────────────────────────────────────────────────────
    # Path to the exported YOLOv8 ONNX model (see scripts/export_yolo_onnx.py).
    # The Pi never loads a .pt file — that would pull in torch.
    detector_model: str = "weights/yolov8n.onnx"
    # Inference input size. 480 is the sweet spot on a Cortex-A76: ~1.8x faster
    # than 640 with a small recall loss on distant players.
    detector_imgsz: int = 480
    detector_conf: float = 0.30
    detector_iou: float = 0.55
    # onnxruntime intra-op threads. 0 = auto (physical cores, capped at 4).
    inference_threads: int = 0

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
    rally_speed_threshold_ms: float = 1.1
    rally_min_duration_s: float = 3.0
    rally_merge_gap_s: float = 1.5

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
    def cors_origin_list(self) -> list[str]:
        """Origins as a list. Empty means: do not enable CORS at all."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

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
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
