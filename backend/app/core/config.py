"""Centralized configuration. All env-vars loaded once, type-validated."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]  # restrict in prod

    # Database
    database_url: str = "postgresql+asyncpg://padel:padel@localhost:5432/padel"
    sync_database_url: str = "postgresql://padel:padel@localhost:5432/padel"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # S3 / MinIO
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "padel"
    s3_secret_key: str = "padelpadel"
    s3_bucket_videos: str = "padel-videos"
    s3_region: str = "us-east-1"

    # Storage backend: "s3" uses MinIO/AWS S3; "local" writes to a mounted SSD.
    # On Raspberry Pi set to "local" to skip MinIO entirely.
    storage_backend: str = "s3"
    # Root directory for local video storage (used only when storage_backend="local").
    # Mount your SSD at this path (e.g. /mnt/ssd) and set accordingly.
    videos_dir: str = "/data/videos"
    # External base URL of this API — used to build upload URLs for local storage.
    # Must be reachable by mobile/web clients (e.g. http://192.168.1.42 or http://padelpi.local).
    api_base_url: str = "http://localhost:8000"

    # ML
    ml_device: str = "cpu"   # "cuda" when a GPU is available
    yolo_weights: str = "yolov8n.pt"
    yolo_pose_weights: str = "yolov8n-pose.pt"
    tracknet_weights: str = "weights/tracknet_padel.pth"
    # Set to a direct download URL to auto-fetch weights on first worker startup.
    # Leave empty to use the MOG2 background-subtraction fallback instead.
    # Example: https://github.com/<user>/<repo>/releases/download/v1.0/tracknet_padel.pth
    tracknet_weights_url: str = ""
    target_fps: int = 30
    max_video_size_mb: int = 2048
    # Process every Nth frame for player tracking (higher = faster, lower accuracy)
    # Recommended: 2 on GPU, 3-4 on Pi 5
    player_stride: int = 2
    torch_num_threads: int = 0  # 0 = auto (min(cpu_count, 4))


@lru_cache
def get_settings() -> Settings:
    return Settings()
