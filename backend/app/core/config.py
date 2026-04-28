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

    # ML
    ml_device: str = "cuda"  # "cpu" for fallback
    yolo_weights: str = "yolov8n.pt"
    yolo_pose_weights: str = "yolov8n-pose.pt"
    tracknet_weights: str = "weights/tracknet_padel.pth"
    target_fps: int = 30
    max_video_size_mb: int = 2048


@lru_cache
def get_settings() -> Settings:
    return Settings()
