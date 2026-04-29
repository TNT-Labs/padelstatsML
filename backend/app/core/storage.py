"""Storage abstraction: S3/MinIO or local filesystem.

Set STORAGE_BACKEND=local in .env to use a mounted SSD instead of MinIO.

S3 backend (default):
  - Client uploads the video DIRECTLY to MinIO/S3 via a presigned PUT URL.
  - API server is never in the upload path — no bandwidth bottleneck.

Local backend:
  - Client uploads the video to PUT /matches/{id}/video on the API.
  - API streams the body to VIDEOS_DIR on the local filesystem.
  - MinIO container is not needed; saves ~250 MB RAM on Raspberry Pi.
"""
from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings


# ── Local filesystem helpers ─────────────────────────────────────────────────

def _videos_dir() -> Path:
    path = Path(get_settings().videos_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_video_path(s3_key: str) -> Path:
    """Map s3_key ('raw/{uuid}.mp4') to its absolute path on the local SSD."""
    return _videos_dir() / Path(s3_key).name


# ── S3/MinIO helpers ─────────────────────────────────────────────────────────

@lru_cache
def get_s3_client():
    import boto3
    from botocore.client import Config
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
        config=Config(signature_version="s3v4"),
    )


# ── Public API (backend-agnostic) ────────────────────────────────────────────

def ensure_storage() -> None:
    """Create the S3 bucket (or local videos directory) on first startup."""
    s = get_settings()
    if s.storage_backend == "local":
        _videos_dir()
        return

    from botocore.exceptions import ClientError
    try:
        get_s3_client().head_bucket(Bucket=s.s3_bucket_videos)
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            get_s3_client().create_bucket(Bucket=s.s3_bucket_videos)
        else:
            raise


# Alias kept for the main.py lifespan import
ensure_bucket = ensure_storage


def generate_upload_url(
    s3_key: str,
    expires_in: int = 3600,
    file_size_bytes: int | None = None,
) -> str:
    """Return the URL the client should PUT the video to.

    S3 backend  → presigned PUT URL pointing directly at MinIO/S3.
    Local backend → URL of the API's streaming upload endpoint
                    (PUT /matches/{match_id}/video).
    """
    s = get_settings()

    if s.storage_backend == "local":
        match_id = Path(s3_key).stem      # "raw/{uuid}.mp4" → "{uuid}"
        return f"{s.api_base_url}/matches/{match_id}/video"

    params: dict = {
        "Bucket": s.s3_bucket_videos,
        "Key": s3_key,
        "ContentType": "video/mp4",
    }
    if file_size_bytes is not None:
        params["ContentLength"] = file_size_bytes
    return get_s3_client().generate_presigned_url(
        "put_object",
        Params=params,
        ExpiresIn=expires_in,
    )


def download_to_path(s3_key: str, local_path: str) -> None:
    """Copy/download the video into the worker's temp directory."""
    s = get_settings()
    if s.storage_backend == "local":
        shutil.copy2(str(local_video_path(s3_key)), local_path)
        return

    get_s3_client().download_file(s.s3_bucket_videos, s3_key, local_path)


# ── Player crop helpers ──────────────────────────────────────────────────────

def _crop_local_path(match_id: str, player_id: int) -> Path:
    path = _videos_dir() / "crops" / match_id
    path.mkdir(parents=True, exist_ok=True)
    return path / f"player_{player_id}.jpg"


def save_crop(match_id: str, player_id: int, image_bytes: bytes) -> str:
    """Persist a player crop image; return the storage key."""
    s = get_settings()
    s3_key = f"crops/{match_id}/player_{player_id}.jpg"

    if s.storage_backend == "local":
        _crop_local_path(match_id, player_id).write_bytes(image_bytes)
    else:
        import io
        get_s3_client().upload_fileobj(
            io.BytesIO(image_bytes),
            s.s3_bucket_videos,
            s3_key,
            ExtraArgs={"ContentType": "image/jpeg"},
        )
    return s3_key


def delete_stored_files(match_id: str, video_s3_key: str) -> None:
    """Remove the video and crop files for a deleted match."""
    s = get_settings()
    if s.storage_backend == "local":
        video = local_video_path(video_s3_key)
        video.unlink(missing_ok=True)
        crop_dir = _videos_dir() / "crops" / match_id
        if crop_dir.exists():
            shutil.rmtree(crop_dir)
        return

    client = get_s3_client()
    bucket = s.s3_bucket_videos
    # Delete video
    client.delete_object(Bucket=bucket, Key=video_s3_key)
    # Delete crop objects (list then batch-delete)
    prefix = f"crops/{match_id}/"
    paginator = client.get_paginator("list_objects_v2")
    objects = [
        {"Key": obj["Key"]}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix)
        for obj in page.get("Contents", [])
    ]
    if objects:
        client.delete_objects(Bucket=bucket, Delete={"Objects": objects})


def get_crop_url(match_id: str, player_id: int, expires_in: int = 86400) -> str:
    """Return a URL to serve the player crop image.

    Local backend → API endpoint served by FastAPI.
    S3 backend    → presigned GET URL (24 h default).
    """
    s = get_settings()
    if s.storage_backend == "local":
        return f"{s.api_base_url}/matches/{match_id}/crops/{player_id}"

    return get_s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": s.s3_bucket_videos, "Key": f"crops/{match_id}/player_{player_id}.jpg"},
        ExpiresIn=expires_in,
    )
