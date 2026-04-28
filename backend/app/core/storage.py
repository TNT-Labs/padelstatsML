"""S3/MinIO storage abstraction.

Presigned URLs: il client carica DIRETTAMENTE su S3 senza passare dal nostro server.
Questo è critico per video da 500MB-2GB: evita di saturare la banda dell'API.
"""
from functools import lru_cache

import boto3
from botocore.client import Config

from app.core.config import get_settings


@lru_cache
def get_s3_client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket() -> None:
    s = get_settings()
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=s.s3_bucket_videos)
    except Exception:
        client.create_bucket(Bucket=s.s3_bucket_videos)


def generate_upload_url(s3_key: str, expires_in: int = 3600) -> str:
    """Presigned PUT URL. Client uploads directly to S3."""
    s = get_settings()
    return get_s3_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": s.s3_bucket_videos, "Key": s3_key, "ContentType": "video/mp4"},
        ExpiresIn=expires_in,
    )


def download_to_path(s3_key: str, local_path: str) -> None:
    """Used by ML worker to fetch video for processing."""
    s = get_settings()
    get_s3_client().download_file(s.s3_bucket_videos, s3_key, local_path)
