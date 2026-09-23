"""Local filesystem storage. There is no S3/MinIO backend any more.

On a single Pi with an attached SSD, an object store buys nothing and costs
~250 MB of RAM. Videos, keyframes and crops are plain files under DATA_DIR.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from app.core.config import get_settings


def video_path(match_id: str) -> Path:
    return get_settings().videos_path / f"{match_id}.mp4"


def keyframe_path(match_id: str) -> Path:
    return get_settings().keyframes_path / f"{match_id}.jpg"


def artifacts_dir(match_id: str) -> Path:
    """Per-match directory holding the raw observations of the analysis."""
    return get_settings().artifacts_path / match_id


def crop_path(match_id: str, player_id: int) -> Path:
    d = get_settings().crops_path / match_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"player_{player_id}.jpg"


def save_crop(match_id: str, player_id: int, image_bytes: bytes) -> str:
    """Persist a crop and return its storage key (relative path)."""
    crop_path(match_id, player_id).write_bytes(image_bytes)
    return f"crops/{match_id}/player_{player_id}.jpg"


def upload_url(match_id: str) -> str:
    """URL the browser PUTs the video to. The API is in the upload path, which
    is fine on a LAN: the Pi writes ~50 MB/s to the SSD, far above Wi-Fi."""
    base = get_settings().api_base_url.rstrip("/")
    return f"{base}/api/matches/{match_id}/video"


def crop_url(match_id: str, player_id: int) -> str:
    """Relative, so it works from whatever address the browser reached the
    Pi at. An absolute URL built on API_BASE_URL broke every thumbnail when
    that setting did not match the address actually used (padelpi.local in
    the .env, the Pi's IP in the browser).

    The version is the file's modification time: a re-analysis rewrites the
    thumbnails under the same path, possibly for different people, and the
    browser must not keep showing the old ones."""
    path = get_settings().crops_path / match_id / f"player_{player_id}.jpg"
    version = int(path.stat().st_mtime) if path.exists() else 0
    return f"/api/matches/{match_id}/crops/{player_id}?v={version}"


def keyframe_url(match_id: str) -> str:
    base = get_settings().api_base_url.rstrip("/")
    return f"{base}/api/matches/{match_id}/keyframe"


def delete_match_files(match_id: str) -> None:
    """Best-effort removal of every file belonging to a match."""
    video_path(match_id).unlink(missing_ok=True)
    keyframe_path(match_id).unlink(missing_ok=True)
    for directory in (get_settings().crops_path / match_id, artifacts_dir(match_id)):
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)


def free_space_bytes() -> int:
    return shutil.disk_usage(get_settings().data_path).free
