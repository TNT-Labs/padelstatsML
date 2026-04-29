"""Utility to ensure ML model weights are present, downloading if needed.

Set TRACKNET_WEIGHTS_URL in .env to enable automatic first-run download.
Leave empty (default) to skip: the ball tracker will fall back to MOG2.

Example .env:
    TRACKNET_WEIGHTS_URL=https://github.com/SamuReyes/TrackNetV2-padel/releases/download/v1.0/tracknet_padel.pth
"""
from __future__ import annotations

import urllib.request
from pathlib import Path


def ensure_tracknet_weights(local_path: str, url: str) -> bool:
    """Download TrackNet weights to *local_path* if they are missing.

    Returns True  — weights are available (already existed or just downloaded).
    Returns False — URL is empty; MOG2 fallback will be used automatically.
    Never raises   — download failures are logged and swallowed so the worker
                     still starts and falls back to MOG2.
    """
    path = Path(local_path)
    if path.exists():
        return True
    if not url:
        print("[weights] TRACKNET_WEIGHTS_URL not set — using MOG2 fallback for ball tracking")
        return False

    print(f"[weights] TrackNet weights not found at '{local_path}'.")
    print(f"[weights] Downloading from {url} …", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")

    try:
        def _hook(n: int, chunk: int, total: int) -> None:
            if total > 0 and n % 100 == 0:
                pct = min(100, n * chunk * 100 // total)
                print(f"[weights]   {pct}%", flush=True)

        urllib.request.urlretrieve(url, str(tmp), reporthook=_hook)
        tmp.rename(path)
        size_mb = path.stat().st_size // 1_048_576
        print(f"[weights] Saved {size_mb} MB → '{local_path}'")
        return True

    except Exception as exc:
        print(f"[weights] Download failed: {exc} — using MOG2 fallback")
        tmp.unlink(missing_ok=True)
        return False
