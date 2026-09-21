"""Shared fixtures.

DATA_DIR must be redirected before `app.core.config` is imported anywhere,
because Settings is cached with lru_cache and the database engine is created
at import time from it.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = tempfile.mkdtemp(prefix="padel-tests-")
os.environ.setdefault("DATA_DIR", _TMP)
os.environ.setdefault("DETECTOR_MODEL", str(Path(_TMP) / "missing.onnx"))
os.environ.setdefault("API_BASE_URL", "http://testserver")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(_TMP)


@pytest.fixture
def calibration():
    """A realistic camera-behind-the-baseline homography on a 1920x1080 frame."""
    import numpy as np

    from app.ml.court import CalibrationSource, build_calibration

    corners = np.array(
        [
            [600.0, 300.0],    # far-left
            [1320.0, 300.0],   # far-right
            [1800.0, 980.0],   # near-right
            [120.0, 980.0],    # near-left
        ],
        dtype=np.float32,
    )
    return build_calibration(corners, (1920, 1080), CalibrationSource.MANUAL)
