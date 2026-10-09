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


# ── Accounts ─────────────────────────────────────────────────────────────────

PASSWORD = "Prova12345x"


def ensure_user(username: str, role: str = "utente", must_change: bool = False,
                password: str = PASSWORD, active: bool = True) -> str:
    """Create or reset a user directly in the database; returns its id."""
    from sqlalchemy import select

    from app.core.database import init_schema, sync_session
    from app.core.security import hash_password
    from app.models import User, UserRole

    init_schema()
    with sync_session() as session:
        user = session.scalar(select(User).where(User.username == username))
        if user is None:
            user = User(username=username, password_hash="")
            session.add(user)
        user.password_hash = hash_password(password)
        user.role = UserRole(role)
        user.must_change_password = must_change
        user.active = active
        user.failed_attempts = 0
        user.locked_until = None
        session.flush()
        return user.id


async def signed_in(username: str, role: str = "utente", **kwargs):
    """An httpx client of the app, signed in as `username`."""
    from httpx import ASGITransport, AsyncClient

    from app.core.security import login_limiter
    from app.main import app

    login_limiter.reset()
    ensure_user(username, role, **kwargs)

    def make() -> AsyncClient:
        return AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver",
            headers={"X-Padel": "1"},
        )

    # Signed in through a throwaway client, so the one returned is still
    # unopened and works with `async with`.
    async with make() as login:
        response = await login.post(
            "/api/auth/login",
            json={"username": username, "password": kwargs.get("password", PASSWORD)},
        )
        assert response.status_code == 200, response.text
        token = login.cookies.get("padel_session")
    client = make()
    client.cookies.set("padel_session", token)
    return client
