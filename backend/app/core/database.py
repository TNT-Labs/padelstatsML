"""SQLite session management (async for the API, sync for the worker).

SQLite is the right database for this deployment: one node, one writer, a few
hundred rows. WAL mode lets the API read while the worker writes, which is the
only concurrency this system has.

Schema is created with `create_all` at startup — there is no migration tool.
A versioned schema stamp (`PRAGMA user_version`) plus small, explicit steps
for existing databases is enough here, and it removes the whole class of
migration failures.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()
settings.ensure_dirs()


class Base(DeclarativeBase):
    pass


def _apply_pragmas(dbapi_conn, _record) -> None:
    """WAL + a generous busy timeout: the worker holds write locks for a while."""
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA busy_timeout=10000")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


# ── Async engine (API) ───────────────────────────────────────────────────────

async_engine = create_async_engine(
    settings.database_url,
    connect_args={"timeout": 10},
)
event.listen(async_engine.sync_engine, "connect", _apply_pragmas)

AsyncSessionLocal = async_sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Sync engine (worker) ─────────────────────────────────────────────────────

sync_engine: Engine = create_engine(
    settings.sync_database_url,
    connect_args={"timeout": 10},
)
event.listen(sync_engine, "connect", _apply_pragmas)

SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False)


@contextmanager
def sync_session() -> Iterator[Session]:
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# 1: accounts — matches and camera presets get an owner.
SCHEMA_VERSION = 1


def init_schema(engine: Engine | None = None) -> None:
    """Create missing tables and bring an older database up to date.

    Safe to call from both processes at once: everything runs inside one
    `BEGIN IMMEDIATE` transaction, so the second caller waits for the first
    and then finds nothing left to do. SQLite DDL is transactional, so a
    failure leaves the database exactly as it was.
    """
    from app import models  # noqa: F401  (registers mappers)

    engine = engine or sync_engine
    with engine.connect() as conn:
        conn.exec_driver_sql("BEGIN IMMEDIATE")
        version = conn.exec_driver_sql("PRAGMA user_version").scalar() or 0
        if version < 1:
            _v1_before_create(conn)
        Base.metadata.create_all(conn)
        if version < 1:
            _v1_after_create(conn)
        if version < SCHEMA_VERSION:
            conn.exec_driver_sql(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        conn.exec_driver_sql("PRAGMA optimize")


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")}


def _v1_before_create(conn) -> None:
    """camera_presets had a global UNIQUE on name; it becomes unique per
    owner. SQLite cannot drop a constraint, so the table is rebuilt: the old
    one steps aside here and its rows are copied back after create_all."""
    if _columns(conn, "camera_presets") and "owner_id" not in _columns(conn, "camera_presets"):
        conn.exec_driver_sql("ALTER TABLE camera_presets RENAME TO camera_presets_v0")


def _v1_after_create(conn) -> None:
    if "owner_id" not in _columns(conn, "matches"):
        conn.exec_driver_sql(
            "ALTER TABLE matches ADD COLUMN owner_id VARCHAR(36) REFERENCES users (id)"
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_matches_owner_id ON matches (owner_id)"
        )
    if _columns(conn, "camera_presets_v0"):
        cols = (
            "id, name, corners_px, frame_width, frame_height, net_px, "
            "times_used, created_at, updated_at"
        )
        conn.exec_driver_sql(
            f"INSERT INTO camera_presets ({cols}) SELECT {cols} FROM camera_presets_v0"
        )
        conn.exec_driver_sql("DROP TABLE camera_presets_v0")
