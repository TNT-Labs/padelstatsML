"""SQLite session management (async for the API, sync for the worker).

SQLite is the right database for this deployment: one node, one writer, a few
hundred rows. WAL mode lets the API read while the worker writes, which is the
only concurrency this system has.

Schema is created with `create_all` at startup — there is no migration tool.
For a single-user appliance a versioned schema stamp plus additive columns is
enough, and it removes the whole class of migration failures.
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


def init_schema() -> None:
    """Create tables that do not exist yet. Safe to call from both processes."""
    from app import models  # noqa: F401  (registers mappers)

    Base.metadata.create_all(sync_engine)
    with sync_engine.connect() as conn:
        conn.execute(text("PRAGMA optimize"))
        conn.commit()
