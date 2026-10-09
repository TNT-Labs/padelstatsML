"""Access log: sign-ins, failures and account changes."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditEntry


def record(
    db: AsyncSession,
    event: str,
    username: str | None,
    detail: str | None = None,
    ip: str | None = None,
) -> None:
    """Added to the request's transaction: it is written only if the action
    it describes is."""
    db.add(AuditEntry(
        event=event,
        username=(username or None) and username[:40],
        detail=detail[:300] if detail else None,
        ip=ip,
    ))
