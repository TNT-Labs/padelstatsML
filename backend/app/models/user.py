"""Accounts, sessions and the access log.

Only users created by an administrator can sign in. A session is a row here,
not a signed token: logging out, deactivating a user or resetting a password
takes effect on the very next request, which a stateless token cannot do.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.match import _enum_col, _utcnow, _uuid_str


class UserRole(str, enum.Enum):
    ADMIN  = "admin"
    UTENTE = "utente"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    # Always stored lower-case: the login is case-insensitive.
    username: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[UserRole] = mapped_column(_enum_col(UserRole), default=UserRole.UTENTE)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Set at creation and after every reset: the temporary password was seen
    # by an administrator, so the user must replace it before doing anything.
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=True)
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    # SHA-256 of the cookie value: a copy of the database does not hand out
    # working sessions.
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(200), nullable=True)


class AuditEntry(Base):
    """Who did what: sign-ins, failures, account changes."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    # Kept as text: the entry must survive the deletion of the account.
    username: Mapped[str | None] = mapped_column(String(40), nullable=True)
    event: Mapped[str] = mapped_column(String(40))
    detail: Mapped[str | None] = mapped_column(String(300), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
