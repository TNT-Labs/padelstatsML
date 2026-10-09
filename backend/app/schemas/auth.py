"""Accounts, sessions and the access log as the API exposes them."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_serializer

from app.core.security import as_utc
from app.models.user import UserRole


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=500)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=500)
    new_password: str = Field(min_length=1, max_length=500)


class Me(BaseModel):
    id: str
    username: str
    role: UserRole
    must_change_password: bool

    @classmethod
    def of(cls, user) -> "Me":
        return cls(id=user.id, username=user.username, role=user.role,
                   must_change_password=user.must_change_password)


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    role: UserRole = UserRole.UTENTE


class UserUpdate(BaseModel):
    role: UserRole | None = None
    active: bool | None = None


class UserRead(BaseModel):
    id: str
    username: str
    role: UserRole
    active: bool
    must_change_password: bool
    locked: bool
    last_login_at: datetime | None
    created_at: datetime
    matches: int = 0
    sessions: int = 0

    @field_serializer("last_login_at", "created_at")
    def _utc(self, value: datetime | None) -> datetime | None:
        return as_utc(value)


class TemporaryPassword(BaseModel):
    user: UserRead
    temporary_password: str


class AuditRead(BaseModel):
    at: datetime
    username: str | None
    event: str
    detail: str | None
    ip: str | None

    @field_serializer("at")
    def _utc(self, value: datetime) -> datetime:
        return as_utc(value)
