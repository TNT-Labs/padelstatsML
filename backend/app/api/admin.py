"""User management and access log — administrators only.

    GET    /api/admin/users
    POST   /api/admin/users                      create, returns a temporary password
    PATCH  /api/admin/users/{id}                 role / active
    POST   /api/admin/users/{id}/reset-password  new temporary password
    POST   /api/admin/users/{id}/unlock
    POST   /api/admin/users/{id}/logout          close every session
    DELETE /api/admin/users/{id}                 with all their matches
    GET    /api/admin/audit
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_admin
from app.core.audit import record
from app.core.database import get_db
from app.core.security import (
    as_utc,
    client_ip,
    hash_password,
    normalise_username,
    password_problem,
    temporary_password,
    username_problem,
    utcnow,
)
from app.core.storage import delete_match_files
from app.models import AuditEntry, AuthSession, CameraPreset, Match, MatchStatus, User, UserRole
from app.schemas.auth import AuditRead, TemporaryPassword, UserCreate, UserRead, UserUpdate

logger = logging.getLogger("padel.api")

router = APIRouter(prefix="/api/admin", tags=["admin"])


async def _read(db: AsyncSession, user: User) -> UserRead:
    matches = await db.scalar(select(func.count()).select_from(Match).where(Match.owner_id == user.id))
    sessions = await db.scalar(
        select(func.count()).select_from(AuthSession).where(
            AuthSession.user_id == user.id, AuthSession.expires_at > utcnow()
        )
    )
    locked_until = as_utc(user.locked_until)
    return UserRead(
        id=user.id, username=user.username, role=user.role, active=user.active,
        must_change_password=user.must_change_password,
        locked=locked_until is not None and locked_until > utcnow(),
        last_login_at=user.last_login_at, created_at=user.created_at,
        matches=matches or 0, sessions=sessions or 0,
    )


async def _target(db: AsyncSession, user_id: str) -> User:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Utente non trovato.")
    return user


async def _other_active_admins(db: AsyncSession, user: User) -> int:
    return await db.scalar(
        select(func.count()).select_from(User).where(
            User.role == UserRole.ADMIN, User.active.is_(True), User.id != user.id
        )
    ) or 0


async def _close_sessions(db: AsyncSession, user: User) -> None:
    await db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))


def _fresh_password(username: str) -> str:
    password = temporary_password()
    while password_problem(password, username):
        password = temporary_password()
    return password


@router.get("/users", response_model=list[UserRead])
async def list_users(
    _admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> list[UserRead]:
    users = (await db.scalars(select(User).order_by(User.username))).all()
    return [await _read(db, u) for u in users]


@router.post("/users", response_model=TemporaryPassword, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate, request: Request,
    admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> TemporaryPassword:
    username = normalise_username(payload.username)
    problem = username_problem(username)
    if problem:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, problem)
    if await db.scalar(select(User.id).where(User.username == username)):
        raise HTTPException(status.HTTP_409_CONFLICT, f"L'utente «{username}» esiste già.")

    password = _fresh_password(username)
    user = User(
        username=username, role=payload.role, must_change_password=True,
        password_hash=await run_in_threadpool(hash_password, password),
    )
    db.add(user)
    await db.flush()
    record(db, "utente_creato", admin.username, f"{username} ({payload.role.value})", client_ip(request))
    return TemporaryPassword(user=await _read(db, user), temporary_password=password)


@router.patch("/users/{user_id}", response_model=UserRead)
async def update_user(
    user_id: str, payload: UserUpdate, request: Request,
    admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> UserRead:
    user = await _target(db, user_id)
    changes = []
    demoting = payload.role is not None and payload.role != user.role and user.is_admin
    deactivating = payload.active is False and user.active

    if user.id == admin.id and (demoting or deactivating):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Non puoi togliere a te stesso il ruolo di amministratore né disattivarti.",
        )
    if (demoting or (deactivating and user.is_admin)) and not await _other_active_admins(db, user):
        raise HTTPException(status.HTTP_409_CONFLICT, "Deve restare almeno un amministratore attivo.")

    if payload.role is not None and payload.role != user.role:
        user.role = payload.role
        changes.append(f"ruolo {payload.role.value}")
    if payload.active is not None and payload.active != user.active:
        user.active = payload.active
        changes.append("attivato" if payload.active else "disattivato")
    if changes:
        # A role change or deactivation applies from the next request: the
        # user signs in again with the new permissions.
        await _close_sessions(db, user)
        record(db, "utente_modificato", admin.username, f"{user.username}: {', '.join(changes)}", client_ip(request))
    await db.flush()
    return await _read(db, user)


@router.post("/users/{user_id}/reset-password", response_model=TemporaryPassword)
async def reset_password(
    user_id: str, request: Request,
    admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> TemporaryPassword:
    user = await _target(db, user_id)
    password = _fresh_password(user.username)
    user.password_hash = await run_in_threadpool(hash_password, password)
    user.must_change_password = True
    user.failed_attempts = 0
    user.locked_until = None
    await _close_sessions(db, user)
    record(db, "password_reimpostata", admin.username, user.username, client_ip(request))
    await db.flush()
    return TemporaryPassword(user=await _read(db, user), temporary_password=password)


@router.post("/users/{user_id}/unlock", response_model=UserRead)
async def unlock_user(
    user_id: str, request: Request,
    admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> UserRead:
    user = await _target(db, user_id)
    user.failed_attempts = 0
    user.locked_until = None
    record(db, "utente_sbloccato", admin.username, user.username, client_ip(request))
    await db.flush()
    return await _read(db, user)


@router.post("/users/{user_id}/logout", response_model=UserRead)
async def logout_user(
    user_id: str, request: Request,
    admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> UserRead:
    user = await _target(db, user_id)
    await _close_sessions(db, user)
    record(db, "sessioni_chiuse", admin.username, user.username, client_ip(request))
    await db.flush()
    return await _read(db, user)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def delete_user(
    user_id: str, request: Request,
    admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> None:
    user = await _target(db, user_id)
    if user.id == admin.id:
        raise HTTPException(status.HTTP_409_CONFLICT, "Non puoi eliminare il tuo stesso account.")
    if user.is_admin and user.active and not await _other_active_admins(db, user):
        raise HTTPException(status.HTTP_409_CONFLICT, "Deve restare almeno un amministratore attivo.")

    matches = (await db.scalars(select(Match).where(Match.owner_id == user.id))).all()
    if any(m.status == MatchStatus.ANALYZING for m in matches):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Una partita di questo utente è in analisi: attendi il termine prima di eliminarlo.",
        )
    match_ids = [m.id for m in matches]
    for match in matches:
        await db.delete(match)
    await db.execute(delete(CameraPreset).where(CameraPreset.owner_id == user.id))
    await _close_sessions(db, user)
    await db.delete(user)
    record(
        db, "utente_eliminato", admin.username,
        f"{user.username} con {len(match_ids)} {'partita' if len(match_ids) == 1 else 'partite'}",
        client_ip(request),
    )
    # Files go only once the rows are gone for good: a failed commit must not
    # leave matches pointing at deleted videos.
    await db.commit()
    for match_id in match_ids:
        try:
            delete_match_files(match_id)
        except OSError as exc:
            logger.warning("Pulizia file del match %s incompleta: %s", match_id, exc)


@router.get("/audit", response_model=list[AuditRead])
async def audit_log(
    limit: int = 200,
    _admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> list[AuditEntry]:
    result = await db.scalars(
        select(AuditEntry).order_by(AuditEntry.id.desc()).limit(max(1, min(limit, 1000)))
    )
    return list(result.all())
