"""Sign in, sign out, who am I, change my password.

    POST /api/auth/login
    POST /api/auth/logout
    GET  /api/auth/me
    POST /api/auth/password
"""
from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    clear_session_cookie,
    session_and_user,
    set_session_cookie,
    signed_in_user,
)
from app.core.audit import record
from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import (
    DUMMY_HASH,
    as_utc,
    client_ip,
    hash_password,
    login_limiter,
    new_session_token,
    normalise_username,
    password_problem,
    utcnow,
    verify_password,
)
from app.models import AuditEntry, AuthSession, User
from app.schemas.auth import LoginRequest, Me, PasswordChange

router = APIRouter(prefix="/api/auth", tags=["auth"])

# One message for every failure: telling "no such user" from "wrong password"
# or "locked" would let anyone probe which accounts exist.
_LOGIN_FAILED = "Nome utente o password non validi, oppure account temporaneamente bloccato."


@router.post("/login", response_model=Me)
async def login(
    payload: LoginRequest, request: Request, response: Response,
    db: AsyncSession = Depends(get_db),
) -> Me:
    settings = get_settings()
    ip = client_ip(request)
    if login_limiter.blocked(ip):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Troppi tentativi da questo indirizzo: riprova tra qualche minuto.",
        )

    username = normalise_username(payload.username)
    user = await db.scalar(select(User).where(User.username == username))
    now = utcnow()
    locked = user is not None and user.locked_until is not None and as_utc(user.locked_until) > now
    password_ok = await run_in_threadpool(
        verify_password, payload.password, user.password_hash if user else DUMMY_HASH
    )

    if user is None or not user.active or locked or not password_ok:
        login_limiter.hit(ip)
        detail = None
        if user is not None and not locked and user.active and not password_ok:
            user.failed_attempts += 1
            if user.failed_attempts >= settings.max_login_attempts:
                user.locked_until = now + timedelta(minutes=settings.lock_minutes)
                user.failed_attempts = 0
                detail = f"account bloccato per {settings.lock_minutes} minuti"
        elif user is not None:
            detail = "account bloccato" if locked else "account disattivato" if not user.active else None
        record(db, "login_fallito", username or None, detail, ip)
        # Committed explicitly: the HTTPException below rolls the request back.
        await db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _LOGIN_FAILED)

    user.failed_attempts = 0
    user.locked_until = None
    user.last_login_at = now
    token, hashed = new_session_token()
    db.add(AuthSession(
        token_hash=hashed,
        user_id=user.id,
        created_at=now,
        last_seen_at=now,
        expires_at=now + timedelta(days=settings.session_max_days),
        ip=ip,
        user_agent=(request.headers.get("user-agent") or "")[:200] or None,
    ))
    # Housekeeping: sessions expired by age or inactivity, of anyone, and
    # access-log entries past their retention (they hold IP addresses).
    await db.execute(
        delete(AuthSession).where(
            (AuthSession.expires_at <= now)
            | (AuthSession.last_seen_at <= now - timedelta(hours=settings.session_idle_hours))
        )
    )
    await db.execute(
        delete(AuditEntry).where(AuditEntry.at < now - timedelta(days=settings.audit_keep_days))
    )
    record(db, "login", user.username, None, ip)
    await db.flush()
    set_session_cookie(request, response, token)
    return Me.of(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
async def logout(
    request: Request,
    found=Depends(session_and_user), db: AsyncSession = Depends(get_db),
) -> Response:
    if found is not None:
        session, user = found
        await db.delete(session)
        record(db, "logout", user.username, None, client_ip(request))
    out = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_session_cookie(request, out)
    return out


@router.get("/me", response_model=Me)
async def me(user: User = Depends(signed_in_user)) -> Me:
    return Me.of(user)


@router.post("/password", response_model=Me)
async def change_password(
    payload: PasswordChange, request: Request,
    found=Depends(session_and_user), db: AsyncSession = Depends(get_db),
) -> Me:
    if found is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sessione scaduta: accedi di nuovo.")
    session, user = found
    ip = client_ip(request)
    if login_limiter.blocked(ip):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Troppi tentativi da questo indirizzo: riprova tra qualche minuto.",
        )
    if not await run_in_threadpool(verify_password, payload.current_password, user.password_hash):
        # Same budget as the login: a stolen session must not become an
        # unlimited oracle for the password behind it.
        settings = get_settings()
        login_limiter.hit(ip)
        user.failed_attempts += 1
        detail = None
        if user.failed_attempts >= settings.max_login_attempts:
            user.locked_until = utcnow() + timedelta(minutes=settings.lock_minutes)
            user.failed_attempts = 0
            await db.execute(delete(AuthSession).where(AuthSession.user_id == user.id))
            detail = f"account bloccato per {settings.lock_minutes} minuti, sessioni chiuse"
        record(db, "cambio_password_fallito", user.username, detail, ip)
        await db.commit()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "La password attuale non è corretta.")
    problem = password_problem(payload.new_password, user.username)
    if problem:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, problem)
    if payload.new_password == payload.current_password:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "La nuova password deve essere diversa da quella attuale.")

    user.password_hash = await run_in_threadpool(hash_password, payload.new_password)
    user.must_change_password = False
    user.failed_attempts = 0
    # Whoever knew the old password is signed out everywhere else.
    await db.execute(
        delete(AuthSession).where(
            AuthSession.user_id == user.id, AuthSession.token_hash != session.token_hash
        )
    )
    record(db, "cambio_password", user.username, None, ip)
    await db.flush()
    return Me.of(user)
