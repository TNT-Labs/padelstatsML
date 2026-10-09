"""Who is calling: the session cookie, resolved to a user on every request."""
from __future__ import annotations

from datetime import timedelta

from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import as_utc, token_hash, utcnow
from app.models import AuthSession, User

COOKIE_NAME = "padel_session"
# Updating last_seen_at on every request would turn each read into a write.
_TOUCH_EVERY = timedelta(minutes=1)


def cookie_path(request: Request) -> str:
    """The prefix the application is mounted under (BASE_PATH): the cookie
    then never reaches the other applications on the same domain."""
    return request.scope.get("root_path", "").rstrip("/") or "/"


def set_session_cookie(request: Request, response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=settings.session_max_days * 86400,
        path=cookie_path(request),
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
    )


def clear_session_cookie(request: Request, response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(
        COOKIE_NAME, path=cookie_path(request), httponly=True,
        secure=settings.cookie_secure, samesite="strict",
    )


async def session_and_user(
    request: Request, db: AsyncSession = Depends(get_db)
) -> tuple[AuthSession, User] | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token or len(token) > 200:
        return None
    session = await db.get(AuthSession, token_hash(token))
    if session is None:
        return None

    settings = get_settings()
    now = utcnow()
    idle_limit = as_utc(session.last_seen_at) + timedelta(hours=settings.session_idle_hours)
    user = await db.get(User, session.user_id)
    if as_utc(session.expires_at) <= now or idle_limit <= now or user is None or not user.active:
        await db.delete(session)
        return None
    if now - as_utc(session.last_seen_at) >= _TOUCH_EVERY:
        session.last_seen_at = now
    return session, user


async def signed_in_user(
    found: tuple[AuthSession, User] | None = Depends(session_and_user),
) -> User:
    """Any signed-in user, including one who still has to change password.
    Only the account endpoints use this directly."""
    if found is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sessione scaduta: accedi di nuovo.")
    return found[1]


async def current_user(user: User = Depends(signed_in_user)) -> User:
    if user.must_change_password:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Devi cambiare la password prima di continuare."
        )
    return user


async def require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Operazione riservata agli amministratori.")
    return user
