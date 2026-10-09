"""Account operations shared by the API, start-up and the command line."""
from __future__ import annotations

import logging

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from app.core.security import (
    hash_password,
    normalise_username,
    password_problem,
    temporary_password,
    username_problem,
)
from app.models import AuditEntry, AuthSession, CameraPreset, Match, User, UserRole

logger = logging.getLogger("padel.accounts")


def ensure_admin(session: Session, username: str, password: str) -> User | None:
    """First start-up: create the administrator named in the configuration.

    Does nothing once any user exists, so the password in the .env stops
    mattering after the first sign-in (the user is asked to replace it)."""
    if session.scalar(select(func.count()).select_from(User)):
        return None
    username = normalise_username(username)
    if not username or not password:
        logger.warning(
            "Nessun utente nel database: imposta PADEL_ADMIN_USERNAME e PADEL_ADMIN_PASSWORD "
            "oppure esegui scripts/create_admin.py."
        )
        return None
    problem = username_problem(username) or password_problem(password, username)
    if problem:
        # Logged, not raised: a crash here would put the container in a
        # restart loop and take the page down with it.
        logger.error("Amministratore iniziale non creato: %s Correggi il .env e riavvia.", problem)
        return None
    user = User(
        username=username, password_hash=hash_password(password),
        role=UserRole.ADMIN, must_change_password=True,
    )
    session.add(user)
    session.add(AuditEntry(event="utente_creato", username=username, detail="amministratore iniziale"))
    session.flush()
    logger.info("Creato l'amministratore iniziale «%s».", username)
    return user


def reset_admin(session: Session, username: str) -> str:
    """Command-line recovery: create or reactivate `username` as an
    administrator with a temporary password, which is returned."""
    username = normalise_username(username)
    problem = username_problem(username)
    if problem:
        raise ValueError(problem)
    password = temporary_password()
    while password_problem(password, username):
        password = temporary_password()
    user = session.scalar(select(User).where(User.username == username))
    if user is None:
        user = User(username=username, password_hash="", role=UserRole.ADMIN)
        session.add(user)
    user.password_hash = hash_password(password)
    user.role = UserRole.ADMIN
    user.active = True
    user.must_change_password = True
    user.failed_attempts = 0
    user.locked_until = None
    session.flush()
    session.execute(AuthSession.__table__.delete().where(AuthSession.user_id == user.id))
    session.add(AuditEntry(event="admin_ripristinato", username=username, detail="da riga di comando"))
    return password


def adopt_orphans(session: Session) -> int:
    """Matches and presets from before accounts existed go to the oldest
    administrator. Returns how many rows were assigned."""
    admin_id = session.scalar(
        select(User.id).where(User.role == UserRole.ADMIN).order_by(User.created_at).limit(1)
    )
    if admin_id is None:
        return 0
    matches = session.execute(
        update(Match).where(Match.owner_id.is_(None)).values(owner_id=admin_id)
    ).rowcount
    # A preset name may already exist for the administrator: those keep a
    # recognisable suffix instead of failing the whole start-up.
    session.execute(
        text("UPDATE OR IGNORE camera_presets SET owner_id = :o WHERE owner_id IS NULL"),
        {"o": admin_id},
    )
    session.execute(
        text(
            "UPDATE camera_presets SET owner_id = :o, "
            "name = substr(name, 1, 90) || ' (' || substr(id, 1, 6) || ')' "
            "WHERE owner_id IS NULL"
        ),
        {"o": admin_id},
    )
    if matches:
        logger.info("%d partite precedenti agli account assegnate all'amministratore.", matches)
    return matches or 0
