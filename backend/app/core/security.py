"""Passwords, session tokens and the login rate limit.

Standard library only: scrypt is memory-hard and ships with Python, so no
native dependency (bcrypt, argon2) has to be built for the Pi.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import time
from datetime import datetime, timezone

from fastapi import Request

from app.core.config import get_settings

# ~16 MB and ~50 ms per hash on a Pi 5: slow for a guesser, fine for a login.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P, _DKLEN = 2**14, 8, 1, 64
_SCRYPT_MAXMEM = 64 * 1024 * 1024

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,39}$")
PASSWORD_MIN = 10
PASSWORD_MAX = 200


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        dklen=_DKLEN, maxmem=_SCRYPT_MAXMEM,
    )
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT_N, _SCRYPT_R, _SCRYPT_P,
        base64.b64encode(salt).decode(), base64.b64encode(digest).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest_b64)
        digest = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt_b64), n=int(n), r=int(r),
            p=int(p), dklen=len(expected), maxmem=_SCRYPT_MAXMEM,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


# Verified against when the username does not exist, so that a wrong
# username costs as much time as a wrong password and cannot be told apart.
DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def normalise_username(value: str) -> str:
    return (value or "").strip().lower()


def username_problem(username: str) -> str | None:
    if not USERNAME_RE.match(username):
        return ("Nome utente non valido: da 3 a 40 caratteri tra lettere minuscole, "
                "cifre, punto, trattino e trattino basso.")
    return None


def password_problem(password: str, username: str) -> str | None:
    """None when acceptable, otherwise the reason in words the user can act on."""
    if len(password) < PASSWORD_MIN:
        return f"La password deve avere almeno {PASSWORD_MIN} caratteri."
    if len(password) > PASSWORD_MAX:
        return f"La password può avere al massimo {PASSWORD_MAX} caratteri."
    if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        return "La password deve contenere lettere e cifre."
    if username and username.lower() in password.lower():
        return "La password non può contenere il nome utente."
    return None


def temporary_password() -> str:
    """Readable (no 0/O, 1/l/I) and always valid for password_problem."""
    letters = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"
    digits = "23456789"
    body = [secrets.choice(letters + digits) for _ in range(10)]
    body[secrets.randbelow(10)] = secrets.choice(digits)
    body.insert(secrets.randbelow(11), secrets.choice(letters))
    return "".join(body)


def new_session_token() -> tuple[str, str]:
    """(value for the cookie, hash for the database)."""
    token = secrets.token_urlsafe(32)
    return token, token_hash(token)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime | None) -> datetime | None:
    """SQLite hands datetimes back naive; they were stored in UTC."""
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def client_ip(request: Request) -> str:
    """The visitor's address. Behind the tunnel every request comes from
    cloudflared, so the real one is in CF-Connecting-IP — trusted only when
    configured, because anyone reaching the API directly could forge it."""
    if get_settings().trust_cloudflare:
        forwarded = request.headers.get("cf-connecting-ip", "").strip()
        if forwarded:
            return forwarded[:64]
    return (request.client.host if request.client else "sconosciuto")[:64]


class FailureLimiter:
    """Fixed-window counter of failed logins per address, in memory: the API
    runs as a single process, and a restart resetting it is harmless next to
    the per-account lock, which is in the database."""

    def __init__(self, max_failures: int, window_s: float) -> None:
        self.max_failures = max_failures
        self.window_s = window_s
        self._hits: dict[str, tuple[int, float]] = {}

    def blocked(self, key: str) -> bool:
        count, reset = self._hits.get(key, (0, 0.0))
        return reset > time.monotonic() and count >= self.max_failures

    def hit(self, key: str) -> None:
        now = time.monotonic()
        if len(self._hits) > 10_000:
            self._hits = {k: v for k, v in self._hits.items() if v[1] > now}
        count, reset = self._hits.get(key, (0, 0.0))
        if reset <= now:
            count, reset = 0, now + self.window_s
        self._hits[key] = (count + 1, reset)

    def reset(self) -> None:
        self._hits.clear()


login_limiter = FailureLimiter(max_failures=20, window_s=15 * 60)

