"""Request checks and response headers that apply to every route."""
from __future__ import annotations

import json
from urllib.parse import urlsplit

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

# No inline scripts anywhere in the UI; inline styles are used by React.
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
    "font-src 'self' data:; object-src 'none'; base-uri 'self'; form-action 'self'; "
    "frame-ancestors 'none'"
)
_HEADERS = [
    (b"content-security-policy", _CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]


def _host_only(value: str) -> str:
    """Host name without port; a proxy may drop the port from Host."""
    try:
        return (urlsplit("//" + value.split(",")[0].strip()).hostname or "").lower()
    except ValueError:
        return ""


class SecurityMiddleware:
    """Pure ASGI, so multi-gigabyte uploads and video ranges stream through
    untouched (BaseHTTPMiddleware would sit in the middle of every body).

    CSRF defence for every request that changes data, on top of the
    SameSite=Strict cookie:
      * a custom header (X-Padel: 1) that another site cannot add without a
        CORS preflight, which is never granted;
      * when the browser sends Origin, its host must be this one.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        root = scope.get("root_path", "")
        if root and path.startswith(root):
            path = path[len(root):]
        is_api = path.startswith("/api/")

        if is_api and scope["method"] not in _SAFE_METHODS:
            problem = self._csrf_problem(scope)
            if problem:
                await self._reject(send, problem)
                return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in _HEADERS if k not in present]
                if is_api and b"cache-control" not in present:
                    # Personal data: never in a shared cache, always revalidated.
                    headers.append((b"cache-control", b"private, no-cache"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)

    @staticmethod
    def _csrf_problem(scope) -> str | None:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        if headers.get("x-padel") != "1":
            return "Richiesta non valida."
        origin = headers.get("origin")
        if origin and origin != "null":
            expected = _host_only(headers.get("x-forwarded-host") or headers.get("host", ""))
            if (urlsplit(origin).hostname or "").lower() != expected:
                return "Origine della richiesta non consentita."
        elif origin == "null":
            return "Origine della richiesta non consentita."
        return None

    @staticmethod
    async def _reject(send, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send({
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                *_HEADERS,
            ],
        })
        await send({"type": "http.response.body", "body": body})
