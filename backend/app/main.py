"""FastAPI entrypoint.

The API also serves the built web UI, so the Pi runs two containers (api and
worker) instead of the previous seven. There is no nginx, no MinIO console,
no Flower and no broker.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import matches, presets
from app.core.config import get_settings

logger = logging.getLogger("padel.api")

# Populated by the Docker build; absent during local backend-only development.
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from app.core.database import init_schema

    settings = get_settings()
    settings.ensure_dirs()
    init_schema()
    logger.info("API pronta · dati in %s", settings.data_dir)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Padel Stats", version="1.0.0", lifespan=lifespan)

    # Same-origin by default. A list is only needed when the Vite dev server
    # runs on another port.
    origins = settings.cors_origin_list
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(matches.router)
    app.include_router(presets.router)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        """Readiness of the three things that can actually be broken here:
        the database, the data directory, and the detector model."""
        from sqlalchemy import text

        from app.core.database import async_engine

        checks: dict[str, str] = {}
        healthy = True

        try:
            async with async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:                # noqa: BLE001
            checks["database"] = f"error: {exc}"
            healthy = False

        try:
            free_gb = __import__("shutil").disk_usage(settings.data_path).free / 1e9
            checks["storage"] = f"ok · {free_gb:.1f} GB liberi"
            if free_gb < 2.0:
                checks["storage"] += " (spazio quasi esaurito)"
                healthy = False
        except OSError as exc:
            checks["storage"] = f"error: {exc}"
            healthy = False

        model = Path(settings.detector_model)
        if model.exists():
            checks["detector"] = f"ok · {model.name}"
        else:
            checks["detector"] = f"mancante: {model}"
            healthy = False

        return JSONResponse(
            {"status": "ok" if healthy else "degraded", "checks": checks},
            status_code=200 if healthy else 503,
        )

    _mount_web_ui(app)
    return app


def _mount_web_ui(app: FastAPI) -> None:
    """Serve the built SPA, falling back to index.html for client routes."""
    if not (WEB_ROOT / "index.html").exists():
        logger.info("UI non presente in %s — modalità solo API.", WEB_ROOT)
        return

    assets = WEB_ROOT / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        # An unmatched /api path is a client error, not a client-side route:
        # returning index.html for it would turn every API typo into a
        # confusing "unexpected token <" in the browser console.
        if full_path.startswith("api/"):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Endpoint inesistente.")

        candidate = (WEB_ROOT / full_path).resolve()
        if (
            full_path
            and WEB_ROOT.resolve() in candidate.parents
            and candidate.is_file()
        ):
            return FileResponse(str(candidate))
        return FileResponse(str(WEB_ROOT / "index.html"))


app = create_app()
