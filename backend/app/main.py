"""FastAPI entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import matches
from app.core.config import get_settings
from app.core.storage import ensure_bucket


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_bucket()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Padel Stats API", version="0.2.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(matches.router)

    @app.get("/health")
    async def health():
        """Deep health check: verify DB, Redis, and S3 are reachable."""
        status: dict[str, str] = {}
        ok = True

        # PostgreSQL
        try:
            from app.core.database import engine
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            status["db"] = "ok"
        except Exception as exc:
            status["db"] = f"error: {exc}"
            ok = False

        # Redis (via Celery broker URL)
        try:
            import redis as _redis
            r = _redis.from_url(settings.redis_url, socket_connect_timeout=2)
            r.ping()
            r.close()
            status["redis"] = "ok"
        except Exception as exc:
            status["redis"] = f"error: {exc}"
            ok = False

        # S3 / MinIO
        try:
            from app.core.storage import get_s3_client
            get_s3_client().head_bucket(Bucket=settings.s3_bucket_videos)
            status["s3"] = "ok"
        except Exception as exc:
            status["s3"] = f"error: {exc}"
            ok = False

        from fastapi.responses import JSONResponse
        return JSONResponse(
            content={"status": "ok" if ok else "degraded", **status},
            status_code=200 if ok else 503,
        )

    return app


app = create_app()
