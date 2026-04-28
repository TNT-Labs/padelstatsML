"""Celery app + ML task."""
import tempfile
from pathlib import Path

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.core.storage import download_to_path
from app.ml.optimize import configure_for_cpu

settings = get_settings()

# Apply CPU optimizations at worker startup (no-op when CUDA is active)
if settings.ml_device == "cpu":
    configure_for_cpu(settings.torch_num_threads or None)

celery_app = Celery(
    "padel",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # re-queue se worker crash
    worker_prefetch_multiplier=1,  # ML tasks heavy → no prefetch
    task_time_limit=3600,  # 1 ora hard limit
)

# Sync engine per worker (Celery non gioca bene con asyncio)
sync_engine = create_engine(settings.sync_database_url, pool_pre_ping=True)
SyncSession = sessionmaker(sync_engine)


@celery_app.task(name="analyze_match", bind=True, max_retries=2)
def analyze_match_task(self, match_id: str) -> dict:
    """Task principale: analizza video e salva stats."""
    from app.ml.pipeline import AnalysisPipeline, PipelineConfig
    from app.models.match import Match, MatchStats, MatchStatus

    settings = get_settings()

    with SyncSession() as session:
        match = session.get(Match, match_id)
        if not match:
            raise ValueError(f"Match {match_id} not found")

        match.status = MatchStatus.PROCESSING
        match.progress = 0
        session.commit()

    def progress_cb(percent: int, message: str) -> None:
        with SyncSession() as session:
            m = session.get(Match, match_id)
            if m:
                m.progress = percent
                session.commit()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            local_path = str(Path(tmp) / f"{match_id}.mp4")

            # Download video da S3
            with SyncSession() as session:
                m = session.get(Match, match_id)
                s3_key = m.video_s3_key
            download_to_path(s3_key, local_path)

            # Run pipeline
            config = PipelineConfig(
                yolo_weights=settings.yolo_weights,
                tracknet_weights=settings.tracknet_weights if Path(settings.tracknet_weights).exists() else None,
                device=settings.ml_device,
                player_stride=settings.player_stride,
            )
            pipeline = AnalysisPipeline(config)
            result = pipeline.run(local_path, progress_callback=progress_cb)

            # Salva stats
            with SyncSession() as session:
                m = session.get(Match, match_id)
                stats = MatchStats(
                    match_id=m.id,
                    per_player=result["per_player"],
                    heatmaps=result["heatmaps"],
                    rallies=result["rallies"],
                    court_calibration=result["court_calibration"],
                )
                session.add(stats)
                m.status = MatchStatus.COMPLETED
                m.progress = 100
                session.commit()

        return {"match_id": match_id, "status": "completed"}

    except Exception as exc:
        with SyncSession() as session:
            m = session.get(Match, match_id)
            if m:
                m.status = MatchStatus.FAILED
                m.error_message = str(exc)[:1000]
                session.commit()
        # Re-raise per Celery retry / logging
        raise self.retry(exc=exc, countdown=60) from exc
