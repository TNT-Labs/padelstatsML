"""Analysis worker: a single process that drains the `jobs` table.

This replaces Celery + Redis. The deployment runs one analysis at a time on
one machine, so a broker adds two services, ~200 MB of RAM and a second
source of truth for job state, in exchange for concurrency that is never
used. A conditional UPDATE against SQLite does the same job.

Two behaviours are deliberately different from the Celery version:

  * The match is only marked FAILED when the worker gives up for good.
    Previously it was marked FAILED before each retry, so a run that later
    succeeded left the UI showing an error it had already recovered from.
  * A crashed run is recovered by heartbeat, not by broker acknowledgement.
    A job whose heartbeat went stale is requeued on the next worker start.
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

from app.core.config import get_settings
from app.ml.runtime import configure_cpu

logger = logging.getLogger("padel.worker")

_shutdown = False


def _handle_signal(signum, _frame) -> None:
    global _shutdown
    _shutdown = True
    logger.info("Segnale %s ricevuto: arresto dopo il job corrente.", signum)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def run_forever() -> None:
    settings = get_settings()
    threads = configure_cpu(settings.inference_threads)
    logger.info("Worker avviato · %d thread di inferenza", threads)

    # Imported after configure_cpu so the native thread budget is already set.
    from app.core.database import init_schema

    init_schema()
    _requeue_stale_jobs()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while not _shutdown:
        job_id = _claim_next_job()
        if job_id is None:
            time.sleep(settings.worker_poll_seconds)
            continue
        try:
            _run_job(job_id)
        except Exception:                      # noqa: BLE001 - worker must survive
            logger.exception("Errore non gestito nel job %s", job_id)

    logger.info("Worker arrestato.")


# ── Queue handling ───────────────────────────────────────────────────────────

def _claim_next_job() -> str | None:
    """Take the oldest queued job. Single writer, so a plain transaction is
    sufficient; `BEGIN IMMEDIATE` still protects against a second worker."""
    from sqlalchemy import select

    from app.core.database import sync_session
    from app.models import Job, JobState, Match, MatchStatus

    with sync_session() as session:
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        job = session.scalar(
            select(Job).where(Job.state == JobState.QUEUED).order_by(Job.created_at).limit(1)
        )
        if job is None:
            return None

        job.state = JobState.RUNNING
        job.attempts += 1
        job.started_at = _now()
        job.heartbeat_at = _now()

        match = session.get(Match, job.match_id)
        if match is not None:
            match.status = MatchStatus.ANALYZING
            match.progress = 0
            match.progress_message = "In attesa di avvio…"
            match.error_message = None
        return job.id


def _requeue_stale_jobs() -> None:
    """Recover jobs abandoned by a crashed or killed worker."""
    from sqlalchemy import select

    from app.core.database import sync_session
    from app.models import Job, JobState, Match, MatchStatus

    settings = get_settings()
    cutoff = _now() - timedelta(seconds=settings.job_heartbeat_timeout_s)

    with sync_session() as session:
        running = session.scalars(select(Job).where(Job.state == JobState.RUNNING)).all()
        for job in running:
            beat = job.heartbeat_at or job.started_at
            if beat is not None and beat.tzinfo is None:
                beat = beat.replace(tzinfo=timezone.utc)
            if beat is not None and beat > cutoff:
                continue

            match = session.get(Match, job.match_id)
            if job.attempts >= settings.job_max_attempts:
                job.state = JobState.FAILED
                job.error = "Worker interrotto durante l'analisi."
                job.finished_at = _now()
                if match is not None:
                    match.status = MatchStatus.FAILED
                    match.error_message = "Analisi interrotta: worker riavviato."
            else:
                logger.warning("Job %s rimasto appeso: rimesso in coda.", job.id)
                job.state = JobState.QUEUED
                job.started_at = None
                job.heartbeat_at = None
                if match is not None:
                    match.status = MatchStatus.QUEUED
                    match.progress = 0
                    match.progress_message = "Rimesso in coda dopo un riavvio."


# ── Job execution ────────────────────────────────────────────────────────────

def _run_job(job_id: str) -> None:
    from app.core.database import sync_session
    from app.core.storage import save_crop, video_path
    from app.ml.court import calibration_from_dict
    from app.ml.pipeline import AnalysisPipeline, PipelineConfig
    from app.models import Job, JobState, Match, MatchStats, MatchStatus

    settings = get_settings()

    with sync_session() as session:
        job = session.get(Job, job_id)
        match = session.get(Match, job.match_id) if job else None
        if job is None or match is None:
            logger.error("Job %s senza match associato: scartato.", job_id)
            if job is not None:
                job.state = JobState.FAILED
                job.error = "Match inesistente."
                job.finished_at = _now()
            return
        match_id = match.id
        calibration_data = match.calibration
        attempts = job.attempts

    if not calibration_data:
        _fail(job_id, match_id, "Il campo non è stato calibrato.", give_up=True)
        return

    logger.info("Avvio analisi match %s (tentativo %d)", match_id, attempts)
    started = time.monotonic()

    try:
        calibration = calibration_from_dict(calibration_data)
        pipeline = AnalysisPipeline(
            PipelineConfig(
                detector_model=settings.detector_model,
                detector_imgsz=settings.detector_imgsz,
                detector_conf=settings.detector_conf,
                detector_iou=settings.detector_iou,
                inference_threads=settings.inference_threads,
                sample_hz=settings.sample_hz,
                max_analysis_minutes=settings.max_analysis_minutes,
                max_player_speed_ms=settings.max_player_speed_ms,
                track_max_age_s=settings.track_max_age_s,
                rally_speed_threshold_ms=settings.rally_speed_threshold_ms,
                rally_min_duration_s=settings.rally_min_duration_s,
                rally_merge_gap_s=settings.rally_merge_gap_s,
            )
        )
        result = pipeline.run(
            video_path(match_id),
            calibration=calibration,
            progress=_progress_writer(job_id, match_id),
        )

        crop_keys: dict[str, str] = {}
        for player_id, image_bytes in result.pop("player_crops_data", {}).items():
            try:
                crop_keys[str(player_id)] = save_crop(match_id, int(player_id), image_bytes)
            except OSError as exc:
                logger.warning("Anteprima giocatore %s non salvata: %s", player_id, exc)

        with sync_session() as session:
            session.query(MatchStats).filter(MatchStats.match_id == match_id).delete()
            session.add(
                MatchStats(
                    match_id=match_id,
                    per_player=result["per_player"],
                    heatmaps=result["heatmaps"],
                    rallies=result["rallies"],
                    summary=result["summary"],
                    data_quality=result["data_quality"],
                    player_crops=crop_keys or None,
                )
            )
            match = session.get(Match, match_id)
            if match is not None:
                match.status = MatchStatus.COMPLETED
                match.progress = 100
                match.progress_message = "Analisi completata"
                match.error_message = None
            job = session.get(Job, job_id)
            if job is not None:
                job.state = JobState.DONE
                job.finished_at = _now()
                job.error = None

        logger.info(
            "Match %s completato in %.1f min", match_id, (time.monotonic() - started) / 60
        )

    except Exception as exc:                   # noqa: BLE001 - reported to the user
        logger.exception("Analisi fallita per il match %s", match_id)
        give_up = attempts >= settings.job_max_attempts
        _fail(job_id, match_id, str(exc)[:1000], give_up=give_up)


def _progress_writer(job_id: str, match_id: str):
    """Throttled progress persistence: at most one write per 2 points or 15 s.

    The pipeline reports often so the ETA stays fresh, but each write is an
    SQLite transaction competing with the API's reads.
    """
    from app.core.database import sync_session
    from app.models import Job, Match

    state = {"percent": -100, "at": 0.0}

    def write(percent: int, message: str) -> None:
        now = time.monotonic()
        if percent < 100 and percent - state["percent"] < 2 and now - state["at"] < 15.0:
            return
        state["percent"], state["at"] = percent, now
        try:
            with sync_session() as session:
                match = session.get(Match, match_id)
                if match is not None:
                    match.progress = max(0, min(100, percent))
                    match.progress_message = message[:200]
                job = session.get(Job, job_id)
                if job is not None:
                    job.heartbeat_at = _now()
        except Exception:                      # noqa: BLE001 - never kill a run
            logger.warning("Scrittura progresso fallita", exc_info=True)

    return write


def _fail(job_id: str, match_id: str, message: str, give_up: bool) -> None:
    """Record a failed attempt.

    While retries remain the match goes back to QUEUED, not FAILED: showing an
    error for a run that is about to be retried is what made the previous
    version's status display untrustworthy.
    """
    from app.core.database import sync_session
    from app.models import Job, JobState, Match, MatchStatus

    with sync_session() as session:
        job = session.get(Job, job_id)
        match = session.get(Match, match_id)

        if job is not None:
            job.error = message
            job.state = JobState.FAILED if give_up else JobState.QUEUED
            job.finished_at = _now() if give_up else None
            job.heartbeat_at = None
            if not give_up:
                job.started_at = None

        if match is not None:
            if give_up:
                match.status = MatchStatus.FAILED
                match.error_message = message
                match.progress_message = "Analisi fallita"
            else:
                match.status = MatchStatus.QUEUED
                match.progress = 0
                match.progress_message = "Nuovo tentativo in corso…"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s · %(message)s",
        stream=sys.stdout,
    )
    run_forever()


if __name__ == "__main__":
    main()
