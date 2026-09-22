"""Queue semantics of the worker that replaced Celery.

The status transitions matter as much as the analysis: the previous version
marked a match FAILED before every retry, so the UI reported an error for runs
that went on to succeed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.core.database import init_schema, sync_session
from app.models import Job, JobState, Match, MatchStatus
from app.worker.runner import _claim_next_job, _fail, _requeue_stale_jobs


@pytest.fixture(autouse=True)
def clean_queue():
    """The whole suite shares one SQLite file, and these tests assert on
    *which* job gets claimed — so the queue has to start empty."""
    init_schema()
    with sync_session() as session:
        session.query(Job).delete()
        session.query(Match).delete()
    yield


def _queued_match(title: str = "Job test") -> tuple[str, str]:
    with sync_session() as session:
        match = Match(
            title=title,
            video_filename="x.mp4",
            status=MatchStatus.QUEUED,
            calibration={"corners_px": [[0, 0]] * 4, "frame_size": [640, 360], "source": "manual"},
        )
        session.add(match)
        session.flush()
        job = Job(match_id=match.id)
        session.add(job)
        session.flush()
        return match.id, job.id


def _reload(match_id: str, job_id: str) -> tuple[Match, Job]:
    with sync_session() as session:
        return session.get(Match, match_id), session.get(Job, job_id)


def test_claiming_a_job_marks_the_match_as_analyzing():
    match_id, job_id = _queued_match()
    assert _claim_next_job() == job_id

    match, job = _reload(match_id, job_id)
    assert job.state == JobState.RUNNING
    assert job.attempts == 1
    assert job.heartbeat_at is not None
    assert match.status == MatchStatus.ANALYZING


def test_an_empty_queue_claims_nothing():
    assert _claim_next_job() is None


def test_jobs_are_claimed_oldest_first():
    first_match, first_job = _queued_match("primo")
    _, second_job = _queued_match("secondo")
    assert _claim_next_job() == first_job
    assert _claim_next_job() == second_job


def test_a_retryable_failure_returns_the_match_to_the_queue():
    match_id, job_id = _queued_match()
    _claim_next_job()
    _fail(job_id, match_id, "errore transitorio", give_up=False)

    match, job = _reload(match_id, job_id)
    assert job.state == JobState.QUEUED
    assert job.error == "errore transitorio"
    # The user must not be shown a failure for a run that is being retried.
    assert match.status == MatchStatus.QUEUED
    assert match.error_message is None


def test_giving_up_marks_the_match_failed_with_the_reason():
    match_id, job_id = _queued_match()
    _claim_next_job()
    _fail(job_id, match_id, "campo non calibrato", give_up=True)

    match, job = _reload(match_id, job_id)
    assert job.state == JobState.FAILED
    assert job.finished_at is not None
    assert match.status == MatchStatus.FAILED
    assert match.error_message == "campo non calibrato"


def test_a_job_abandoned_by_a_crashed_worker_is_requeued():
    match_id, job_id = _queued_match()
    _claim_next_job()
    with sync_session() as session:
        session.get(Job, job_id).heartbeat_at = datetime.now(timezone.utc) - timedelta(hours=2)

    _requeue_stale_jobs()
    match, job = _reload(match_id, job_id)
    assert job.state == JobState.QUEUED
    assert match.status == MatchStatus.QUEUED


def test_a_running_job_is_requeued_even_with_a_fresh_heartbeat():
    """This deployment runs one worker, so a job still marked RUNNING when it
    starts belongs to the process that just died.

    Waiting for the heartbeat to age out stranded the job for good:
    `_claim_next_job` only picks up QUEUED rows, so the match stayed in
    ANALYZING forever. The earlier version of this test asserted the buggy
    behaviour."""
    match_id, job_id = _queued_match()
    _claim_next_job()          # heartbeat scritto adesso
    _requeue_stale_jobs()

    match, job = _reload(match_id, job_id)
    assert job.state == JobState.QUEUED
    assert job.started_at is None and job.heartbeat_at is None
    assert match.status == MatchStatus.QUEUED


def test_an_abandoned_job_out_of_attempts_is_failed_not_looped():
    """Il recupero non deve diventare un ciclo infinito: esaurititi i
    tentativi il job fallisce e l'utente vede l'errore."""
    match_id, job_id = _queued_match()
    with sync_session() as session:
        job = session.get(Job, job_id)
        job.state = JobState.RUNNING
        job.attempts = 99
        job.heartbeat_at = datetime.now(timezone.utc) - timedelta(hours=2)

    _requeue_stale_jobs()
    match, job = _reload(match_id, job_id)
    assert job.state == JobState.FAILED
    assert match.status == MatchStatus.FAILED


def test_deleting_a_match_removes_its_jobs():
    match_id, job_id = _queued_match()
    with sync_session() as session:
        session.delete(session.get(Match, match_id))
    with sync_session() as session:
        assert session.get(Job, job_id) is None
        assert session.scalars(select(Job).where(Job.match_id == match_id)).all() == []
