"""Job queue — SQLite-backed FIFO queue for training jobs.

Replaces ClearML's queue management.  One job runs at a time (CPU-bound).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from classify_api import repositories as repo
from classify_api.orm.models import Job


def enqueue(db: Session, report_uuid: str, args: dict[str, Any]) -> Job:
    """Add a training job to the queue."""
    job = repo.create_job(db, report_uuid=report_uuid, args=args)
    db.commit()
    return job


def get_next_pending(db: Session) -> Job | None:
    """Get the next queued job (FIFO)."""
    return repo.get_next_queued_job(db)


def get_running(db: Session) -> Job | None:
    """Get the currently running job (if any)."""
    return repo.get_running_job(db)


def set_state(db: Session, job_id: str, state: str, error: str | None = None) -> None:
    """Update a job's state."""
    repo.update_job_state(db, job_id, state, error)
    db.commit()


def update_progress(
    db: Session, job_id: str, completed: int, total: int, message: str | None = None
) -> None:
    """Update a job's progress."""
    repo.update_job_progress(db, job_id, completed, total, message)
    db.commit()


def get_job(db: Session, job_id: str) -> Job | None:
    """Get a job by ID."""
    return repo.get_job(db, job_id)


def list_jobs(db: Session) -> list[Job]:
    """List all jobs, newest first."""
    return repo.list_jobs(db)


def mark_stale_jobs_failed(db: Session) -> int:
    """On startup, mark any running/cancelling jobs as failed (interrupted).

    Returns the count of stale jobs recovered.
    """
    count = 0
    jobs = repo.list_jobs(db)
    for job in jobs:
        if job.state in ("running", "cancelling"):
            repo.update_job_state(db, job.id, "failed", error="Interrupted by app restart")
            count += 1
    if count:
        db.commit()
    return count
