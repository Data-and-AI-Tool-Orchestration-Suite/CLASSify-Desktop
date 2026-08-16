"""Repository functions for database CRUD operations.

All functions take a SQLAlchemy ``Session`` as the first argument so they
can be used from both FastAPI dependencies and background tasks.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, func, or_, select
from sqlalchemy.orm import Session

from classify_api.orm.models import Action, Job, Report, Result, Setting

# ── Report ──


def create_report(
    db: Session,
    *,
    filename: str,
    original_filename: str | None = None,
    status: str = "Preview",
    column_changes: dict[str, Any] | None = None,
) -> Report:
    report = Report(
        filename=filename,
        original_filename=original_filename,
        status=status,
        column_changes=column_changes,
    )
    db.add(report)
    db.flush()
    return report


def get_report(db: Session, report_uuid: str) -> Report | None:
    return db.get(Report, report_uuid)


def get_report_by_filename(db: Session, filename: str) -> Report | None:
    return db.scalar(select(Report).where(Report.filename == filename))


def is_report_present(db: Session, filename: str) -> bool:
    result = db.scalar(select(func.count()).select_from(Report).where(Report.filename == filename))
    return (result or 0) > 0


def list_reports(
    db: Session,
    *,
    start: int = 0,
    length: int = 100,
    search: str = "",
    order_by: str = "created_at",
    order_dir: str = "desc",
) -> list[Report]:
    stmt = select(Report)
    if search:
        pattern = f"%{search}%"
        stmt = stmt.where(or_(Report.filename.ilike(pattern), Report.status.ilike(pattern)))

    col_map = {
        "filename": Report.filename,
        "status": Report.status,
        "created_at": Report.created_at,
        "comments": Report.comments,
    }
    col = col_map.get(order_by, Report.created_at)
    stmt = stmt.order_by(desc(col) if order_dir == "desc" else col)
    stmt = stmt.offset(start).limit(length)
    return list(db.scalars(stmt))


def count_reports(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(Report)) or 0


def count_reports_filtered(db: Session, search: str = "") -> int:
    stmt = select(func.count()).select_from(Report)
    if search:
        pattern = f"%{search}%"
        stmt = stmt.where(or_(Report.filename.ilike(pattern), Report.status.ilike(pattern)))
    return db.scalar(stmt) or 0


def update_report_status(db: Session, report_uuid: str, status: str) -> None:
    report = db.get(Report, report_uuid)
    if report:
        report.status = status
        db.flush()


def update_report_comments(db: Session, report_uuid: str, comments: str) -> None:
    report = db.get(Report, report_uuid)
    if report:
        report.comments = comments
        db.flush()


def update_report_column_changes(db: Session, report_uuid: str, changes: dict[str, Any]) -> None:
    report = db.get(Report, report_uuid)
    if report:
        report.column_changes = changes
        db.flush()


def update_report_job_id(db: Session, report_uuid: str, job_id: str) -> None:
    report = db.get(Report, report_uuid)
    if report:
        report.job_id = job_id
        db.flush()


def delete_report(db: Session, report_uuid: str) -> bool:
    report = db.get(Report, report_uuid)
    if report:
        db.delete(report)
        db.flush()
        return True
    return False


# ── Job ──


def create_job(db: Session, *, report_uuid: str, args: dict[str, Any] | None = None) -> Job:
    job = Job(report_uuid=report_uuid, args=args)
    db.add(job)
    db.flush()
    return job


def get_job(db: Session, job_id: str) -> Job | None:
    return db.get(Job, job_id)


def get_job_by_report(db: Session, report_uuid: str) -> Job | None:
    return db.scalar(
        select(Job).where(Job.report_uuid == report_uuid).order_by(desc(Job.created_at))
    )


def get_previous_job_by_report(db: Session, report_uuid: str, exclude_job_id: str) -> Job | None:
    """Get the most recent job for a report, excluding the given job id."""
    return db.scalar(
        select(Job)
        .where(Job.report_uuid == report_uuid, Job.id != exclude_job_id)
        .order_by(desc(Job.created_at))
    )


def get_next_queued_job(db: Session) -> Job | None:
    return db.scalar(select(Job).where(Job.state == "queued").order_by(Job.created_at).limit(1))


def get_running_job(db: Session) -> Job | None:
    return db.scalar(select(Job).where(Job.state.in_(["running", "cancelling"])).limit(1))


def list_jobs(db: Session) -> list[Job]:
    return list(db.scalars(select(Job).order_by(desc(Job.created_at))))


def update_job_state(db: Session, job_id: str, state: str, error: str | None = None) -> None:
    job = db.get(Job, job_id)
    if job:
        job.state = state
        if error is not None:
            job.error = error
        now = datetime.now(UTC)
        if state == "running" and job.started_at is None:
            job.started_at = now
        if state in ("succeeded", "failed"):
            job.finished_at = now
        db.flush()


def update_job_progress(
    db: Session, job_id: str, completed: int, total: int, message: str | None = None
) -> None:
    job = db.get(Job, job_id)
    if job:
        job.progress = completed
        job.progress_total = total
        if message:
            job.progress_message = message
        db.flush()


# ── Result ──


def create_result(db: Session, *, report_uuid: str) -> Result:
    result = Result(report_uuid=report_uuid)
    db.add(result)
    db.flush()
    return result


def get_result_by_report(db: Session, report_uuid: str) -> Result | None:
    return db.scalar(
        select(Result)
        .where(Result.report_uuid == report_uuid)
        .order_by(desc(Result.date_processed))
    )


# ── Setting ──


def get_setting(db: Session, key: str, default: str | None = None) -> str | None:
    s = db.get(Setting, key)
    return s.value if s else default


def set_setting(db: Session, key: str, value: str) -> None:
    s = db.get(Setting, key)
    if s:
        s.value = value
    else:
        db.add(Setting(key=key, value=value))
    db.flush()


def get_all_settings(db: Session) -> dict[str, str]:
    rows = db.scalars(select(Setting))
    return {row.key: row.value for row in rows}


def seed_default_settings(db: Session) -> None:
    """Insert default settings if they don't exist."""
    import os

    defaults = {
        "n_jobs": str(os.cpu_count() or 1),
        "max_upload_mb": "500",
        "theme": "default",
        "encryption_enabled": "false",
    }
    for key, value in defaults.items():
        if db.get(Setting, key) is None:
            db.add(Setting(key=key, value=value))
    db.flush()


# ── Action ──


def create_action(
    db: Session,
    *,
    action: str,
    report_uuid: str | None = None,
    addl_info: dict[str, Any] | None = None,
) -> Action:
    a = Action(action=action, report_uuid=report_uuid, addl_info=addl_info)
    db.add(a)
    db.flush()
    return a


def list_recent_actions(db: Session, limit: int = 50) -> list[Action]:
    return list(db.scalars(select(Action).order_by(desc(Action.time_performed)).limit(limit)))


# ── Utility ──


def serialize_report(report: Report) -> dict[str, Any]:
    """Convert a Report ORM object to a JSON-serializable dict."""
    return {
        "uuid": report.uuid,
        "filename": report.filename,
        "original_filename": report.original_filename,
        "status": report.status,
        "job_id": report.job_id,
        "column_changes": report.column_changes,
        "comments": report.comments,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


def serialize_job(job: Job) -> dict[str, Any]:
    """Convert a Job ORM object to a JSON-serializable dict."""
    return {
        "id": job.id,
        "report_uuid": job.report_uuid,
        "state": job.state,
        "args": job.args,
        "progress": job.progress,
        "progress_total": job.progress_total,
        "progress_message": job.progress_message,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }
