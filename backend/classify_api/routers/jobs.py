"""Job endpoints — start training, check status, SSE progress, cancel."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from classify_api import repositories as repo
from classify_api.db import get_session
from classify_api.schemas.jobs import JobListResponse, JobResponse, TrainRequest
from classify_api.settings import Settings, get_settings
from ml.options import get_options
from runner.cancellation import set_cancel_flag
from runner.queue import enqueue, get_job, get_running, list_jobs, mark_stale_jobs_failed, set_state
from storage.factory import get_storage

router = APIRouter()


def _serialize_job(job: Any) -> JobResponse:
    return JobResponse(
        id=job.id,
        report_uuid=job.report_uuid,
        state=job.state,
        args=job.args,
        progress=job.progress,
        progress_total=job.progress_total,
        progress_message=job.progress_message,
        error=job.error,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
    )


@router.post("", response_model=JobResponse)
def start_training(
    request: TrainRequest,
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> JobResponse:
    """Start a training job.  Parses options and enqueues."""
    report = repo.get_report(db, request.report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Dataset {request.report_id} not found")

    if report.status == "Processing":
        raise HTTPException(status_code=409, detail="A job is already running for this dataset")

    # Check if another job is already running (one at a time)
    running = get_running(db)
    if running:
        raise HTTPException(
            status_code=409,
            detail=f"Another job is already running (job {running.id}). Wait for it to finish.",
        )

    # Parse options into args dict
    args: dict[str, Any] = {}
    for opt in request.options:
        name = opt.get("name", "")
        value = opt.get("value", "")
        if name == "train_group":
            if name in args and isinstance(args[name], list):
                args[name].append(value)
            else:
                args[name] = [value]
        else:
            if isinstance(value, str):
                if value == "True":
                    args[name] = True
                elif value == "False":
                    args[name] = False
                else:
                    try:
                        if "." in value or "e" in value.lower():
                            args[name] = float(value)
                        else:
                            args[name] = int(value)
                    except ValueError:
                        args[name] = value
            else:
                args[name] = value

    # Flatten nested train_group (defensive)
    if "train_group" in args:
        has_sublists = any(isinstance(i, list) for i in args["train_group"])
        if has_sublists:
            flattened: list[Any] = []
            for i in args["train_group"]:
                if isinstance(i, list):
                    flattened.extend(i)
                else:
                    flattened.append(i)
            args["train_group"] = flattened

    # Set runner-populated fields
    args["report_uuid"] = request.report_id
    args["disable_model_save"] = False
    args["n_jobs"] = settings.n_jobs
    args["max_features"] = ["auto", "sqrt"]
    args["min_samples_split"] = [2, 5, 10]
    args["min_samples_leaf"] = [1, 2, 4]
    args["bootstrap"] = [True, False]

    # Enqueue
    job = enqueue(db, request.report_id, args)
    repo.update_report_job_id(db, request.report_id, job.id)
    db.commit()

    return _serialize_job(job)


@router.get("", response_model=JobListResponse)
def get_jobs(db: Session = Depends(get_session)) -> JobListResponse:
    """List all jobs."""
    jobs = list_jobs(db)
    return JobListResponse(jobs=[_serialize_job(j) for j in jobs])


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(
    job_id: str,
    db: Session = Depends(get_session),
) -> JobResponse:
    """Get a single job's status."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _serialize_job(job)


@router.post("/{job_id}/cancel")
def cancel_job(
    job_id: str,
    db: Session = Depends(get_session),
) -> dict[str, str]:
    """Cancel a running or queued job."""
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.state == "queued":
        # Just remove from queue
        set_state(db, job_id, "failed", error="Cancelled by user")
        return {"status": "cancelled", "message": "Queued job cancelled"}

    if job.state in ("running", "cancelling"):
        storage = get_storage()
        set_cancel_flag(storage, job.report_uuid)
        set_state(db, job_id, "cancelling")
        # The manager will detect the cancel flag and kill the subprocess
        return {"status": "cancelling", "message": "Cancellation requested"}

    return {"status": job.state, "message": f"Job is already {job.state}"}


@router.get("/{job_id}/events")
async def job_events(
    job_id: str,
    db: Session = Depends(get_session),
) -> StreamingResponse:
    """SSE stream of job progress updates.

    The frontend subscribes to this endpoint to get live progress.
    """
    job = get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    storage = get_storage()
    report_id = job.report_uuid

    async def event_stream() -> Any:
        last_progress = None
        while True:
            # Read current job state
            db_factory = __import__(
                "classify_api.db", fromlist=["get_session_factory"]
            ).get_session_factory()
            session = db_factory()
            try:
                current_job = repo.get_job(session, job_id)
                if current_job is None:
                    break

                state = current_job.state

                # Read progress from storage
                from runner.progress import read_progress

                progress = read_progress(storage, report_id)

                # Emit event if progress changed
                current = (
                    (progress.completed, progress.total, progress.message) if progress else None
                )
                if current != last_progress or state in ("succeeded", "failed"):
                    last_progress = current
                    event_data = {
                        "state": state,
                        "progress": progress.completed if progress else current_job.progress,
                        "total": progress.total if progress else current_job.progress_total,
                        "message": progress.message if progress else current_job.progress_message,
                        "error": current_job.error,
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"

                # Stop if job is done
                if state in ("succeeded", "failed"):
                    break

            finally:
                session.close()

            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/ml-options/supervised")
def ml_options_supervised() -> dict[str, Any]:
    """Get supervised ML options for the Prepare page."""
    return get_options(supervised=True)


@router.get("/ml-options/unsupervised")
def ml_options_unsupervised() -> dict[str, Any]:
    """Get unsupervised ML options for the Prepare page."""
    return get_options(supervised=False)


@router.post("/recover")
def recover_stale_jobs(db: Session = Depends(get_session)) -> dict[str, int]:
    """Mark stale running jobs as failed (called on app startup)."""
    count = mark_stale_jobs_failed(db)
    return {"recovered": count}
