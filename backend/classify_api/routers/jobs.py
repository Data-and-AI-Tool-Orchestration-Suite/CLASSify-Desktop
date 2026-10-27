"""Job endpoints — start training, check status, SSE progress, cancel."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
from typing import Any

import pandas as pd
import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from classify_api import repositories as repo
from classify_api.db import get_session
from classify_api.schemas.jobs import JobListResponse, JobResponse, TrainRequest
from classify_api.settings import Settings, get_settings
from ml.backends import is_available
from ml.column_types import detect_encoding
from ml.options import get_forecasting_options, get_options
from runner.cancellation import clear_cancel_flag, set_cancel_flag
from runner.manager import get_manager
from runner.progress import append_log
from runner.queue import enqueue, get_job, list_jobs, mark_stale_jobs_failed, set_state
from storage.factory import get_storage

router = APIRouter()
log = structlog.get_logger()

FORECAST_GENERATION_MODE = "forecast-generation"


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
        created_at=repo.serialize_datetime(job.created_at),
        started_at=repo.serialize_datetime(job.started_at),
        finished_at=repo.serialize_datetime(job.finished_at),
    )


# Training is CPU-bound and jobs run one at a time; cap how many may pile
# up in the queue so runs can't queue indefinitely behind each other.
MAX_QUEUED_JOBS = 10


def _parse_options(request: TrainRequest) -> dict[str, Any]:
    """Parse the flat name/value option list into a typed args dict."""
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
                lowered = value.lower()
                if lowered == "true":
                    args[name] = True
                elif lowered == "false":
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
    return args


def _guard_queue_capacity(db: Session, report_id: str, report: Any) -> None:
    if report.status == "Processing":
        raise HTTPException(status_code=409, detail="A job is already running for this dataset")

    active = repo.get_active_job_for_report(db, report_id)
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail="A job is already queued or running for this dataset "
            f"(job {active.id[:8]}). Wait for it to finish.",
        )

    queued = repo.count_queued_jobs(db)
    if queued >= MAX_QUEUED_JOBS:
        raise HTTPException(
            status_code=409,
            detail=f"The job queue is full ({queued} jobs waiting, limit {MAX_QUEUED_JOBS}). "
            "Wait for one to finish before starting another.",
        )


def _enqueue_job(db: Session, report_id: str, args: dict[str, Any], report: Any) -> JobResponse:
    args["report_uuid"] = report_id
    args["prev_report_status"] = report.status

    job = enqueue(db, report_id, args)
    repo.update_report_job_id(db, report_id, job.id)
    repo.update_report_status(db, report_id, "Queued")
    db.commit()

    clear_cancel_flag(get_storage(), report_id)
    return _serialize_job(job)


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

    _guard_queue_capacity(db, request.report_id, report)

    args = _parse_options(request)

    if request.mode == "forecasting":
        return _start_forecasting_job(db, request, report, args, settings)

    # Training runs sequentially on purpose (CPU-bound, limited compute on
    # smaller machines) — new jobs wait in the FIFO queue for the current one.

    # Supervised training needs a target column; unsupervised does not.
    supervised = args.get("supervised", True)
    if supervised and not args.get("class_column"):
        log.warning(
            "job_start_rejected",
            report_id=request.report_id,
            supervised=supervised,
            provided_option_names=[o.get("name") for o in request.options],
        )
        raise HTTPException(
            status_code=400,
            detail="A class (target) column is required for supervised training",
        )
    if not supervised and not args.get("class_column"):
        args.pop("class_column", None)

    # Set runner-populated fields
    args["disable_model_save"] = False
    args["n_jobs"] = settings.n_jobs
    args["max_features"] = ["sqrt", "log2"]
    args["min_samples_split"] = [2, 5, 10]
    args["min_samples_leaf"] = [1, 2, 4]
    args["bootstrap"] = [True, False]

    return _enqueue_job(db, request.report_id, args, report)


def _ts_role_args(report: Any) -> dict[str, Any]:
    """Extract forecasting column roles from a report's saved ts_roles."""
    roles: dict[str, Any] = {
        "time_component": "",
        "target": [],
        "grouping": [],
        "past_covariates": [],
        "future_covariates": [],
        "static_covariates": [],
    }
    column_changes = report.column_changes or {}
    for role_entry in column_changes.get("ts_roles") or []:
        column, role = role_entry.get("column", ""), role_entry.get("role", "")
        if role == "time-component":
            roles["time_component"] = column
        elif role == "target":
            roles["target"].append(column)
        elif role == "grouping":
            roles["grouping"].append(column)
        elif role == "past-covariate":
            roles["past_covariates"].append(column)
        elif role == "future-covariate":
            roles["future_covariates"].append(column)
        elif role == "static-covariate":
            roles["static_covariates"].append(column)
    return roles


def _start_forecasting_job(
    db: Session,
    request: TrainRequest,
    report: Any,
    args: dict[str, Any],
    settings: Settings,
) -> JobResponse:
    """Build forecasting args from the dataset's saved time-series roles."""
    if not is_available("darts"):
        raise HTTPException(
            status_code=400,
            detail="The Forecasting add-on is not installed. "
            "Install it via the Add-ons page to train forecasting models.",
        )

    ts_roles = (report.column_changes or {}).get("ts_roles") or []
    if not ts_roles:
        raise HTTPException(
            status_code=400,
            detail="This dataset has no time-series configuration. "
            "Assign the time component, target and covariate columns first.",
        )

    args.update(_ts_role_args(report))

    if not args["time_component"] or not args["target"]:
        raise HTTPException(
            status_code=400,
            detail="A time component and at least one target column are required for forecasting",
        )

    if args.get("retrain"):
        storage = get_storage()
        if not storage.exists(f"{request.report_id}/retrain_file"):
            raise HTTPException(
                status_code=400,
                detail="No new data file has been uploaded for retraining.",
            )

    args["mode"] = "forecasting"
    args["n_jobs"] = settings.n_jobs
    return _enqueue_job(db, request.report_id, args, report)


@router.post("/forecast-generation", response_model=JobResponse)
async def start_forecast_generation(
    report_id: str = Form(...),
    forecast_horizon: int = Form(12),
    models: str = Form(""),
    future_covariates: UploadFile | None = File(None),
    db: Session = Depends(get_session),
) -> JobResponse:
    """Generate future forecasts with saved models from a completed run."""
    report = repo.get_report(db, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Dataset {report_id} not found")

    _guard_queue_capacity(db, report_id, report)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        raise HTTPException(status_code=400, detail="Select at least one model to forecast with.")
    if forecast_horizon < 1:
        raise HTTPException(status_code=400, detail="Forecast horizon must be at least 1.")

    if future_covariates is not None and future_covariates.filename:
        if not future_covariates.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Future covariates file must be a CSV")
        raw = await future_covariates.read()
        encoding = detect_encoding(raw)
        try:
            text = raw.decode(encoding)
            cov_df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to parse future covariates CSV: {e}"
            ) from e
        csv_buf = io.StringIO()
        cov_df.to_csv(csv_buf, index=False)
        get_storage().put_text(f"{report_id}/future_covariates", csv_buf.getvalue())

    args: dict[str, Any] = {
        "mode": FORECAST_GENERATION_MODE,
        "train_group": model_list,
        "forecast_horizon": forecast_horizon,
    }
    args.update(_ts_role_args(report))
    if not args["time_component"] or not args["target"]:
        raise HTTPException(
            status_code=400,
            detail="This dataset has no time-series configuration — "
            "train a forecasting model first.",
        )

    return _enqueue_job(db, report_id, args, report)


@router.get("/active/{report_id}", response_model=JobResponse | None)
def get_active_job_for_report(
    report_id: str,
    db: Session = Depends(get_session),
) -> JobResponse | None:
    """Get the queued or running job for a dataset, if any."""
    job = repo.get_active_job_for_report(db, report_id)
    return _serialize_job(job) if job else None


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
        # Just remove from queue; restore the report's pre-queue status.
        # The job never started, so create its (empty) output log with the
        # reason — failed jobs always have a log tab entry.
        set_state(db, job_id, "failed", error="Cancelled by user")
        with contextlib.suppress(Exception):
            append_log(get_storage(), job.report_uuid, "Queued job cancelled before starting.")
        prev_status = (job.args or {}).get("prev_report_status")
        repo.update_report_status(db, job.report_uuid, prev_status or "Failed")
        db.commit()
        return {"status": "cancelled", "message": "Queued job cancelled"}

    if job.state in ("running", "cancelling"):
        # Kill the worker immediately. A cooperative flag alone would let the
        # current model run to completion first (which can take an hour of
        # tuning) — partial artifacts of the killed run are archived/cleared
        # by the next run of the same dataset.
        storage = get_storage()
        set_cancel_flag(storage, job.report_uuid)
        set_state(db, job_id, "cancelling")
        get_manager().cancel_current_job()
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
        last_log_len = -1
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

                # Read live log from storage (written incrementally by jobworker)
                log_text = ""
                try:
                    if storage.exists(f"{report_id}/output_log"):
                        log_text = storage.get_text(f"{report_id}/output_log")
                except Exception:
                    pass
                log_tail = log_text[-10000:] if len(log_text) > 10000 else log_text
                log_len = len(log_text)
                # A queued job hasn't started: any log on disk belongs to a
                # previous run and must not initialize the checklist.
                if state == "queued":
                    log_tail = ""
                    log_len = 0

                # Authoritative per-model states (written by the worker); the
                # log tail is display-only and truncated, so the checklist
                # must not be derived from it.
                from runner.progress import read_models

                model_states = (
                    read_models(storage, report_id)
                    if state in ("running", "succeeded", "failed")
                    else None
                )

                # Emit event if progress or log changed, or job is done
                current = (
                    (progress.completed, progress.total, progress.message) if progress else None
                )
                if (
                    current != last_progress
                    or log_len != last_log_len
                    or state in ("succeeded", "failed")
                ):
                    last_progress = current
                    last_log_len = log_len
                    event_data = {
                        "id": current_job.id,
                        "report_uuid": current_job.report_uuid,
                        "state": state,
                        "args": current_job.args,
                        "models": model_states,
                        "progress": progress.completed if progress else current_job.progress,
                        "progress_total": progress.total
                        if progress
                        else current_job.progress_total,
                        "progress_message": progress.message
                        if progress
                        else current_job.progress_message,
                        "error": current_job.error,
                        "log": log_tail,
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


@router.get("/ml-options/forecasting")
def ml_options_forecasting() -> dict[str, Any]:
    """Get forecasting options for the Prepare page."""
    return get_forecasting_options()


@router.post("/recover")
def recover_stale_jobs(db: Session = Depends(get_session)) -> dict[str, int]:
    """Mark stale running jobs as failed (called on app startup)."""
    count = mark_stale_jobs_failed(db)
    return {"recovered": count}
