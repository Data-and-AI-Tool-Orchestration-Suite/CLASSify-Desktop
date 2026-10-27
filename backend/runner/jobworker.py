"""Job worker — subprocess entry point that runs the ML trainer.

Launched by the JobManager as a separate process for crash isolation
and clean cancellation.  Reads args from the DB, calls engine.trainer(),
writes results to storage, and exits.

Entry point: ``python -m runner.jobworker <job_id>``
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure backend/ is on sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


from classify_api import repositories as repo  # noqa: E402
from classify_api.db import get_session_factory, run_migrations  # noqa: E402
from classify_api.settings import get_settings  # noqa: E402
from ml.args import ForecastArgs, TrainingArgs  # noqa: E402
from ml.column_types import read_csv_from_storage  # noqa: E402
from runner.cancellation import CancelToken, clear_cancel_flag  # noqa: E402
from runner.progress import write_models, write_progress  # noqa: E402
from storage.factory import get_storage  # noqa: E402

FORECASTING_MODE = "forecasting"
FORECAST_GENERATION_MODE = "forecast-generation"

_RUN_ARTIFACT_SUFFIXES = [
    "results",
    "results.json",
    "output_log",
    "models",
    "worker_stderr.log",
    "scaler.joblib",
    "labeled",
    "logisticregression_odds_ratio",
    "forecasts",
    "forecast_generation_log",
]


def _archive_run(storage: Any, report_id: str, prev_job_id: str) -> None:
    """Copy run-specific artifacts to an archive dir so reruns preserve history."""
    archives_prefix = f"{report_id}/archive/"
    all_keys = storage.list(f"{report_id}/")
    for key in all_keys:
        if key.startswith(archives_prefix):
            continue
        suffix = key[len(f"{report_id}/") :]
        is_run_artifact = (
            suffix in _RUN_ARTIFACT_SUFFIXES
            or suffix.endswith("_model.joblib")
            or suffix.startswith("shap_rows_")
            or suffix.startswith("forecast_data_")
            or suffix.startswith("viz/")
        )
        if is_run_artifact:
            storage.copy(key, f"{archives_prefix}{prev_job_id}/{suffix}")


def _clear_run_artifacts(storage: Any, report_id: str) -> None:
    """Delete previous run's top-level artifacts (they are archived by now).

    Without this, artifacts from earlier runs (viz, SHAP rows, models) linger
    and leak into the new run's results page.
    """
    archives_prefix = f"{report_id}/archive/"
    all_keys = storage.list(f"{report_id}/")
    for key in all_keys:
        if key.startswith(archives_prefix):
            continue
        suffix = key[len(f"{report_id}/") :]
        is_run_artifact = (
            suffix in _RUN_ARTIFACT_SUFFIXES
            or suffix.endswith("_model.joblib")
            or suffix.startswith("shap_rows_")
            or suffix.startswith("forecast_data_")
            or suffix.startswith("viz/")
        )
        if is_run_artifact:
            storage.delete(key)


def run_job(job_id: str) -> int:
    """Run a single training job.  Returns exit code (0=success, 1=failure)."""
    settings = get_settings()
    settings.ensure_dirs()

    # Ensure migrations are up
    run_migrations()

    factory = get_session_factory()
    db = factory()
    storage = None
    report_id: str | None = None

    try:
        job = repo.get_job(db, job_id)
        if job is None:
            print(f"Job {job_id} not found", file=sys.stderr)
            return 1

        report_id = job.report_uuid
        args_dict = job.args or {}
        job_mode = args_dict.get("mode", "")
        if job_mode == FORECASTING_MODE:
            args: Any = ForecastArgs.from_dict(args_dict)
        elif job_mode == FORECAST_GENERATION_MODE:
            args = ForecastArgs.from_dict(args_dict)
        else:
            args = TrainingArgs.from_dict(args_dict)
        args.report_uuid = report_id
        if not args.n_jobs:
            args.n_jobs = os.cpu_count() or 1

        storage = get_storage()

        # Add-ons live in a separate dir on disk; the main app prepends it at
        # boot, but this worker is its own process and needs it too — without
        # it, add-on models (TabPFN/SDV) fail the import check and get skipped.
        # The faker _MEIPASS override is job-side only (faker is used during
        # training/synthesis; the main app's static mount needs the real path).
        from classify_api.services.addon_service import init_addons

        init_addons()

        # Archive previous run's artifacts (preserves run history — including
        # failed/cancelled runs' logs, which is the only way to inspect why
        # they failed), then clear them so the new run starts clean.
        # Forecast-generation jobs are additive to the current run (they only
        # write forecasts + a generation log) — archiving here would hide the
        # metrics/charts the forecasts belong to.
        if job_mode != FORECAST_GENERATION_MODE:
            prev_job = repo.get_previous_job_by_report(db, report_id, job_id)
            # Additive forecast-generation jobs never own artifacts — walk back
            # to the training run the current artifacts belong to.
            while (
                prev_job is not None
                and (prev_job.args or {}).get("mode") == FORECAST_GENERATION_MODE
            ):
                prev_job = repo.get_previous_job_by_report(db, report_id, prev_job.id)
            if prev_job:
                _archive_run(storage, report_id, prev_job.id)
            _clear_run_artifacts(storage, report_id)

        # Start the output log immediately after the clear so ANY later
        # failure (dataset read errors, addon init, training) is visible in
        # the results page log tab.
        log_lines: list[str] = [f"Job {job_id} started."]
        storage.put_text(f"{report_id}/output_log", "\n".join(log_lines) + "\n")

        # Read the processed dataset (classification/unsupervised only — the
        # forecasting engine reads and parses its own data from storage)
        df = None
        testset = None
        if job_mode not in (FORECASTING_MODE, FORECAST_GENERATION_MODE):
            try:
                df = read_csv_from_storage(storage, f"{report_id}/file", index_col="index")
            except Exception:
                df = read_csv_from_storage(storage, f"{report_id}/file")

            # Read separate testset if present
            if storage.exists(f"{report_id}/testset"):
                try:
                    testset = read_csv_from_storage(
                        storage, f"{report_id}/testset", index_col="index"
                    )
                except Exception:
                    testset = read_csv_from_storage(storage, f"{report_id}/testset")

        # Create cancel token
        cancel_token = CancelToken(storage, report_id)

        # Reset progress from any previous run
        write_progress(storage, report_id, 0, 0, "Starting...")
        repo.update_job_progress(db, job_id, 0, 0, "Starting...")
        db.commit()

        # Progress callback
        def on_progress(completed: int, total: int, message: str) -> None:
            write_progress(storage, report_id, completed, total, message)
            repo.update_job_progress(db, job_id, completed, total, message)
            db.commit()

        # ── Per-model state tracking ──
        # The engine's log lines are the source of truth for which models are
        # running/done/failed.  Track them here and persist to models.json so
        # the UI shows the authoritative checklist without re-parsing the
        # (truncated) log tail.
        requested_models: list[str] = args.train_group if isinstance(args.train_group, list) else []
        model_states: dict[str, dict[str, Any]] = {
            name: {"name": name, "state": "pending", "score": None, "note": ""}
            for name in requested_models
        }
        current_model: str | None = None

        def _persist_models() -> None:
            with contextlib.suppress(Exception):
                write_models(storage, report_id, list(model_states.values()))

        def _track_model_line(msg: str) -> None:
            nonlocal current_model

            m = re.match(r"^Evaluating model:\s*(\S+)", msg.strip())
            if m:
                current_model = m.group(1)
                if current_model in model_states:
                    model_states[current_model]["state"] = "running"
                    _persist_models()
                return
            m = re.match(r"^Forecasting model:\s*(\S+)", msg.strip())
            if m:
                current_model = m.group(1)
                if current_model in model_states:
                    model_states[current_model]["state"] = "running"
                    _persist_models()
                return
            m = re.match(r"^Best params:.*\bscore:\s*(-?[\d.]+)", msg.strip())
            if m and current_model and current_model in model_states:
                model_states[current_model]["state"] = "done"
                with contextlib.suppress(Exception):
                    model_states[current_model]["score"] = float(m.group(1))
                _persist_models()
                return
            m = re.match(r"^(\S+) training complete\.$", msg.strip())
            if m and m.group(1) in model_states:
                model_states[m.group(1)]["state"] = "done"
                _persist_models()
                return
            m = re.match(r"^(\S+) forecast complete\.$", msg.strip())
            if m and m.group(1) in model_states:
                model_states[m.group(1)]["state"] = "done"
                _persist_models()
                return
            m = re.match(r"^Skipping\s+(\S+?)(?:\s+[—–-]\s+)(.+)$", msg.strip())
            if m:
                name = m.group(1).rstrip(":,;")
                if name in model_states:
                    model_states[name]["state"] = "skipped"
                    model_states[name]["note"] = m.group(2)
                    _persist_models()
                return
            m = re.match(
                r"^Error in (?:training|clustering|train/evaluation of|forecasting)\s+(\S+)",
                msg.strip(),
            )
            if m:
                name = m.group(1).rstrip(":,;")
                if name in model_states:
                    model_states[name]["state"] = "failed"
                    model_states[name]["note"] = "error — see log"
                    _persist_models()

        _persist_models()

        # Log callback — writes incrementally to storage for live streaming
        def on_log(msg: str) -> None:
            log_lines.append(msg)
            with contextlib.suppress(Exception):
                storage.put_text(f"{report_id}/output_log", "\n".join(log_lines) + "\n")
            with contextlib.suppress(Exception):
                _track_model_line(msg)

        # Capture the frozen flag BEFORE the faker unfreeze below — the
        # threading-backend decision must reflect how this process started.
        was_frozen = getattr(sys, "frozen", False)

        # faker (SDV synthesis) resolves its data paths via sys.frozen
        # checks — unfreeze for the training phase so add-on packages
        # resolve from the add-on dir instead of sys._MEIPASS
        if was_frozen:
            sys.__dict__["frozen"] = False

        # Run the trainer (or the forecast-generation flow)

        def _invoke() -> None:
            if job_mode == FORECAST_GENERATION_MODE:
                from ml.forecast.generate import run_forecast_generation

                run_forecast_generation(args, storage, on_progress, on_log, cancel_token)
                return
            from ml.engine import trainer

            trainer(
                args=args,
                storage=storage,
                full_dataset=df,
                testset=testset,
                on_progress=on_progress,
                log_cb=on_log,
                cancel_token=cancel_token,
            )

        # Process-based joblib backends (loky) deadlock inside frozen apps —
        # spawn attempts relaunch the bundled exe instead of a worker. Force
        # thread-based parallelism; identical math, no worker processes.
        # (The was_frozen snapshot is taken before the faker unfreeze —
        # checking sys.frozen here again would always be False and leak a
        # loky pool of classify-jobworker.exe processes per training run.)
        if was_frozen:
            from joblib import parallel_backend

            with parallel_backend("threading"):
                _invoke()
        else:
            _invoke()

        # Check if cancelled
        if cancel_token.is_set():
            repo.update_job_state(db, job_id, "failed", error="Cancelled by user")
            db.commit()
            return 1

        # Mark success
        repo.update_job_state(db, job_id, "succeeded")
        repo.update_report_status(db, report_id, "Processed")
        repo.create_result(db, report_uuid=report_id)
        db.commit()
        return 0

    except Exception as e:
        traceback.print_exc()
        try:
            repo.update_job_state(db, job_id, "failed", error=str(e))
            if report_id is not None:
                repo.update_report_status(db, report_id, "Failed")
            db.commit()
        except Exception:
            pass
        # Persist the failure into the output log — the results page log tab
        # is where users look for the reason a job failed.
        if storage is not None and report_id is not None:
            with contextlib.suppress(Exception):
                from runner.progress import append_log

                append_log(storage, report_id, f"Training failed: {type(e).__name__}: {e}")
        return 1

    finally:
        try:
            if storage is not None and report_id is not None:
                clear_cancel_flag(storage, report_id)
        except Exception:
            pass
        db.close()


def main() -> None:
    """Console-script entry point (``classify-jobworker``)."""
    import multiprocessing

    multiprocessing.freeze_support()
    if len(sys.argv) < 2:
        print("Usage: classify-jobworker <job_id>", file=sys.stderr)
        sys.exit(2)
    job_id = sys.argv[1]
    exit_code = run_job(job_id)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
