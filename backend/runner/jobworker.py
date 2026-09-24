"""Job worker — subprocess entry point that runs the ML trainer.

Launched by the JobManager as a separate process for crash isolation
and clean cancellation.  Reads args from the DB, calls engine.trainer(),
writes results to storage, and exits.

Entry point: ``python -m runner.jobworker <job_id>``
"""

from __future__ import annotations

import contextlib
import os
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
from ml.args import TrainingArgs  # noqa: E402
from ml.column_types import read_csv_from_storage  # noqa: E402
from runner.cancellation import CancelToken, clear_cancel_flag  # noqa: E402
from runner.progress import write_progress  # noqa: E402
from storage.factory import get_storage  # noqa: E402

_RUN_ARTIFACT_SUFFIXES = [
    "results",
    "results.json",
    "output_log",
    "scaler.joblib",
    "labeled",
    "logisticregression_odds_ratio",
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

    try:
        job = repo.get_job(db, job_id)
        if job is None:
            print(f"Job {job_id} not found", file=sys.stderr)
            return 1

        report_id = job.report_uuid
        args_dict = job.args or {}
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

        # Archive previous run's artifacts if they exist (preserves run history),
        # then clear them so the new run starts from a clean slate.
        prev_job = repo.get_previous_job_by_report(db, report_id, job_id)
        if prev_job and storage.exists(f"{report_id}/results"):
            _archive_run(storage, report_id, prev_job.id)
        _clear_run_artifacts(storage, report_id)

        # Read the processed dataset
        try:
            df = read_csv_from_storage(storage, f"{report_id}/file", index_col="index")
        except Exception:
            df = read_csv_from_storage(storage, f"{report_id}/file")

        # Read separate testset if present
        testset = None
        if storage.exists(f"{report_id}/testset"):
            try:
                testset = read_csv_from_storage(storage, f"{report_id}/testset", index_col="index")
            except Exception:
                testset = read_csv_from_storage(storage, f"{report_id}/testset")

        # Create cancel token
        cancel_token = CancelToken(storage, report_id)

        # Reset progress from any previous run
        write_progress(storage, report_id, 0, 0, "Starting...")
        repo.update_job_progress(db, job_id, 0, 0, "Starting...")
        db.commit()

        # Also clear the output log from any previous run
        log_lines: list[str] = []
        storage.put_text(f"{report_id}/output_log", "")

        # Progress callback
        def on_progress(completed: int, total: int, message: str) -> None:
            write_progress(storage, report_id, completed, total, message)
            repo.update_job_progress(db, job_id, completed, total, message)
            db.commit()

        # Log callback — writes incrementally to storage for live streaming
        def on_log(msg: str) -> None:
            log_lines.append(msg)
            with contextlib.suppress(Exception):
                storage.put_text(f"{report_id}/output_log", "\n".join(log_lines) + "\n")

        # faker (SDV synthesis) resolves its data paths via sys.frozen
        # checks — unfreeze for the training phase so add-on packages
        # resolve from the add-on dir instead of sys._MEIPASS
        if getattr(sys, "frozen", False):
            sys.frozen = False

        # Run the trainer
        from ml.engine import trainer

        # Process-based joblib backends (loky) deadlock inside frozen apps —
        # spawn attempts relaunch the bundled exe instead of a worker. Force
        # thread-based parallelism; identical math, no worker processes.
        if getattr(sys, "frozen", False):
            from joblib import parallel_backend

            with parallel_backend("threading"):
                trainer(
                    args=args,
                    storage=storage,
                    full_dataset=df,
                    testset=testset,
                    on_progress=on_progress,
                    log_cb=on_log,
                    cancel_token=cancel_token,
                )
        else:
            trainer(
                args=args,
                storage=storage,
                full_dataset=df,
                testset=testset,
                on_progress=on_progress,
                log_cb=on_log,
                cancel_token=cancel_token,
            )

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
            report_uuid = job.report_uuid if job else "unknown"
            repo.update_job_state(db, job_id, "failed", error=str(e))
            repo.update_report_status(db, report_uuid, "Failed")
            db.commit()
        except Exception:
            pass
        return 1

    finally:
        try:
            if "storage" in dir() and "report_id" in dir():
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
