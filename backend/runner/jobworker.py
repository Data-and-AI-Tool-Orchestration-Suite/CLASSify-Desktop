"""Job worker — subprocess entry point that runs the ML trainer.

Launched by the JobManager as a separate process for crash isolation
and clean cancellation.  Reads args from the DB, calls engine.trainer(),
writes results to storage, and exits.

Entry point: ``python -m runner.jobworker <job_id>``
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

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

        # Progress callback
        def on_progress(completed: int, total: int, message: str) -> None:
            write_progress(storage, report_id, completed, total, message)
            repo.update_job_progress(db, job_id, completed, total, message)
            db.commit()

        # Log callback
        log_lines: list[str] = []

        def on_log(msg: str) -> None:
            log_lines.append(msg)

        # Run the trainer
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
    if len(sys.argv) < 2:
        print("Usage: classify-jobworker <job_id>", file=sys.stderr)
        sys.exit(2)
    job_id = sys.argv[1]
    exit_code = run_job(job_id)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
