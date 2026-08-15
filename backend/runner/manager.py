"""Job manager — background thread that runs jobs as subprocesses.

Started at app boot.  Pulls the next queued job, spawns a subprocess
(``python -m runner.jobworker <job_id>``), monitors it, and handles
cancellation via process-group kill.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
import time

import structlog

from classify_api import repositories as repo
from classify_api.db import get_session_factory
from classify_api.settings import get_settings
from runner.cancellation import clear_cancel_flag, set_cancel_flag
from storage.factory import get_storage

log = structlog.get_logger()

# Grace period (seconds) before SIGKILL after cancel
_CANCEL_GRACE_SECONDS = 10


class JobManager:
    """Manages the job queue and subprocess lifecycle.

    Runs in a background daemon thread.  Only one job runs at a time.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._current_process: subprocess.Popen[bytes] | None = None
        self._current_job_id: str | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the manager thread (called on app startup)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="job-manager")
        self._thread.start()
        log.info("job_manager.started")

    def stop(self) -> None:
        """Signal the manager to stop (called on app shutdown).

        Waits for any in-flight subprocess to finish rather than killing it,
        so completed results aren't lost.
        """
        self._stop_event.set()
        # Don't cancel — wait for the current job to finish naturally
        if self._thread is not None:
            self._thread.join(timeout=120)

    def cancel_current_job(self) -> None:
        """Cancel the currently running job (if any)."""
        with self._lock:
            if self._current_job_id:
                db_factory = get_session_factory()
                db = db_factory()
                try:
                    storage = get_storage()
                    job = repo.get_job(db, self._current_job_id)
                    if job:
                        report_id = job.report_uuid
                        set_cancel_flag(storage, report_id)
                        repo.update_job_state(db, self._current_job_id, "cancelling")
                        db.commit()

                    # Send SIGTERM to the process group
                    if self._current_process is not None:
                        try:
                            if sys.platform == "win32":
                                self._current_process.terminate()
                            else:
                                os.killpg(os.getpgid(self._current_process.pid), signal.SIGTERM)
                        except (ProcessLookupError, OSError):
                            pass

                        # Wait for grace period, then SIGKILL
                        try:
                            self._current_process.wait(timeout=_CANCEL_GRACE_SECONDS)
                        except subprocess.TimeoutExpired:
                            if sys.platform == "win32":
                                self._current_process.kill()
                            else:
                                with contextlib.suppress(ProcessLookupError, OSError):
                                    os.killpg(os.getpgid(self._current_process.pid), signal.SIGKILL)

                    if self._current_job_id:
                        repo.update_job_state(
                            db, self._current_job_id, "failed", error="Cancelled by user"
                        )
                        db.commit()
                finally:
                    db.close()
                    self._current_process = None
                    self._current_job_id = None

    def is_running(self) -> bool:
        """Check if a job is currently running."""
        with self._lock:
            return self._current_process is not None

    def _run_loop(self) -> None:
        """Main loop: pull next job, run it, repeat."""
        db_factory = get_session_factory()

        while not self._stop_event.is_set():
            db = db_factory()
            try:
                # Check if a job is already running
                if self.is_running():
                    time.sleep(1)
                    continue

                # Get next queued job
                job = repo.get_next_queued_job(db)
                if job is None:
                    time.sleep(2)
                    continue

                # Mark as running
                repo.update_job_state(db, job.id, "running")
                repo.update_report_status(db, job.report_uuid, "Processing")
                db.commit()

                report_id = job.report_uuid
                job_id = job.id
                db.close()

                # Run the job in a subprocess
                self._run_subprocess(job_id, report_id)

            except Exception as e:
                log.error("job_manager.loop_error", error=str(e))
            finally:
                db.close()

    def _run_subprocess(self, job_id: str, report_id: str) -> None:
        """Spawn the jobworker subprocess and monitor it."""
        settings = get_settings()

        # Build the command
        python_exe = sys.executable
        cmd = [python_exe, "-m", "runner.jobworker", job_id]

        # Set environment
        env = os.environ.copy()
        env["CLASSIFY_DATA_DIR"] = str(settings.data_dir)
        env["MPLBACKEND"] = "Agg"

        # Start the subprocess in its own process group
        with self._lock:
            self._current_job_id = job_id

        try:
            if sys.platform == "win32":
                self._current_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                self._current_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                )

            log.info("job_manager.subprocess_started", job_id=job_id, pid=self._current_process.pid)

            # Wait for completion (handle cancellation setting process to None)
            proc = self._current_process
            if proc is None:
                log.info("job_manager.subprocess_cancelled_before_wait", job_id=job_id)
                return

            stdout, stderr = proc.communicate()
            exit_code = proc.returncode

            log.info("job_manager.subprocess_finished", job_id=job_id, exit_code=exit_code)

            # Update DB based on exit code
            db_factory = get_session_factory()
            db = db_factory()
            try:
                job = repo.get_job(db, job_id)
                if job:
                    if exit_code == 0:
                        if job.state != "succeeded":
                            repo.update_job_state(db, job_id, "succeeded")
                            repo.update_report_status(db, report_id, "Processed")
                            repo.create_result(db, report_uuid=report_id)
                    elif job.state == "succeeded":
                        log.info("job_manager.subprocess_nonzero_but_succeeded", job_id=job_id)
                    else:
                        if job.state != "failed":
                            error_msg = (
                                stderr.decode("utf-8", errors="replace")[-500:]
                                if stderr
                                else "Unknown error"
                            )
                            repo.update_job_state(db, job_id, "failed", error=error_msg)
                            repo.update_report_status(db, report_id, "Failed")
                    db.commit()

                # Clean up cancel flag
                storage = get_storage()
                clear_cancel_flag(storage, report_id)
            finally:
                db.close()

        except Exception as e:
            log.error("job_manager.subprocess_error", job_id=job_id, error=str(e))
            db_factory = get_session_factory()
            db = db_factory()
            try:
                repo.update_job_state(db, job_id, "failed", error=str(e))
                repo.update_report_status(db, report_id, "Failed")
                db.commit()
            finally:
                db.close()
        finally:
            with self._lock:
                self._current_process = None
                self._current_job_id = None


# Singleton
_manager: JobManager | None = None


def get_manager() -> JobManager:
    """Return the singleton JobManager instance."""
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def start_manager() -> None:
    """Start the job manager (called on app startup)."""
    get_manager().start()


def stop_manager() -> None:
    """Stop the job manager (called on app shutdown)."""
    global _manager
    if _manager is not None:
        _manager.stop()
        _manager = None
