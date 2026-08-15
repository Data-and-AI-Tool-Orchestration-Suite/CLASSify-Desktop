"""Cancellation — flag-based cancellation for subprocess workers.

The manager sets a cancel flag (file on disk + DB state).
The worker checks the flag periodically and exits cleanly.
If the worker doesn't exit within the grace period, the process group is killed.
"""

from __future__ import annotations

from storage.base import Storage


def set_cancel_flag(storage: Storage, report_id: str) -> None:
    """Set the cancel flag for a job (worker checks this and exits)."""
    storage.put_text(f"{report_id}/cancel.flag", "cancel")


def is_cancelled(storage: Storage, report_id: str) -> bool:
    """Check if the cancel flag is set (called by the worker)."""
    return storage.exists(f"{report_id}/cancel.flag")


def clear_cancel_flag(storage: Storage, report_id: str) -> None:
    """Remove the cancel flag (after job completes or is cancelled)."""
    storage.delete(f"{report_id}/cancel.flag")


class CancelToken:
    """Token object passed to the engine for cancellation checks.

    Wraps a storage + report_id so the engine can call ``token.is_set()``
    without knowing about storage internals.
    """

    def __init__(self, storage: Storage, report_id: str) -> None:
        self._storage = storage
        self._report_id = report_id

    def is_set(self) -> bool:
        """Return True if the job has been cancelled."""
        return is_cancelled(self._storage, self._report_id)
