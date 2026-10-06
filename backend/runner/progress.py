"""Progress tracking — worker writes progress to a file, manager reads it.

The worker subprocess writes ``<report_id>/progress.json`` periodically.
The manager polls it and feeds SSE events to the frontend.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from storage.base import Storage


@dataclass
class ProgressUpdate:
    """A single progress update from the worker."""

    completed: int
    total: int
    message: str
    timestamp: float


def write_progress(
    storage: Storage, report_id: str, completed: int, total: int, message: str
) -> None:
    """Write a progress update to storage (called by the worker)."""
    data: dict[str, Any] = {
        "completed": completed,
        "total": total,
        "message": message,
        "timestamp": time.time(),
    }
    storage.put_text(f"{report_id}/progress.json", json.dumps(data))


def read_progress(storage: Storage, report_id: str) -> ProgressUpdate | None:
    """Read the latest progress update from storage (called by the manager)."""
    try:
        text = storage.get_text(f"{report_id}/progress.json")
        data = json.loads(text)
        return ProgressUpdate(
            completed=data["completed"],
            total=data["total"],
            message=data["message"],
            timestamp=data["timestamp"],
        )
    except Exception:
        return None


def append_log(storage: Storage, report_id: str, message: str) -> None:
    """Append a line to the live output log."""
    try:
        existing = storage.get_text(f"{report_id}/output_log")
    except Exception:
        existing = ""
    storage.put_text(f"{report_id}/output_log", existing + message + "\n")


# ── Per-model run states ──
# The worker tracks model states in <report_id>/models so the UI can show the
# authoritative checklist without re-parsing the (truncated) log tail.


def write_models(storage: Storage, report_id: str, models: list[dict[str, Any]]) -> None:
    """Persist the per-model state list (called by the worker)."""
    storage.put_text(f"{report_id}/models", json.dumps(models))


def read_models(storage: Storage, report_id: str) -> list[dict[str, Any]] | None:
    """Read the per-model state list (called by the manager)."""
    try:
        data = json.loads(storage.get_text(f"{report_id}/models"))
        return data if isinstance(data, list) else None
    except Exception:
        return None
