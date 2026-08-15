"""First-run detection and wizard state management."""

from __future__ import annotations

from pathlib import Path

from classify_api.settings import get_settings


def is_first_run() -> bool:
    """Check if this is the first time the app is launched.

    Returns True if the data directory exists but has no marker file.
    """
    settings = get_settings()
    settings.ensure_dirs()
    marker = settings.data_dir / ".classify_initialized"
    return not marker.exists()


def mark_initialized() -> None:
    """Mark the app as initialized (first-run wizard completed)."""
    settings = get_settings()
    settings.ensure_dirs()
    marker = settings.data_dir / ".classify_initialized"
    marker.write_text("true")


def get_wizard_state() -> dict[str, object]:
    """Get the current wizard state for the frontend."""
    settings = get_settings()
    return {
        "first_run": is_first_run(),
        "data_dir": str(settings.data_dir),
        "disk_free_bytes": _get_disk_free(settings.data_dir),
        "cpu_count": _get_cpu_count(),
        "has_addons": _check_addon_status(),
    }


def _get_disk_free(path: Path) -> int:
    import shutil

    target = path if path.exists() else path.parent
    return shutil.disk_usage(str(target)).free


def _get_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def _check_addon_status() -> dict[str, bool]:
    from ml.backends import is_available

    return {
        "tabpfn": is_available("tabpfn"),
        "sdv": is_available("sdv"),
    }
