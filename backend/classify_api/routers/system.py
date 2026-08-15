"""System-level endpoints: health, version, disk usage, update check."""

from __future__ import annotations

import json
import os
import platform
import shutil
import urllib.request

from fastapi import APIRouter

from classify_api.settings import get_settings

router = APIRouter()

# Stable URL for the update manifest — always redirects to the latest stable release
UPDATE_MANIFEST_URL = (
    "https://github.com/uk-applied-ai/CLASSify-app/releases/latest/download/latest.json"
)

APP_VERSION = "1.0.0.dev0"


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe used by the desktop shell on boot."""
    return {"status": "ok"}


@router.get("/info")
def info() -> dict[str, object]:
    """Application + environment metadata."""
    settings = get_settings()
    return {
        "app": "CLASSify Desktop",
        "version": APP_VERSION,
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "data_dir": str(settings.data_dir),
        "dev_mode": settings.dev_mode,
    }


@router.get("/usage")
def usage() -> dict[str, int]:
    """Disk usage of the data directory."""
    settings = get_settings()
    target = settings.data_dir if settings.data_dir.exists() else settings.data_dir.parent
    usage_stat = shutil.disk_usage(str(target))
    return {
        "total": usage_stat.total,
        "used": usage_stat.used,
        "free": usage_stat.free,
    }


@router.get("/cpu-count")
def cpu_count() -> dict[str, int]:
    """Number of logical CPUs available for ``n_jobs`` defaults."""
    return {"count": os.cpu_count() or 1}


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a semver string into a comparable tuple."""
    parts = version.lstrip("v").split(".")
    result = []
    for part in parts:
        try:
            result.append(int(part))
        except ValueError:
            result.append(0)
    return tuple(result)


@router.get("/check-updates")
def check_updates() -> dict[str, object]:
    """Check for app updates by fetching the latest.json manifest.

    Returns info about the latest available version and whether an update
    is needed.  The user downloads the installer manually via the URL.
    """
    settings = get_settings()

    result: dict[str, object] = {
        "current_version": APP_VERSION,
        "update_available": False,
        "latest_version": None,
        "download_url": None,
        "release_notes": "",
        "error": None,
    }

    if settings.dev_mode:
        result["error"] = "Update check disabled in dev mode"
        return result

    try:
        req = urllib.request.Request(
            UPDATE_MANIFEST_URL,
            headers={"User-Agent": "CLASSify-Desktop/" + APP_VERSION},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            manifest = json.loads(resp.read().decode())

        latest_version = manifest.get("version", "0.0.0")
        result["latest_version"] = latest_version
        result["release_notes"] = manifest.get("notes", "")

        if _parse_version(latest_version) > _parse_version(APP_VERSION):
            result["update_available"] = True

            # Pick the right asset for this platform
            assets = manifest.get("assets", {})
            system = platform.system()
            if system == "Windows":
                asset = assets.get("windows-x64", {})
            elif system == "Darwin":
                asset = assets.get("macos-universal2", {})
            else:
                asset = assets.get("linux-x86_64-appimage") or assets.get("linux-x86_64-deb", {})

            result["download_url"] = asset.get("url")
            result["download_size"] = asset.get("size", 0)
            result["sha256"] = asset.get("sha256")

    except Exception as e:
        result["error"] = f"Failed to check for updates: {e}"

    return result
