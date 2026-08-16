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
UPDATE_MANIFEST_URL = "https://github.com/Data-and-AI-Tool-Orchestration-Suite/CLASSify-Desktop/releases/latest/download/latest.json"

APP_VERSION = "1.0.0"


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


@router.get("/first-run")
def first_run_status() -> dict[str, object]:
    """Get first-run wizard state."""
    from classify_api.services.first_run import get_wizard_state

    return get_wizard_state()


@router.post("/first-run/complete")
def complete_first_run() -> dict[str, str]:
    """Mark the first-run wizard as complete."""
    from classify_api.services.first_run import mark_initialized

    mark_initialized()
    return {"status": "ok"}


@router.get("/metric-defs")
def metric_definitions() -> dict[str, object]:
    """Get metric tooltip definitions for the results table.

    Ported from the web app's tooltip dictionary.
    """
    return {
        "test_auc": "Area Under the ROC Curve (test set). Measures discrimination: 0.5=chance, 1.0=perfect.",
        "test_acc": "Accuracy (test set). Proportion of correct predictions.",
        "test_sensitivity": "Sensitivity / Recall (test set). True positive rate: TP/(TP+FN).",
        "test_specificity": "Specificity (test set). True negative rate: TN/(TN+FP).",
        "test_npv": "Negative Predictive Value (test set). TN/(TN+FN).",
        "test_ppv": "Positive Predictive Value / Precision (test set). TP/(TP+FP).",
        "test_f1score": "F1 Score (test set). Harmonic mean of precision and recall.",
        "test_kappa": "Cohen's Kappa (test set). Agreement corrected for chance.",
        "trt_auc": "Area Under the ROC Curve (training set). Check for overfitting vs test AUC.",
        "trt_acc": "Accuracy (training set).",
        "trt_sensitivity": "Sensitivity (training set).",
        "trt_specificity": "Specificity (training set).",
        "trt_f1score": "F1 Score (training set).",
        "cvt_auc": "Cross-validated AUC (mean ± margin of error).",
        "cvt_acc": "Cross-validated accuracy (mean ± margin of error).",
        "cvt_sensitivity": "Cross-validated sensitivity (mean ± margin of error).",
        "cvt_specificity": "Cross-validated specificity (mean ± margin of error).",
        "cvt_f1score": "Cross-validated F1 score (mean ± margin of error).",
        "best_score": "Best Optuna tuning score for this model.",
        "silhouette_score": "Clustering: how similar an object is to its own cluster vs others. Range [-1, 1], higher is better.",
        "calinski_harabasz_score": "Clustering: ratio of between-cluster to within-cluster dispersion. Higher is better.",
        "davies_bouldin_score": "Clustering: average similarity ratio of each cluster. Lower is better (minimum 0).",
    }
