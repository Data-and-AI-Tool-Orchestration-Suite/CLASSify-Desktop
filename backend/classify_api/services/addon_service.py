"""Add-on installer service — manages torch-gated optional ML packages.

Add-ons (TabPFN, SDV) pull torch (~2GB) and are NOT included in the base
installer.  Users install them on demand via Settings → Add-ons or the
API.  Installed add-ons live in ``<appdata>/addons/pythonlibs`` and are
prepended to ``sys.path`` at boot so the ML engine can import them.

Installs run asynchronously in a background thread so the API doesn't
block for the 5-10 minutes a large download takes.  Progress is tracked
in a shared dict that the frontend polls via the install-status endpoint.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from classify_api.settings import get_settings
from ml.backends import refresh_cache

log = structlog.get_logger()

_CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


@dataclass
class AddonManifest:
    """Manifest describing an installable add-on."""

    name: str
    version: str
    description: str
    pip_deps: list[str]
    size_estimate_mb: int
    min_app_version: str
    provides: list[str]
    install_script: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "pip_deps": self.pip_deps,
            "size_estimate_mb": self.size_estimate_mb,
            "min_app_version": self.min_app_version,
            "provides": self.provides,
        }


# ── Built-in add-on definitions ──

BUILTIN_ADDONS: dict[str, AddonManifest] = {
    "tabpfn": AddonManifest(
        name="tabpfn",
        version="2.0.0",
        description="TabPFN — Prior-Data Fitted Networks for tabular classification. Requires torch (~2GB download).",
        pip_deps=["torch>=2.3", "tabpfn>=2.0", "huggingface-hub>=0.24"],
        size_estimate_mb=2500,
        min_app_version="1.0.0",
        provides=["tabpfn", "torch"],
    ),
    "sdv": AddonManifest(
        name="sdv",
        version="1.13.0",
        description="SDV — Synthetic Data Vault for generating synthetic training data (CTGAN, CopulaGAN, TVAE). Requires torch.",
        pip_deps=["torch>=2.3", "sdv>=1.13"],
        size_estimate_mb=2200,
        min_app_version="1.0.0",
        provides=["sdv", "torch"],
    ),
}


# ── Install status tracking (shared between background thread + API) ──


@dataclass
class InstallStatus:
    """Live status of an add-on installation."""

    addon: str
    state: str = "idle"  # idle, installing, succeeded, failed
    progress: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "addon": self.addon,
            "state": self.state,
            "progress": self.progress,
            "error": self.error,
        }


_install_status: dict[str, InstallStatus] = {}
_install_lock = threading.Lock()
_install_serial_lock = threading.Lock()


def get_install_status(name: str) -> dict[str, Any]:
    """Return the current install status for an add-on."""
    with _install_lock:
        status = _install_status.get(name)
        if status is None:
            return {"addon": name, "state": "idle", "progress": [], "error": None}
        return status.to_dict()


def get_addon_dir() -> Path:
    """Return the directory where add-on packages are installed."""
    settings = get_settings()
    addon_dir = settings.addon_dir
    addon_dir.mkdir(parents=True, exist_ok=True)
    return addon_dir


def get_installed_addons_file() -> Path:
    """Return the path to the installed add-ons registry file."""
    return get_addon_dir().parent / "installed.json"


def list_available_addons() -> list[dict[str, Any]]:
    """List all known add-ons with their installation status."""
    installed = get_installed_addons()
    result = []
    for name, manifest in BUILTIN_ADDONS.items():
        result.append(
            {
                **manifest.to_dict(),
                "installed": name in installed,
            }
        )
    return result


def get_installed_addons() -> dict[str, str]:
    """Return a dict of installed add-on name → version."""
    registry_file = get_installed_addons_file()
    if not registry_file.exists():
        return {}
    try:
        data: dict[str, str] = json.loads(registry_file.read_text())
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def is_addon_installed(name: str) -> bool:
    """Check if an add-on is installed."""
    return name in get_installed_addons()


def install_addon(name: str) -> dict[str, Any]:
    """Start an add-on installation in a background thread.

    Returns immediately with the initial status.  Poll
    ``get_install_status(name)`` to track progress.
    """
    if name not in BUILTIN_ADDONS:
        return {"success": False, "message": f"Unknown add-on: {name}"}

    with _install_lock:
        existing = _install_status.get(name)
        if existing and existing.state == "installing":
            return {"success": False, "message": f"{name} is already installing"}

        # Block if ANY other addon is installing (shared target dir)
        for other_name, other_status in _install_status.items():
            if other_name != name and other_status.state == "installing":
                return {
                    "success": False,
                    "message": f"Another add-on ({other_name}) is installing. Wait for it to finish.",
                }

        status = InstallStatus(addon=name, state="queued", progress=[])
        _install_status[name] = status

    thread = threading.Thread(target=_run_install, args=(name,), daemon=True)
    thread.start()

    return {"success": True, "message": f"Installation started for {name}"}


def _verify_in_subprocess(modules: list[str], addon_dir: Path) -> tuple[bool, str | None]:
    """Verify that modules can be imported in a fresh subprocess.

    Uses a subprocess so that imported DLLs are released when the process
    exits, keeping them unlocked for future reinstalls on Windows.
    """
    import os

    import_str = "; ".join(f"import {m}" for m in modules)
    env = {**os.environ, "PYTHONPATH": str(addon_dir)}
    result = subprocess.run(
        [sys.executable, "-c", import_str],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if result.returncode != 0:
        err = result.stderr.strip()[-500:] if result.stderr else "Unknown error"
        return False, err
    return True, None


def _clear_addon_dir(addon_dir: Path) -> bool:
    """Try to clear the addon dir before a fresh install.

    Returns True if successful, False if some files were locked (Windows).
    """
    import shutil

    try:
        if addon_dir.exists():
            shutil.rmtree(addon_dir)
        addon_dir.mkdir(parents=True, exist_ok=True)
        return True
    except PermissionError:
        return False


def _run_install(name: str) -> None:
    """Background install worker — runs pip and updates status."""
    manifest = BUILTIN_ADDONS[name]
    addon_dir = get_addon_dir()

    def update(msg: str) -> None:
        with _install_lock:
            status = _install_status.get(name)
            if status:
                status.progress.append(msg)
        log.info("addon.install_progress", addon=name, message=msg)

    def fail(msg: str) -> None:
        update(msg)
        with _install_lock:
            if name in _install_status:
                _install_status[name].state = "failed"
                _install_status[name].error = msg

    try:
        # Serialize installs — only one at a time (shared target directory)
        with _install_lock:
            if name in _install_status:
                _install_status[name].state = "queued"
        update("Waiting for other installations to complete...")
        _install_serial_lock.acquire()
        with _install_lock:
            if name in _install_status:
                _install_status[name].state = "installing"

        update(f"Installing {name} add-on ({manifest.size_estimate_mb} MB estimated)...")
        update(f"Packages: {', '.join(manifest.pip_deps)}")
        update("Using CPU-only torch index to minimize download size...")

        # Clear stale files from previous/failed installs
        update("Clearing previous installation files...")
        if not _clear_addon_dir(addon_dir):
            fail(
                "Cannot clear previous installation — files are locked. "
                "Restart the app and try again."
            )
            return

        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(addon_dir),
            "--no-cache-dir",
            "--extra-index-url",
            _CPU_TORCH_INDEX,
            *manifest.pip_deps,
        ]

        update("Running pip install (this may take several minutes)...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
        )

        if result.returncode != 0:
            error = result.stderr[-1000:] if result.stderr else "Unknown pip error"
            fail(f"pip failed: {error}")
            return

        # Verify in a subprocess so DLLs are released after verification
        update("Verifying installation...")
        _prepend_addon_path()
        refresh_cache()

        ok, err = _verify_in_subprocess(manifest.provides, addon_dir)
        if not ok:
            fail(f"Verification failed: {err}")
            return

        installed = get_installed_addons()
        installed[name] = manifest.version
        get_installed_addons_file().write_text(json.dumps(installed, indent=2))

        update(f"{name} add-on installed successfully!")
        with _install_lock:
            if name in _install_status:
                _install_status[name].state = "succeeded"

    except subprocess.TimeoutExpired:
        fail("Installation timed out (15 min limit)")
    except Exception as e:
        fail(f"Installation error: {e}")
    finally:
        _install_serial_lock.release()


def uninstall_addon(name: str) -> dict[str, Any]:
    """Uninstall an add-on by removing its files from the addon dir.

    Note: this removes ALL files in the addon dir (since add-ons share torch).
    In a future iteration we could track per-add-on files.
    """
    if name not in BUILTIN_ADDONS:
        return {"success": False, "message": f"Unknown add-on: {name}"}

    installed = get_installed_addons()
    if name not in installed:
        return {"success": False, "message": f"{name} is not installed"}

    addon_dir = get_addon_dir()

    other_installed = [k for k in installed if k != name]
    if not other_installed:
        import shutil

        if addon_dir.exists():
            shutil.rmtree(addon_dir)
        addon_dir.mkdir(parents=True, exist_ok=True)
    else:
        log.info("addon.uninstall_skipped_shared_deps", addon=name, others=other_installed)

    del installed[name]
    get_installed_addons_file().write_text(json.dumps(installed, indent=2))

    refresh_cache()
    return {"success": True, "message": f"{name} add-on uninstalled"}


def _prepend_addon_path() -> None:
    """Prepend the add-on directory to sys.path if not already present."""
    addon_dir = str(get_addon_dir())
    if addon_dir not in sys.path:
        sys.path.insert(0, addon_dir)


def init_addons() -> None:
    """Initialize add-ons at app startup — prepend path and log status."""
    _prepend_addon_path()
    installed = get_installed_addons()
    if installed:
        log.info("addons.loaded", addons=list(installed.keys()))
        refresh_cache()
    else:
        log.info("addons.none_installed")


def get_addon_status(name: str) -> dict[str, Any]:
    """Get detailed status for a single add-on."""
    if name not in BUILTIN_ADDONS:
        return {"success": False, "message": f"Unknown add-on: {name}"}

    manifest = BUILTIN_ADDONS[name]
    installed = is_addon_installed(name)

    # Verify in subprocess to avoid locking DLLs in the main process
    addon_dir = get_addon_dir()
    modules_available: dict[str, bool] = {}
    if installed and addon_dir.exists():
        ok, _ = _verify_in_subprocess(manifest.provides, addon_dir)
        modules_available = {mod: ok for mod in manifest.provides}
    else:
        modules_available = {mod: False for mod in manifest.provides}

    return {
        "name": name,
        "installed": installed,
        "version": manifest.version,
        "description": manifest.description,
        "size_estimate_mb": manifest.size_estimate_mb,
        "modules_available": modules_available,
        "all_modules_available": all(modules_available.values()),
    }
