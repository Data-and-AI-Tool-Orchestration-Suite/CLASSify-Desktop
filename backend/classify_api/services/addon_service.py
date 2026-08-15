"""Add-on installer service — manages torch-gated optional ML packages.

Add-ons (TabPFN, SDV) pull torch (~2GB) and are NOT included in the base
installer.  Users install them on demand via Settings → Add-ons or the
API.  Installed add-ons live in ``<appdata>/addons/pythonlibs`` and are
prepended to ``sys.path`` at boot so the ML engine can import them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from classify_api.settings import get_settings
from ml.backends import refresh_cache

log = structlog.get_logger()


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


def install_addon(name: str, on_progress: Any = None) -> dict[str, Any]:
    """Install an add-on by pip-installing its deps into the addon dir.

    Args:
        name: Add-on name (must be in BUILTIN_ADDONS)
        on_progress: Optional callback(message: str) for progress updates

    Returns: {"success": bool, "message": str}
    """
    if name not in BUILTIN_ADDONS:
        return {"success": False, "message": f"Unknown add-on: {name}"}

    manifest = BUILTIN_ADDONS[name]
    addon_dir = get_addon_dir()

    def log_progress(msg: str) -> None:
        log.info("addon.install_progress", addon=name, message=msg)
        if on_progress:
            on_progress(msg)

    log_progress(f"Installing {name} add-on ({manifest.size_estimate_mb} MB estimated)...")
    log_progress(f"Installing packages: {', '.join(manifest.pip_deps)}")

    try:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(addon_dir),
            "--no-deps" if name == "tabpfn" else "--upgrade",
        ]

        if name == "tabpfn":
            cmd.append("--no-deps")

        cmd.extend(manifest.pip_deps)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            error = result.stderr[-500:] if result.stderr else "Unknown pip error"
            log_progress(f"pip failed: {error}")
            return {"success": False, "message": f"pip install failed: {error}"}

        # Also install dependencies (for --no-deps case)
        if name == "tabpfn":
            log_progress("Installing torch and huggingface-hub dependencies...")
            dep_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(addon_dir),
                    "torch>=2.3",
                    "huggingface-hub>=0.24",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if dep_result.returncode != 0:
                log_progress(f"Warning: dependency install had issues: {dep_result.stderr[-200:]}")

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Installation timed out (10 min limit)"}
    except Exception as e:
        return {"success": False, "message": f"Installation error: {e}"}

    # Verify the import works
    log_progress("Verifying installation...")
    _prepend_addon_path()
    refresh_cache()

    for module in manifest.provides:
        try:
            __import__(module)
        except ImportError as e:
            log_progress(f"Verification failed for {module}: {e}")
            return {
                "success": False,
                "message": f"Module {module} could not be imported after install",
            }

    # Register as installed
    installed = get_installed_addons()
    installed[name] = manifest.version
    get_installed_addons_file().write_text(json.dumps(installed, indent=2))

    log_progress(f"{name} add-on installed successfully!")
    return {"success": True, "message": f"{name} add-on installed"}


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

    # Check if other add-ons are still installed (they share torch)
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

    from ml.backends import is_available

    modules_available = {mod: is_available(mod) for mod in manifest.provides}

    return {
        "name": name,
        "installed": installed,
        "version": manifest.version,
        "description": manifest.description,
        "size_estimate_mb": manifest.size_estimate_mb,
        "modules_available": modules_available,
        "all_modules_available": all(modules_available.values()),
    }
