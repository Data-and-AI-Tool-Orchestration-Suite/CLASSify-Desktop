"""Add-on endpoints — list, install, uninstall, status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from classify_api.schemas.addons import (
    AddonConfigResponse,
    AddonConfigUpdate,
    AddonInfo,
    AddonInstallResponse,
    AddonListResponse,
    AddonStatusResponse,
)
from classify_api.services.addon_service import (
    BUILTIN_ADDONS,
    get_addon_settings,
    get_addon_status,
    get_install_status,
    install_addon,
    list_available_addons,
    set_addon_setting,
    uninstall_addon,
)
from ml.backends import list_addons

router = APIRouter()

BUILTIN_ADDON_NAMES = set(BUILTIN_ADDONS.keys())

# Settings keys users may configure per add-on, e.g. tabpfn -> tabpfn_token
ADDON_SETTING_KEYS: dict[str, set[str]] = {"tabpfn": {"tabpfn_token"}}


@router.get("", response_model=AddonListResponse)
def list_addons_endpoint() -> AddonListResponse:
    """List all available add-ons with installation status."""
    addons_data = list_available_addons()
    addons = [
        AddonInfo(
            name=a["name"],
            version=a["version"],
            description=a["description"],
            pip_deps=a.get("pip_deps", []),
            size_estimate_mb=a.get("size_estimate_mb", 0),
            min_app_version=a.get("min_app_version", "1.0.0"),
            provides=a.get("provides", []),
            installed=a.get("installed", False),
        )
        for a in addons_data
    ]
    return AddonListResponse(addons=addons)


@router.get("/{name}/status", response_model=AddonStatusResponse)
def addon_status(name: str) -> AddonStatusResponse:
    """Get detailed status for a single add-on."""
    if name not in BUILTIN_ADDON_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown add-on: {name}")
    status = get_addon_status(name)
    return AddonStatusResponse(**status)


@router.post("/{name}/install", response_model=AddonInstallResponse)
def install_addon_endpoint(name: str) -> AddonInstallResponse:
    """Start an add-on installation (runs in background — poll install-status)."""
    if name not in BUILTIN_ADDON_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown add-on: {name}")
    result = install_addon(name)
    return AddonInstallResponse(success=result["success"], message=result["message"])


@router.get("/{name}/install-status")
def install_status(name: str) -> dict[str, Any]:
    """Poll the live status of an in-progress add-on installation."""
    if name not in BUILTIN_ADDON_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown add-on: {name}")
    return get_install_status(name)


@router.post("/{name}/uninstall", response_model=AddonInstallResponse)
def uninstall_addon_endpoint(name: str) -> AddonInstallResponse:
    """Uninstall an add-on."""
    result = uninstall_addon(name)
    return AddonInstallResponse(success=result["success"], message=result["message"])


@router.get("/modules/check")
def check_modules() -> dict[str, Any]:
    """Check which optional ML modules are currently importable."""
    return {"modules": list_addons()}


@router.get("/{name}/config", response_model=AddonConfigResponse)
def get_addon_config(name: str) -> AddonConfigResponse:
    """Get add-on configuration state (secret values are never returned)."""
    if name not in BUILTIN_ADDON_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown add-on: {name}")
    allowed = ADDON_SETTING_KEYS.get(name, set())
    stored = get_addon_settings()
    settings_state = {key: bool(stored.get(key)) for key in sorted(allowed)}
    return AddonConfigResponse(name=name, settings=settings_state)


@router.put("/{name}/config", response_model=AddonConfigResponse)
def update_addon_config(name: str, update: AddonConfigUpdate) -> AddonConfigResponse:
    """Update add-on settings (e.g. store an API key for model weights)."""
    if name not in BUILTIN_ADDON_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown add-on: {name}")
    allowed = ADDON_SETTING_KEYS.get(name, set())
    for key, value in update.settings.items():
        if key not in allowed:
            raise HTTPException(status_code=400, detail=f"Unknown setting: {key}")
        set_addon_setting(key, value.strip())
    return get_addon_config(name)
