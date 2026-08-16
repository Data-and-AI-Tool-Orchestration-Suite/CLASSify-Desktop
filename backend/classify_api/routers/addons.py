"""Add-on endpoints — list, install, uninstall, status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from classify_api.schemas.addons import (
    AddonInfo,
    AddonInstallResponse,
    AddonListResponse,
    AddonStatusResponse,
)
from classify_api.services.addon_service import (
    BUILTIN_ADDONS,
    get_addon_status,
    get_install_status,
    install_addon,
    list_available_addons,
    uninstall_addon,
)
from ml.backends import list_addons

router = APIRouter()

BUILTIN_ADDON_NAMES = set(BUILTIN_ADDONS.keys())


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
