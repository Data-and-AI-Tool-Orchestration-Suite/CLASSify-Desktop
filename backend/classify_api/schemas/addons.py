"""Add-on schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AddonInfo(BaseModel):
    """Info about a single add-on."""

    name: str
    version: str
    description: str
    pip_deps: list[str] = Field(default_factory=list)
    size_estimate_mb: int = 0
    min_app_version: str = "1.0.0"
    provides: list[str] = Field(default_factory=list)
    installed: bool = False


class AddonListResponse(BaseModel):
    """List of all add-ons."""

    addons: list[AddonInfo] = Field(default_factory=list)


class AddonStatusResponse(BaseModel):
    """Detailed status for a single add-on."""

    name: str
    installed: bool
    version: str
    description: str
    size_estimate_mb: int
    modules_available: dict[str, bool] = Field(default_factory=dict)
    all_modules_available: bool = False


class AddonInstallResponse(BaseModel):
    """Response from install/uninstall."""

    success: bool
    message: str
