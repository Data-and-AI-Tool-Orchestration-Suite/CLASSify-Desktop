"""System schemas."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class InfoResponse(BaseModel):
    app: str
    version: str
    os: str
    os_version: str
    arch: str
    python: str
    data_dir: str
    dev_mode: bool


class UsageResponse(BaseModel):
    total: int
    used: int
    free: int
