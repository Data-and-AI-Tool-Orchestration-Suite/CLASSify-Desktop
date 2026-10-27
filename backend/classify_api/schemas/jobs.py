"""Job schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    """A single job's status."""

    id: str
    report_uuid: str
    state: str
    args: dict[str, Any] | None = None
    progress: int = 0
    progress_total: int = 0
    progress_message: str | None = None
    error: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class JobListResponse(BaseModel):
    """List of jobs."""

    jobs: list[JobResponse] = Field(default_factory=list)


class TrainRequest(BaseModel):
    """POST /api/jobs — start training.

    ``mode`` selects the engine: "classification" (supervised or
    unsupervised) or "forecasting" (time series via the Darts addon).
    """

    report_id: str
    options: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "classification"
