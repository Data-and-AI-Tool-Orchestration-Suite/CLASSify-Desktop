"""Results schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResultsResponse(BaseModel):
    """GET /api/results/{report_id} — metrics table + results JSON."""

    success: bool
    report_csv: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    results_json: dict[str, Any] | None = None


class VizListResponse(BaseModel):
    """GET /api/results/{report_id}/viz — list of visualization names."""

    success: bool
    visualizations: list[str] = Field(default_factory=list)


class ShapRowsResponse(BaseModel):
    """GET /api/results/{report_id}/shap-rows/{model}."""

    success: bool
    rows: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)


class RetestRequest(BaseModel):
    """POST /api/results/{report_id}/retest."""

    model_names: list[str]
    class_column: str = "class"


class RetestResponse(BaseModel):
    """Response for retest."""

    success: bool
    message: str = ""


class OutputLogResponse(BaseModel):
    """GET /api/results/{report_id}/output-log."""

    success: bool
    log: str = ""


class PrepareParamsResponse(BaseModel):
    """GET /api/datasets/{id}/prepare-params — for rerun with same params."""

    success: bool
    parameters: dict[str, Any] | None = None
    class_column: str | None = None


class RunInfo(BaseModel):
    """A single training run (current or archived)."""

    job_id: str
    state: str
    created_at: str | None = None
    is_current: bool = False
    args: dict[str, Any] | None = None


class RunListResponse(BaseModel):
    """GET /api/results/{report_id}/runs."""

    success: bool
    runs: list[RunInfo] = Field(default_factory=list)
