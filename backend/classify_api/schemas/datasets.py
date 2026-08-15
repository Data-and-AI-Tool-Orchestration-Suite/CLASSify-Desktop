"""Dataset request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ColumnChangeRequest(BaseModel):
    """A single column change from the column preview modal."""

    column: str
    data_type: str = "string"
    checked: bool = True
    missing: str = ""
    fill_value: str = ""
    is_class: bool = False


class ColumnChangesRequest(BaseModel):
    """POST /api/datasets/{id}/column-changes."""

    data_types: list[ColumnChangeRequest] = Field(default_factory=list)


class ClassMappingRequest(BaseModel):
    """POST /api/datasets/{id}/class-mapping."""

    class_column: str
    mapping: dict[str, int]


class CommentUpdate(BaseModel):
    """PATCH /api/datasets/{id}/comment."""

    comments: str


class DatasetListResponse(BaseModel):
    """GET /api/datasets — DataTable-style response."""

    draw: int = 0
    recordsTotal: int = 0
    recordsFiltered: int = 0
    data: list[dict[str, Any]] = Field(default_factory=list)


class DatasetUploadResponse(BaseModel):
    """POST /api/datasets/upload."""

    success: bool
    report_id: str | None = None
    filename: str | None = None
    data_types: dict[str, str] = Field(default_factory=dict)
    missing_values: dict[str, bool] = Field(default_factory=dict)
    message: str | None = None


class ColumnChangesResponse(BaseModel):
    """Response for column-changes endpoint."""

    success: bool
    message: str = ""
    data_types: list[dict[str, Any]] = Field(default_factory=list)


class ClassValuesResponse(BaseModel):
    """GET /api/datasets/{id}/class-values."""

    success: bool
    class_values: list[str] = Field(default_factory=list)


class SuccessResponse(BaseModel):
    """Generic success/error response."""

    success: bool
    message: str = ""
