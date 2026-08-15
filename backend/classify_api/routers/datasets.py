"""Dataset processing endpoints — upload, column-changes, class-values,
class-mapping, testset, duplicate, delete, list, comment.

Ported from CLASSify-2's api.py and ReportsController.php.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from classify_api import repositories as repo
from classify_api.db import get_session
from classify_api.schemas.datasets import (
    ClassMappingRequest,
    ClassValuesResponse,
    ColumnChangesRequest,
    ColumnChangesResponse,
    CommentUpdate,
    DatasetListResponse,
    DatasetUploadResponse,
    SuccessResponse,
)
from classify_api.settings import Settings, get_settings
from ml.column_types import (
    ColumnChangeError,
    apply_column_changes,
    create_mapping_column,
    detect_encoding,
    get_column_types_internal,
)
from storage.factory import get_storage

router = APIRouter()


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename: spaces→underscores, strip extension, limit length."""
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_\-]", "", name)
    if len(name) > 100:
        name = name[:100]
    return name


def _get_report_or_404(db: Session, report_id: str) -> Any:
    report = repo.get_report(db, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Dataset {report_id} not found")
    return report


# ── E1: Upload ──


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> DatasetUploadResponse:
    """Upload a CSV file, auto-detect column types, create a Report."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_mb} MB",
        )

    encoding = detect_encoding(raw)
    try:
        text = raw.decode(encoding)
        df = pd.read_csv(StringIO(text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}") from e

    filename = _sanitize_filename(file.filename)

    # Save original data BEFORE detection (get_column_types_internal modifies df in-place)
    original_csv = df.to_csv(index=False)

    # Auto-detect column types (modifies df in-place — e.g. yes/no → 0/1)
    result = get_column_types_internal(df.copy())

    # Check if filename already exists; if so, append a suffix
    base_filename = filename
    suffix = 1
    while repo.is_report_present(db, filename):
        filename = f"{base_filename}_{suffix}"
        suffix += 1

    # Create report
    report = repo.create_report(
        db,
        filename=filename,
        original_filename=file.filename,
        status="Preview",
    )
    db.commit()

    # Save to storage (original unmodified data)
    storage = get_storage()
    storage.put_text(f"{report.uuid}/file", original_csv)
    storage.put_text(f"{report.uuid}/original_file", original_csv)

    return DatasetUploadResponse(
        success=True,
        report_id=report.uuid,
        filename=filename,
        data_types=result.data_types,
        missing_values=result.missing_values,
    )


# ── E2: Column changes ──


@router.post("/{report_id}/column-changes", response_model=ColumnChangesResponse)
def column_changes(
    report_id: str,
    request: ColumnChangesRequest,
    db: Session = Depends(get_session),
) -> ColumnChangesResponse:
    """Apply user-specified column type changes to the dataset."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    try:
        raw = storage.get_bytes(f"{report_id}/file")
        encoding = detect_encoding(raw)
        df = pd.read_csv(StringIO(raw.decode(encoding)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {e}") from e

    try:
        changes_data = [c.model_dump() for c in request.data_types]
        df = apply_column_changes(df, changes_data)
    except ColumnChangeError as e:
        return ColumnChangesResponse(success=False, message=str(e), data_types=changes_data)

    # Save transformed dataset
    csv_buf = StringIO()
    df.to_csv(csv_buf)
    storage.put_text(f"{report_id}/file", csv_buf.getvalue())

    # Update report status
    repo.update_report_status(db, report_id, "Uploaded")
    repo.update_report_column_changes(db, report_id, {"changes": changes_data})
    db.commit()

    return ColumnChangesResponse(
        success=True, message="Uploaded to storage", data_types=changes_data
    )


# ── E3: Class values ──


@router.get("/{report_id}/class-values", response_model=ClassValuesResponse)
def get_class_values(
    report_id: str,
    class_column: str = Query(..., description="Name of the class column"),
    db: Session = Depends(get_session),
) -> ClassValuesResponse:
    """Get unique class label values for the mapping modal."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    try:
        raw = storage.get_bytes(f"{report_id}/file")
        encoding = detect_encoding(raw)
        df = pd.read_csv(StringIO(raw.decode(encoding)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {e}") from e

    if class_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{class_column}' not found in dataset")

    unique_values = df[class_column].dropna().unique().tolist()
    class_values = [str(v) for v in unique_values]

    return ClassValuesResponse(success=True, class_values=class_values)


# ── E4: Class mapping ──


@router.post("/{report_id}/class-mapping", response_model=SuccessResponse)
def set_class_mapping(
    report_id: str,
    request: ClassMappingRequest,
    db: Session = Depends(get_session),
) -> SuccessResponse:
    """Create an integer mapping column for a categorical class column."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    try:
        raw = storage.get_bytes(f"{report_id}/file")
        encoding = detect_encoding(raw)
        df = pd.read_csv(StringIO(raw.decode(encoding)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {e}") from e

    try:
        mapping_int = {k: int(v) for k, v in request.mapping.items()}
        df = create_mapping_column(df, request.class_column, mapping_int)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create mapping: {e}") from e

    csv_buf = StringIO()
    df.to_csv(csv_buf, index=False)
    storage.put_text(f"{report_id}/file", csv_buf.getvalue())

    return SuccessResponse(success=True, message="Class mapping applied")


# ── E5: Testset upload ──


@router.post("/{report_id}/testset", response_model=SuccessResponse)
async def upload_testset(
    report_id: str,
    file: UploadFile = File(...),
    class_column: str = Query(None),
    db: Session = Depends(get_session),
) -> SuccessResponse:
    """Upload a separate test set CSV."""
    _get_report_or_404(db, report_id)

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    raw = await file.read()
    encoding = detect_encoding(raw)
    try:
        text = raw.decode(encoding)
        test_df = pd.read_csv(StringIO(text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}") from e

    storage = get_storage()
    csv_buf = StringIO()
    test_df.to_csv(csv_buf, index=False)
    storage.put_text(f"{report_id}/testset", csv_buf.getvalue())

    return SuccessResponse(success=True, message="Testset uploaded")


# ── E6: Duplicate ──


@router.post("/{report_id}/duplicate", response_model=SuccessResponse)
def duplicate_dataset(
    report_id: str,
    db: Session = Depends(get_session),
) -> SuccessResponse:
    """Duplicate a dataset (copies the file + original_file)."""
    report = _get_report_or_404(db, report_id)

    # Generate a new filename with suffix
    base = report.filename
    suffix = 1
    while repo.is_report_present(db, f"{base}_copy_{suffix}"):
        suffix += 1
    new_filename = f"{base}_copy_{suffix}"

    new_report = repo.create_report(
        db,
        filename=new_filename,
        original_filename=report.original_filename,
        status="Preview",
    )
    db.commit()

    storage = get_storage()
    if storage.exists(f"{report_id}/file"):
        storage.copy(f"{report_id}/file", f"{new_report.uuid}/file")
    if storage.exists(f"{report_id}/original_file"):
        storage.copy(f"{report_id}/original_file", f"{new_report.uuid}/original_file")

    return SuccessResponse(success=True, message=f"Duplicated as {new_filename}")


# ── E7: Delete ──


@router.delete("/{report_id}", response_model=SuccessResponse)
def delete_dataset(
    report_id: str,
    db: Session = Depends(get_session),
) -> SuccessResponse:
    """Delete a dataset and all its artifacts from storage."""
    report = _get_report_or_404(db, report_id)

    # Delete from storage (keep original_file if other copies exist)
    storage = get_storage()
    all_keys = storage.list(f"{report_id}/")

    has_copies = False
    if report.original_filename:
        all_reports = repo.list_reports(db, length=10000)
        for r in all_reports:
            if r.uuid != report_id and r.original_filename == report.original_filename:
                has_copies = True
                break

    for key in all_keys:
        if key.endswith("/original_file") and has_copies:
            continue
        storage.delete(key)

    # Delete from DB
    repo.delete_report(db, report_id)
    db.commit()

    return SuccessResponse(success=True, message="Dataset deleted")


# ── E8: List (DataTable-style) ──


@router.get("", response_model=DatasetListResponse)
def list_datasets(
    db: Session = Depends(get_session),
    start: int = Query(0, ge=0),
    length: int = Query(100, ge=1, le=10000),
    search: str = Query(""),
    order_by: str = Query("created_at"),
    order_dir: str = Query("desc"),
    draw: int = Query(0),
) -> DatasetListResponse:
    """List datasets in DataTables format."""
    total = repo.count_reports(db)
    filtered = repo.count_reports_filtered(db, search)

    reports = repo.list_reports(
        db,
        start=start,
        length=length,
        search=search,
        order_by=order_by,
        order_dir=order_dir,
    )

    data = [repo.serialize_report(r) for r in reports]

    return DatasetListResponse(
        draw=draw,
        recordsTotal=total,
        recordsFiltered=filtered,
        data=data,
    )


# ── E9: Comment ──


@router.patch("/{report_id}/comment", response_model=SuccessResponse)
def update_comment(
    report_id: str,
    request: CommentUpdate,
    db: Session = Depends(get_session),
) -> SuccessResponse:
    """Update the comments on a dataset."""
    _get_report_or_404(db, report_id)
    repo.update_report_comments(db, report_id, request.comments)
    db.commit()
    return SuccessResponse(success=True, message="Comment updated")


# ── E10: Get single dataset ──


@router.get("/{report_id}", response_model=dict)
def get_dataset(
    report_id: str,
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Get details of a single dataset."""
    report = _get_report_or_404(db, report_id)
    return repo.serialize_report(report)
