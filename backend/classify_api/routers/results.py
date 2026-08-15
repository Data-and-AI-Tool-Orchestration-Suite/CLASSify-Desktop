"""Results & re-test endpoints — serves data for the results detail page.

Tabs: Results Table, Visualizations, Download Data, Re-Test Models,
Prediction Insights (SHAP), Output Log.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from classify_api import repositories as repo
from classify_api.db import get_session
from classify_api.schemas.results import (
    OutputLogResponse,
    PrepareParamsResponse,
    ResultsResponse,
    RetestRequest,
    RetestResponse,
    ShapRowsResponse,
    VizListResponse,
)
from ml.column_types import detect_encoding
from ml.retest import retest
from ml.shap_explain import get_shap_row_graph
from storage.base import KeyNotFound
from storage.factory import get_storage

router = APIRouter()


def _get_report_or_404(db: Session, report_id: str) -> Any:
    report = repo.get_report(db, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Dataset {report_id} not found")
    return report


# ── G1: Results table (report.csv) ──


@router.get("/{report_id}", response_model=ResultsResponse)
def get_results(
    report_id: str,
    db: Session = Depends(get_session),
) -> ResultsResponse:
    """Get the results table (report.csv) for a trained dataset."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    try:
        df: pd.DataFrame = storage.read_csv(f"{report_id}/results")
    except KeyNotFound:
        return ResultsResponse(success=False)

    columns = df.columns.tolist()
    rows = df.to_dict(orient="records")

    # Try to read results.json if it exists
    results_json: dict[str, Any] | None = None
    if storage.exists(f"{report_id}/results.json"):
        try:
            import json

            results_json = json.loads(storage.get_text(f"{report_id}/results.json"))
        except Exception:
            pass

    return ResultsResponse(
        success=True,
        report_csv=rows,
        columns=columns,
        results_json=results_json,
    )


# ── G2: Visualizations ──


@router.get("/{report_id}/viz", response_model=VizListResponse)
def list_visualizations(
    report_id: str,
    db: Session = Depends(get_session),
) -> VizListResponse:
    """List all visualization PNGs for a dataset."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    viz_keys = storage.list(f"{report_id}/viz/")
    # Strip the prefix for the frontend
    viz_names = [k.split("viz/")[-1] for k in viz_keys]

    return VizListResponse(success=True, visualizations=viz_names)


@router.get("/{report_id}/viz/{viz_name}")
def get_visualization(
    report_id: str,
    viz_name: str,
    db: Session = Depends(get_session),
) -> StreamingResponse:
    """Serve a single visualization PNG."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    try:
        png_bytes = storage.get_bytes(f"{report_id}/viz/{viz_name}")
    except KeyNotFound:
        raise HTTPException(status_code=404, detail="Visualization not found") from None

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


# ── G3: SHAP rows ──


@router.get("/{report_id}/shap-rows/{model}", response_model=ShapRowsResponse)
def get_shap_rows(
    report_id: str,
    model: str,
    db: Session = Depends(get_session),
) -> ShapRowsResponse:
    """Get the per-row SHAP CSV for a model."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    try:
        df: pd.DataFrame = storage.read_csv(f"{report_id}/shap_rows_{model}")
    except KeyNotFound:
        return ShapRowsResponse(success=False)

    columns = df.columns.tolist()
    rows = df.to_dict(orient="records")

    return ShapRowsResponse(success=True, rows=rows, columns=columns)


# ── G4: SHAP row graph ──


@router.get("/{report_id}/shap-row-graph")
def get_shap_row_graph_endpoint(
    report_id: str,
    model: str = Query(...),
    row_num: int = Query(...),
    train_test: str = Query("test"),
    class_column: str = Query("class"),
    db: Session = Depends(get_session),
) -> StreamingResponse:
    """Generate a per-row SHAP impact bar chart PNG."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    png_bytes = get_shap_row_graph(storage, report_id, model, row_num, train_test, class_column)
    if png_bytes is None:
        raise HTTPException(status_code=404, detail="SHAP data not found for this model/row")

    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")


# ── G5: Re-test ──


@router.post("/{report_id}/retest", response_model=RetestResponse)
async def retest_models(
    report_id: str,
    request: RetestRequest,
    testset: UploadFile = File(...),
    db: Session = Depends(get_session),
) -> RetestResponse:
    """Re-test saved models on a new testset."""
    _get_report_or_404(db, report_id)

    if not testset.filename or not testset.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    raw = await testset.read()
    encoding = detect_encoding(raw)
    try:
        test_df = pd.read_csv(io.StringIO(raw.decode(encoding)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}") from e

    storage = get_storage()

    # Save the testset
    csv_buf = io.StringIO()
    test_df.to_csv(csv_buf, index=False)
    storage.put_text(f"{report_id}/retest", csv_buf.getvalue())

    try:
        result = retest(
            storage=storage,
            model_names=request.model_names,
            testset_key=f"{report_id}/retest",
            class_column=request.class_column,
            dataset_prefix=report_id,
        )
        return RetestResponse(success=result["success"], message=result["message"])
    except Exception as e:
        error_str = str(e)
        if (
            "features, but MinMaxScaler is expecting" in error_str
            or "features, but StandardScaler is expecting" in error_str
        ):
            import re

            numbers = list(map(int, re.findall(r"\d+", error_str)))
            if len(numbers) <= 2:
                error_str = f"Uploaded test set has {numbers[0]} features, but original training set had {numbers[1]} features."
        return RetestResponse(success=False, message=error_str)


# ── G6: Output log ──


@router.get("/{report_id}/output-log", response_model=OutputLogResponse)
def get_output_log(
    report_id: str,
    db: Session = Depends(get_session),
) -> OutputLogResponse:
    """Get the training output log."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    try:
        log_text = storage.get_text(f"{report_id}/output_log")
    except KeyNotFound:
        return OutputLogResponse(success=False, log="")

    return OutputLogResponse(success=True, log=log_text)


# ── G7: Download artifact ──


@router.get("/{report_id}/download")
def download_artifact(
    report_id: str,
    suffix: str = Query(
        ..., description="Artifact key suffix (e.g. 'file', 'results', 'randomforest_model.joblib')"
    ),
    db: Session = Depends(get_session),
) -> StreamingResponse:
    """Download any artifact from storage."""
    _get_report_or_404(db, report_id)

    storage = get_storage()
    key = f"{report_id}/{suffix}"
    try:
        data = storage.get_bytes(key)
    except KeyNotFound:
        raise HTTPException(status_code=404, detail=f"Artifact '{suffix}' not found") from None

    # Determine media type
    if suffix.endswith(".joblib"):
        media_type = "application/octet-stream"
    elif suffix.endswith(".csv") or suffix in (
        "results",
        "file",
        "original_file",
        "testset",
        "retest",
        "labeled",
        "retest_results",
    ):
        media_type = "text/csv"
    elif suffix.endswith(".json") or suffix in (
        "results.json",
        "progress.json",
        "metadata",
        "synthetic_metrics",
    ):
        media_type = "application/json"
    elif suffix.startswith("viz/"):
        media_type = "image/png"
    elif suffix in ("output_log", "logisticregression_odds_ratio"):
        media_type = "text/plain"
    else:
        media_type = "application/octet-stream"

    filename = suffix.replace("/", "_")
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── G8: Prepare params (rerun with previous) ──


@router.get("/{report_id}/prepare-params", response_model=PrepareParamsResponse)
def get_prepare_params(
    report_id: str,
    db: Session = Depends(get_session),
) -> PrepareParamsResponse:
    """Get previous training parameters for rerun-with-same-params."""
    _get_report_or_404(db, report_id)

    job = repo.get_job_by_report(db, report_id)
    if job is None or job.args is None:
        return PrepareParamsResponse(success=False)

    args = job.args
    class_column = args.get("class_column")

    return PrepareParamsResponse(
        success=True,
        parameters=args,
        class_column=class_column,
    )
