"""Synthetic data generation — ported from CLASSify-2's synthesize.py.

SDV is a torch-gated addon.  All SDV imports are guarded behind
``backends.require('sdv')`` so the base engine runs without torch.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from ml.backends import sdv_available
from storage.base import Storage


class SynthesisError(Exception):
    """Raised when synthetic data generation fails."""


def get_metadata(real_data: pd.DataFrame) -> dict:
    """Build SDV metadata from a DataFrame's dtypes."""
    type_map: dict[str, dict[str, str]] = {
        "int64": {"sdtype": "numerical", "computer_representation": "Int64"},
        "float64": {"sdtype": "numerical", "computer_representation": "Float"},
        "bool": {"sdtype": "boolean"},
    }
    metadata: dict[str, Any] = {"columns": {}}
    for col in real_data.columns:
        data_type = str(real_data[col].dtype)
        metadata["columns"][col] = type_map.get(data_type, {"sdtype": "numerical"})
    return metadata


def train_model(model_name: str, real_data: pd.DataFrame, metadata_obj: Any) -> Any:
    """Train an SDV synthesizer model."""
    from sdv.lite import SingleTablePreset
    from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer, TVAESynthesizer

    if model_name == "tabular":
        model = SingleTablePreset(metadata_obj, name="FAST_ML")
    elif model_name == "ctgan":
        model = CTGANSynthesizer(metadata_obj)
    elif model_name == "copulagan":
        model = CopulaGANSynthesizer(metadata_obj)
    elif model_name == "tvae":
        model = TVAESynthesizer(metadata_obj)
    else:
        model = SingleTablePreset(metadata_obj, name="FAST_ML")

    model.fit(real_data)
    return model


def get_conditions(size: int) -> list:
    """Create balanced conditions for binary classification."""
    from sdv.sampling import Condition

    num_rows_false = math.floor(size / 2)
    num_rows_true = num_rows_false
    if num_rows_false + num_rows_true < size:
        num_rows_true += 1
    return [
        Condition({"class": True}, num_rows=num_rows_false),
        Condition({"class": False}, num_rows=num_rows_true),
    ]


def get_conditions_multiclass(size: int, classes: list) -> list:
    """Create balanced conditions for multiclass."""
    from sdv.sampling import Condition

    size_class = int(size / len(classes))
    return [Condition({"class": mclass}, num_rows=size_class) for mclass in classes]


def get_balance(real_data: pd.DataFrame) -> tuple:
    """Determine if data is balanced and what to synthesize for binary."""
    from sdv.sampling import Condition

    is_balanced = False
    false_count = len(real_data) - real_data["class"].sum()
    true_count = len(real_data) - false_count
    if false_count == true_count:
        is_balanced = True
    make_type = True
    make_count = false_count - true_count
    if true_count > false_count:
        make_type = False
        make_count = true_count - false_count
    conditions = [Condition({"class": make_type}, num_rows=int(make_count))]
    return conditions, make_count, is_balanced


def get_balance_multiclass(real_data: pd.DataFrame) -> tuple:
    """Determine if data is balanced and what to synthesize for multiclass."""
    from sdv.sampling import Condition

    is_balanced = False
    classes = real_data["class"].unique().tolist()
    counts = {c: real_data["class"].value_counts()[c] for c in classes}
    if len(set(counts.values())) == 1:
        is_balanced = True
    max_class = max(counts, key=lambda k: counts[k])
    lower_counts = {k: v for k, v in counts.items() if k != max_class}
    conditions = []
    make_count = 0
    for mclass, count in lower_counts.items():
        diff = counts[max_class] - count
        make_count += diff
        conditions.append(Condition({"class": mclass}, num_rows=diff))
    return conditions, make_count, is_balanced


def get_report(
    real_data: pd.DataFrame,
    metadata_obj: Any,
    model: Any,
    num_rows: int | None = None,
    conditions: list | None = None,
) -> tuple:
    """Generate synthetic data and quality report."""
    from sdmetrics.reports.single_table import QualityReport

    if num_rows is None:
        num_rows = real_data.shape[0]
    if conditions is not None:
        synthetic_data = model.sample_from_conditions(
            conditions=conditions, max_tries_per_batch=1000
        )
    else:
        synthetic_data = model.sample(num_rows=num_rows)
    synthetic_data = synthetic_data.sample(frac=1).reset_index(drop=True)

    metadata_dict = metadata_obj.to_dict()
    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata_dict)

    details = report.get_details(property_name="Column Shapes")
    dataset_metrics: dict[str, Any] = {
        "overall_score": report.get_score(),
        "size": int(num_rows),
        "columns": [],
    }
    for _, row in details.iterrows():
        dataset_metrics["columns"].append(
            {
                "name": row[0],
                "metric": row[1],
                "quality_score": row[2],
            }
        )
    return dataset_metrics, synthetic_data


def build_dataset(
    args: Any, real_data: pd.DataFrame, storage: Storage, filename_key: str
) -> pd.DataFrame:
    """Generate synthetic data to augment or replace the training set.

    Mirrors CLASSify-2's build_dataset function.
    Requires the SDV addon (torch).
    """
    if not sdv_available():
        raise SynthesisError(
            "Synthetic data generation requires the SDV addon. Install it via Settings → Add-ons."
        )

    from sdv.metadata import SingleTableMetadata

    metadata = get_metadata(real_data)
    metadata_obj = SingleTableMetadata.load_from_dict(metadata)
    storage.put_text(f"{filename_key}/metadata", json.dumps(metadata))

    size_list = [real_data.shape[0]]
    all_metrics: dict[str, list] = {}
    return_df = pd.DataFrame()
    model_name = (
        args.synthesize_model[0]
        if isinstance(args.synthesize_model, list)
        else args.synthesize_model
    )
    model = train_model(model_name, real_data, metadata_obj)

    if args.multiclass:
        conditions, make_size, is_balanced = get_balance_multiclass(real_data)
    else:
        conditions, make_size, is_balanced = get_balance(real_data)

    if not is_balanced and args.synthesize_original:
        dataset_metrics, synthetic_data = get_report(
            real_data, metadata_obj, model, num_rows=make_size, conditions=conditions
        )
        real_data_balanced = pd.concat([real_data, synthetic_data], axis=0)
        real_data_balanced = real_data_balanced.sample(frac=1).reset_index(drop=True)
        storage.write_csv(f"{filename_key}/synthetic_balanced", real_data_balanced, index=False)
        return_df = real_data_balanced
        all_metrics[model_name] = [dataset_metrics]
    elif is_balanced and args.synthesize_original:
        return_df = real_data

    if args.synthesize_new:
        for size in size_list:
            if args.multiclass:
                classes = real_data["class"].unique().tolist()
                conditions = get_conditions_multiclass(size, classes)
            else:
                conditions = get_conditions(size)
            dataset_metrics, synthetic_data = get_report(
                real_data, metadata_obj, model, num_rows=size, conditions=conditions
            )
            storage.write_csv(f"{filename_key}/synthetic_new", synthetic_data, index=False)
            return_df = synthetic_data
            if model_name not in all_metrics:
                all_metrics[model_name] = [dataset_metrics]
            else:
                all_metrics[model_name].append(dataset_metrics)

    storage.put_text(
        f"{filename_key}/synthetic_metrics", json.dumps(all_metrics, default=_convert_int)
    )
    return return_df


def _convert_int(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.float64):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
