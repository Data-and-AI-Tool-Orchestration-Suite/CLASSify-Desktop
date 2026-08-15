"""Column type detection and dataset transformation logic.

Ported from CLASSify-2's api.py: get_column_types_internal,
createMappingColumn, change_column_types, upload_testset.
Uses ``storage`` instead of S3, returns typed dataclasses.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from charset_normalizer import detect

from storage.base import Storage


@dataclass
class ColumnInfo:
    """Detected or user-specified info for a single column."""

    column: str
    data_type: str = "string"  # float, integer, bool, categorical, string
    checked: bool = True
    missing: str = ""  # "", "drop", "constant", "synthetic"
    fill_value: str = ""
    is_class: bool = False
    categories: list[str] = field(default_factory=list)


@dataclass
class ColumnTypeResult:
    """Result of column type auto-detection."""

    data_types: dict[str, str]
    missing_values: dict[str, bool]


def detect_encoding(raw_bytes: bytes) -> str:
    """Detect the encoding of raw file bytes."""
    result = detect(raw_bytes)
    encoding = result.get("encoding") if result else None
    return encoding or "utf-8"


def read_csv_from_storage(storage: Storage, key: str, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV from storage with automatic encoding detection."""
    raw = storage.get_bytes(key)
    encoding = detect_encoding(raw)
    text = raw.decode(encoding)
    return pd.read_csv(__import__("io").StringIO(text), **kwargs)


def write_csv_to_storage(storage: Storage, key: str, df: pd.DataFrame, **kwargs: Any) -> None:
    """Write a DataFrame to storage as CSV."""
    import io

    buf = io.StringIO()
    df.to_csv(buf, **kwargs)
    storage.put_text(key, buf.getvalue())


def get_column_types_internal(df: pd.DataFrame) -> ColumnTypeResult:
    """Auto-detect column types from a DataFrame.

    Mirrors CLASSify-2's get_column_types_internal exactly:
    - Dashes as missing values → try numeric
    - float64 with all int values → integer
    - 0/1 values → bool
    - yes/no, true/false → bool (with mapping)
    - otherwise → string
    """
    columns_with_dash = df.columns[(df == "-").any()].tolist()
    if columns_with_dash:
        df.replace("-", np.nan, inplace=True)
        for col in columns_with_dash:
            with contextlib.suppress(ValueError):
                df[col] = pd.to_numeric(df[col])

    data_types = df.dtypes.apply(str).to_dict()
    missing_values: dict[str, bool] = {}

    for column_name, data_type in data_types.items():
        if data_type == "float64":
            data_types[column_name] = "float"
            if df[column_name].isnull().any() and (df[column_name].dropna() % 1 == 0).all():
                if df[column_name].dropna().isin([0, 1]).all():
                    data_types[column_name] = "bool"
                else:
                    data_types[column_name] = "integer"
        elif data_type == "int64":
            if df[column_name].isin([0, 1]).all():
                data_types[column_name] = "bool"
            else:
                data_types[column_name] = "integer"
        elif data_type == "object":
            if df[column_name].isnull().any():
                temp_column = df[column_name].dropna().astype(str)
                if temp_column.str.lower().isin(["yes", "no"]).all():
                    df[column_name] = (
                        df[column_name].astype(str).str.lower().map({"yes": 1, "no": 0})
                    )
                    data_types[column_name] = "bool"
                elif temp_column.str.lower().isin(["true", "false"]).all():
                    df[column_name] = (
                        df[column_name].astype(str).str.lower().map({"true": 1, "false": 0})
                    )
                    data_types[column_name] = "bool"
                else:
                    data_types[column_name] = "string"
            else:
                if df[column_name].astype(str).str.lower().isin(["yes", "no"]).all():
                    df[column_name] = (
                        df[column_name].astype(str).str.lower().map({"yes": 1, "no": 0})
                    )
                    data_types[column_name] = "bool"
                elif df[column_name].astype(str).str.lower().isin(["true", "false"]).all():
                    df[column_name] = (
                        df[column_name].astype(str).str.lower().map({"true": 1, "false": 0})
                    )
                    data_types[column_name] = "bool"
                else:
                    data_types[column_name] = "string"

        missing_values[column_name] = df[column_name].isnull().any()

    return ColumnTypeResult(data_types=data_types, missing_values=missing_values)


def format_float_to_string(value: float) -> str:
    """Format a float value as a string without trailing .0."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def create_mapping_column(
    dataset: pd.DataFrame, class_column: str, mapping: dict[str, int]
) -> pd.DataFrame:
    """Create an integer mapping column for a categorical class column.

    Mirrors CLASSify-2's createMappingColumn exactly.
    """
    if class_column + "_mapping" in dataset.columns:
        suffix = 0
        while True:
            if class_column + f"_mapping_feature_{suffix}" not in dataset.columns:
                dataset = dataset.rename(
                    columns={class_column + "_mapping": class_column + f"_mapping_feature_{suffix}"}
                )
                break
    dataset = dataset.rename(columns={class_column: class_column + "_mapping"})
    if pd.api.types.is_float_dtype(dataset[class_column + "_mapping"]):
        dataset[class_column + "_mapping"] = dataset[class_column + "_mapping"].apply(
            format_float_to_string
        )
    dataset[class_column + "_mapping"] = dataset[class_column + "_mapping"].astype(str)
    dataset[class_column] = dataset[class_column + "_mapping"].map(mapping)
    return dataset


class ColumnChangeError(Exception):
    """Raised when a column type change fails."""


def apply_column_changes(df: pd.DataFrame, column_changes: list[dict[str, Any]]) -> pd.DataFrame:
    """Apply user-specified column type changes to a DataFrame.

    Mirrors CLASSify-2's change_column_types logic:
    - Drop unchecked columns
    - Handle missing values (drop, constant fill, synthetic impute)
    - Cast types (float, bool, integer, categorical)
    - One-hot encode low-cardinality categoricals (reject high-cardinality)
    - Validate class column (≥2 classes)

    Returns the transformed DataFrame.  Raises ColumnChangeError on failure.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    df.index.name = "index"

    # Sort: handle categorical first so imputing works
    column_changes = sorted(column_changes, key=lambda x: x["data_type"] != "categorical")

    for row in column_changes:
        column_name = row["column"]
        if column_name == df.index.name:
            df = df.set_index(column_name)
            continue
        if column_name not in df.columns:
            if row["checked"]:
                raise ColumnChangeError(
                    f"Column {column_name} cannot be included because it was not in the original dataset."
                )
            continue

        if not row["checked"]:
            df = df.drop([column_name], axis=1)
        else:
            # Handle missing values
            if row["missing"] == "constant":
                fill_value = row["fill_value"]
                try:
                    df[column_name] = df[column_name].fillna(fill_value)
                except Exception as e:
                    raise ColumnChangeError(
                        f"Cannot fill column {column_name} with value {fill_value}"
                    ) from e
            elif row["missing"] == "drop":
                df = df.dropna(subset=[column_name])
            elif row["missing"] == "synthetic":
                if row["data_type"] == "categorical":
                    df[column_name] = df[column_name].fillna(df[column_name].mode()[0])
                else:
                    columns = df.columns
                    missing_indices = df.isna().stack()[df.isna().stack()].index.tolist()
                    new = IterativeImputer().fit_transform(df)
                    if len(columns) != len(new[0]):
                        raise ColumnChangeError(
                            "Cannot synthetically fill columns with no real values."
                        )
                    df = pd.DataFrame(new, columns=columns)
                    for df_row, col in missing_indices:
                        if col != column_name:
                            df.at[df_row, col] = np.nan

            # Cast types
            try:
                if row["data_type"] in ("float", "bool"):
                    df[column_name] = df[column_name].astype(row["data_type"])
                elif row["data_type"] == "integer":
                    df[column_name] = df[column_name].round().astype(int)
                else:  # categorical
                    if not row.get("is_class"):
                        column_ratio = len(df[column_name].unique()) / len(df[column_name])
                        if column_ratio <= 0.1:
                            one_hot_encoded = pd.get_dummies(df[column_name])
                            encoding_columns = one_hot_encoded.columns.tolist()
                            row["categories"] = encoding_columns
                            one_hot_encoded = one_hot_encoded.add_prefix(column_name + "_")
                            df = pd.concat([df, one_hot_encoded], axis=1)
                            df = df.drop([column_name], axis=1)
                        else:
                            raise ColumnChangeError(
                                f"Column {column_name} has too many categories to be one-hot encoded. "
                                "Drop this column or encode as integer instead."
                            )
                    else:
                        if f"{column_name}_mapping" in df.columns:
                            row["categories"] = df[column_name + "_mapping"].unique().tolist()
            except ColumnChangeError:
                raise
            except Exception as e:
                if "Unable to allocate" in str(e):
                    raise ColumnChangeError(
                        f"Column {column_name} has too many categories to be one-hot encoded. "
                        "Drop this column or encode as integer instead."
                    ) from e
                raise ColumnChangeError(
                    f"Cannot convert column {column_name} to type {row['data_type']}"
                ) from e

            # Validate class column
            if row.get("is_class"):
                num_classes = len(df[column_name].unique())
                if num_classes < 2:
                    raise ColumnChangeError("The class column must have more than one class.")

            if column_name.lower() == "index":
                try:
                    df.set_index(column_name, inplace=True)
                    df.index.name = "index"
                except Exception:
                    df = df.drop(["index"], axis=1)

    if len(df) == 0:
        raise ColumnChangeError(
            "All rows dropped due to missing values. Choose alternate fill method."
        )
    if df.index.name is None:
        df.index.name = "index"
    return df


def validate_testset(
    train_df: pd.DataFrame, test_df: pd.DataFrame, class_column: str | None = None
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Validate and prepare a test set against a training set.

    Mirrors CLASSify-2's upload_testset logic:
    - Drop index columns
    - Validate columns match training set
    - Handle unexpected categories in one-hot encoded columns
    - Return (test_X, test_y) or (test_df, None) if no class column
    """
    if "index" in test_df.columns:
        test_df = test_df.drop(["index"], axis=1)
    if "Index" in test_df.columns:
        test_df = test_df.drop(["Index"], axis=1)

    if class_column and class_column in test_df.columns:
        test_y = test_df[class_column]
        test_X = test_df.drop([class_column], axis=1)
    else:
        test_X = test_df
        test_y = None

    # Validate columns match
    train_columns = set(train_df.columns) - ({class_column} if class_column else set())
    test_columns = set(test_X.columns)

    missing_from_test = train_columns - test_columns
    if missing_from_test:
        raise ColumnChangeError(
            f"Test set is missing columns: {', '.join(sorted(missing_from_test))}"
        )

    extra_in_test = test_columns - train_columns
    if extra_in_test:
        raise ColumnChangeError(
            f"Test set has unexpected columns: {', '.join(sorted(extra_in_test))}"
        )

    return test_X, test_y
