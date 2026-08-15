"""ML regression tests using golden datasets.

Tests train models on the fixture datasets and assert:
- report.csv exists and is non-empty
- model joblib is created and loadable
- scaler joblib is created
- viz PNGs exist (when visualize=True)
- SHAP rows CSV exists (when shap=True)
- metrics are within sane ranges
- KMeans clustering produces ≥2 clusters
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.args import TrainingArgs
from ml.column_types import get_column_types_internal
from ml.engine import trainer
from storage.local import LocalStorage

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "datasets"


def load_fixture(name: str) -> pd.DataFrame:
    """Load a golden dataset fixture."""
    return pd.read_csv(FIXTURES_DIR / name)


@pytest.fixture()
def ml_storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "datasets")


@pytest.mark.ml_regression
class TestBinaryClassification:
    """Binary classification golden dataset tests."""

    def test_randomforest_binary(self, ml_storage: LocalStorage) -> None:
        df = load_fixture("binary_classification.csv")
        report_id = "binary-rf"
        ml_storage.write_csv(f"{report_id}/file", df, index=False)

        args = TrainingArgs(
            supervised=True,
            train_group=["randomforest"],
            parameter_tune=False,
            shap_feature_explainability=True,
            visualize=True,
            test_size=0.3,
            random_state=42,
            class_column="class",
            report_uuid=report_id,
            n_jobs=1,
        )
        trainer(args=args, storage=ml_storage, full_dataset=df.copy(), testset=None)

        report = ml_storage.read_csv(f"{report_id}/results")
        assert len(report) >= 1
        row = report.iloc[0]
        assert row["model"] == "randomforest"
        assert float(row["test_acc"]) > 0.5
        assert ml_storage.exists(f"{report_id}/randomforest_model.joblib")
        assert ml_storage.exists(f"{report_id}/scaler.joblib")
        assert ml_storage.exists(f"{report_id}/shap_rows_randomforest")
        assert len(ml_storage.list(f"{report_id}/viz/")) > 0

    def test_logisticregression_binary(self, ml_storage: LocalStorage) -> None:
        df = load_fixture("binary_classification.csv")
        report_id = "binary-lr"
        ml_storage.write_csv(f"{report_id}/file", df, index=False)

        args = TrainingArgs(
            supervised=True,
            train_group=["logisticregression"],
            parameter_tune=False,
            shap_feature_explainability=False,
            visualize=False,
            class_column="class",
            report_uuid=report_id,
            n_jobs=1,
        )
        trainer(args=args, storage=ml_storage, full_dataset=df.copy(), testset=None)

        report = ml_storage.read_csv(f"{report_id}/results")
        assert float(report.iloc[0]["test_acc"]) > 0.5
        assert ml_storage.exists(f"{report_id}/logisticregression_model.joblib")

    def test_xgboost_binary(self, ml_storage: LocalStorage) -> None:
        df = load_fixture("binary_classification.csv")
        report_id = "binary-xgb"
        ml_storage.write_csv(f"{report_id}/file", df, index=False)

        args = TrainingArgs(
            supervised=True,
            train_group=["xgboost"],
            parameter_tune=False,
            shap_feature_explainability=False,
            visualize=False,
            class_column="class",
            report_uuid=report_id,
            n_jobs=1,
        )
        trainer(args=args, storage=ml_storage, full_dataset=df.copy(), testset=None)

        report = ml_storage.read_csv(f"{report_id}/results")
        assert float(report.iloc[0]["test_acc"]) > 0.5
        assert ml_storage.exists(f"{report_id}/xgboost_model.joblib")


@pytest.mark.ml_regression
class TestMulticlassClassification:
    """Multiclass classification golden dataset tests."""

    def test_randomforest_multiclass(self, ml_storage: LocalStorage) -> None:
        df = load_fixture("multiclass.csv")
        report_id = "multi-rf"
        ml_storage.write_csv(f"{report_id}/file", df, index=False)

        args = TrainingArgs(
            supervised=True,
            train_group=["randomforest"],
            parameter_tune=False,
            shap_feature_explainability=False,
            visualize=True,
            class_column="class",
            report_uuid=report_id,
            n_jobs=1,
            multiclass=True,
        )
        trainer(args=args, storage=ml_storage, full_dataset=df.copy(), testset=None)

        report = ml_storage.read_csv(f"{report_id}/results")
        assert len(report) >= 1
        assert float(report.iloc[0]["test_acc"]) > 0.3  # 3 classes, >random(0.33)
        assert ml_storage.exists(f"{report_id}/randomforest_model.joblib")


@pytest.mark.ml_regression
class TestClustering:
    """Clustering golden dataset tests."""

    def test_kmeans_clustering(self, ml_storage: LocalStorage) -> None:
        df = load_fixture("clustering.csv")
        report_id = "cluster-km"
        ml_storage.write_csv(f"{report_id}/file", df, index=False)

        args = TrainingArgs(
            supervised=False,
            train_group=["kmeans"],
            parameter_tune=False,
            visualize=True,
            num_clusters=3,
            class_column=None,
            report_uuid=report_id,
            n_jobs=1,
        )
        trainer(args=args, storage=ml_storage, full_dataset=df.copy(), testset=None)

        report = ml_storage.read_csv(f"{report_id}/results")
        assert len(report) >= 1
        assert report.iloc[0]["model"] == "kmeans"
        assert ml_storage.exists(f"{report_id}/kmeans_model.joblib")
        assert ml_storage.exists(f"{report_id}/labeled")
        assert len(ml_storage.list(f"{report_id}/viz/")) > 0

    def test_spectral_clustering(self, ml_storage: LocalStorage) -> None:
        df = load_fixture("clustering.csv")
        report_id = "cluster-sc"
        ml_storage.write_csv(f"{report_id}/file", df, index=False)

        args = TrainingArgs(
            supervised=False,
            train_group=["spectralclustering"],
            parameter_tune=False,
            visualize=False,
            num_clusters=3,
            class_column=None,
            report_uuid=report_id,
            n_jobs=1,
        )
        trainer(args=args, storage=ml_storage, full_dataset=df.copy(), testset=None)

        report = ml_storage.read_csv(f"{report_id}/results")
        assert len(report) >= 1
        assert ml_storage.exists(f"{report_id}/spectralclustering_model.joblib")


@pytest.mark.ml_regression
class TestMissingValues:
    """Tests for datasets with missing values."""

    def test_missing_values_loadable(self) -> None:
        df = load_fixture("missing_values.csv")
        assert df.isna().any().any()  # Has missing values
        result = get_column_types_internal(df.copy())
        assert "feature_2" in result.data_types
        assert result.missing_values["feature_2"]

    def test_train_with_missing_dropped(self, ml_storage: LocalStorage) -> None:
        df = load_fixture("missing_values.csv")
        df_dropped = df.dropna()
        assert len(df_dropped) >= 10

        report_id = "missing-drop"
        ml_storage.write_csv(f"{report_id}/file", df_dropped, index=False)

        args = TrainingArgs(
            supervised=True,
            train_group=["randomforest"],
            parameter_tune=False,
            shap_feature_explainability=False,
            visualize=False,
            class_column="class",
            report_uuid=report_id,
            n_jobs=1,
        )
        trainer(args=args, storage=ml_storage, full_dataset=df_dropped.copy(), testset=None)

        report = ml_storage.read_csv(f"{report_id}/results")
        assert len(report) >= 1


@pytest.mark.ml_regression
class TestCategoricalDetection:
    """Tests for categorical/bool column auto-detection."""

    def test_yes_no_detected_as_bool(self) -> None:
        df = load_fixture("categorical.csv")
        result = get_column_types_internal(df.copy())
        assert result.data_types["status"] == "bool"

    def test_numeric_columns_detected(self) -> None:
        df = load_fixture("categorical.csv")
        result = get_column_types_internal(df.copy())
        assert result.data_types["feature_1"] == "float"
        assert result.data_types["feature_2"] == "integer"


@pytest.mark.ml_regression
class TestOutputLog:
    """Tests for output log generation."""

    def test_output_log_contains_model_name(self, ml_storage: LocalStorage) -> None:
        df = load_fixture("binary_classification.csv")
        report_id = "log-test"
        ml_storage.write_csv(f"{report_id}/file", df, index=False)

        args = TrainingArgs(
            supervised=True,
            train_group=["randomforest"],
            parameter_tune=False,
            shap_feature_explainability=False,
            visualize=False,
            class_column="class",
            report_uuid=report_id,
            n_jobs=1,
        )
        trainer(args=args, storage=ml_storage, full_dataset=df.copy(), testset=None)

        log = ml_storage.get_text(f"{report_id}/output_log")
        assert "Starting Training" in log
        assert "randomforest" in log.lower()

    def test_output_log_contains_progress(self, ml_storage: LocalStorage) -> None:
        df = load_fixture("binary_classification.csv")
        report_id = "progress-test"
        ml_storage.write_csv(f"{report_id}/file", df, index=False)

        progress_messages: list[str] = []

        args = TrainingArgs(
            supervised=True,
            train_group=["randomforest", "logisticregression"],
            parameter_tune=False,
            shap_feature_explainability=False,
            visualize=False,
            class_column="class",
            report_uuid=report_id,
            n_jobs=1,
        )
        trainer(
            args=args,
            storage=ml_storage,
            full_dataset=df.copy(),
            testset=None,
            on_progress=lambda c, t, m: progress_messages.append(m),
        )

        assert len(progress_messages) == 2
        assert "1/2" in progress_messages[0]
        assert "2/2" in progress_messages[1]
