"""Golden-dataset ML regression test: train RandomForest on a small CSV.

This is the Phase D acceptance test — verifies the engine produces the
same artifacts the web version does: report.csv, results.json (as report),
joblib model, scaler, output_log, and viz PNGs.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from ml.args import TrainingArgs
from ml.engine import trainer
from storage.local import LocalStorage


@pytest.fixture()
def golden_csv() -> str:
    """A small binary classification CSV with mixed column types."""
    data = """feature_1,feature_2,feature_3,class
3.5,10,1,1
2.1,20,0,0
4.8,15,1,1
1.9,25,0,0
3.2,12,1,1
2.8,18,0,0
4.1,14,1,1
1.5,22,0,0
3.9,11,1,1
2.3,19,0,0
5.0,16,1,1
1.7,24,0,0
3.6,13,1,1
2.5,17,0,0
4.3,15,1,1
1.8,21,0,0
3.4,14,1,1
2.2,20,0,0
4.7,12,1,1
1.6,23,0,0
"""
    return data


@pytest.fixture()
def golden_df(golden_csv: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(golden_csv))


@pytest.fixture()
def ml_storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "datasets")


@pytest.mark.ml_regression
class TestEngineGoldenDataset:
    """Train RandomForest on a tiny dataset and verify artifacts."""

    def test_train_randomforest_produces_artifacts(
        self, golden_df: pd.DataFrame, ml_storage: LocalStorage
    ) -> None:
        """End-to-end: upload → train RF → verify report + model + viz."""
        report_id = "test-report-001"

        # Save dataset to storage (simulating the upload step)
        ml_storage.write_csv(f"{report_id}/file", golden_df, index=False)

        # Configure training args
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
            disable_model_save=False,
        )

        # Run the trainer
        trainer(
            args=args,
            storage=ml_storage,
            full_dataset=golden_df.copy(),
            testset=None,
            on_progress=lambda c, t, m: None,
        )

        # ── Verify artifacts exist ──

        # Output log
        assert ml_storage.exists(f"{report_id}/output_log"), "output_log not created"
        log_text = ml_storage.get_text(f"{report_id}/output_log")
        assert "Starting Training" in log_text
        assert "randomforest" in log_text.lower() or "Evaluating model" in log_text

        # Report CSV (results)
        assert ml_storage.exists(f"{report_id}/results"), "results CSV not created"
        report_df = ml_storage.read_csv(f"{report_id}/results")
        assert "model" in report_df.columns
        assert len(report_df) >= 1
        assert report_df.iloc[0]["model"] == "randomforest"

        # Model joblib
        assert ml_storage.exists(f"{report_id}/randomforest_model.joblib"), (
            "model joblib not created"
        )

        # Scaler joblib
        assert ml_storage.exists(f"{report_id}/scaler.joblib"), "scaler joblib not created"

        # SHAP rows CSV
        assert ml_storage.exists(f"{report_id}/shap_rows_randomforest"), "SHAP rows not created"

        # Visualizations
        viz_keys = ml_storage.list(f"{report_id}/viz/")
        assert len(viz_keys) > 0, f"No visualizations created. Keys: {ml_storage.list(report_id)}"

        # At least some of these should exist:
        found_viz = [k.split("/")[-1] for k in viz_keys]
        print(f"Viz files created: {found_viz}")

    def test_train_randomforest_metrics_in_sane_range(
        self, golden_df: pd.DataFrame, ml_storage: LocalStorage
    ) -> None:
        """Verify the trained model's metrics are within expected ranges."""
        report_id = "test-report-002"
        ml_storage.write_csv(f"{report_id}/file", golden_df, index=False)

        args = TrainingArgs(
            supervised=True,
            train_group=["randomforest"],
            parameter_tune=False,
            shap_feature_explainability=False,
            visualize=False,
            test_size=0.3,
            random_state=42,
            class_column="class",
            report_uuid=report_id,
            n_jobs=1,
        )

        trainer(
            args=args,
            storage=ml_storage,
            full_dataset=golden_df.copy(),
            testset=None,
        )

        report_df = ml_storage.read_csv(f"{report_id}/results")
        row = report_df.iloc[0]

        # Accuracy should be > 0.5 (better than random) on this separable dataset
        test_acc = float(row["test_acc"])
        assert test_acc > 0.5, f"test_acc={test_acc} — expected > 0.5"

        # n_train + n_test should sum to total rows
        n_train = int(row["n_train"])
        n_test = int(row["n_test"])
        assert n_train + n_test == len(golden_df), (
            f"n_train={n_train} + n_test={n_test} != {len(golden_df)}"
        )

    def test_train_kmeans_clustering(
        self, golden_df: pd.DataFrame, ml_storage: LocalStorage
    ) -> None:
        """Train KMeans clustering and verify cluster artifacts."""
        report_id = "test-cluster-001"
        # Drop class column for unsupervised
        cluster_df = golden_df.drop(columns=["class"])
        ml_storage.write_csv(f"{report_id}/file", cluster_df, index=False)

        args = TrainingArgs(
            supervised=False,
            train_group=["kmeans"],
            parameter_tune=False,
            visualize=True,
            num_clusters=2,
            random_state=42,
            class_column=None,
            report_uuid=report_id,
            n_jobs=1,
        )

        trainer(
            args=args,
            storage=ml_storage,
            full_dataset=cluster_df.copy(),
            testset=None,
        )

        # Output log
        assert ml_storage.exists(f"{report_id}/output_log")

        # Results
        assert ml_storage.exists(f"{report_id}/results")
        report_df = ml_storage.read_csv(f"{report_id}/results")
        assert report_df.iloc[0]["model"] == "kmeans"

        # Model + scaler
        assert ml_storage.exists(f"{report_id}/kmeans_model.joblib")

        # Labeled file (cluster assignments)
        assert ml_storage.exists(f"{report_id}/labeled"), "labeled file not created"

        # Viz
        viz_keys = ml_storage.list(f"{report_id}/viz/")
        assert len(viz_keys) > 0, "No clustering visualizations created"

    def test_output_log_contains_progress(
        self, golden_df: pd.DataFrame, ml_storage: LocalStorage
    ) -> None:
        """Verify progress callback was called and output log captures it."""
        report_id = "test-progress-001"
        ml_storage.write_csv(f"{report_id}/file", golden_df, index=False)

        progress_messages: list[str] = []

        args = TrainingArgs(
            supervised=True,
            train_group=["randomforest", "logisticregression"],
            parameter_tune=False,
            shap_feature_explainability=False,
            visualize=False,
            test_size=0.3,
            random_state=42,
            class_column="class",
            report_uuid=report_id,
            n_jobs=1,
        )

        trainer(
            args=args,
            storage=ml_storage,
            full_dataset=golden_df.copy(),
            testset=None,
            on_progress=lambda c, t, m: progress_messages.append(m),
        )

        assert len(progress_messages) == 2, f"Expected 2 progress messages, got {progress_messages}"
        assert "1/2" in progress_messages[0]
        assert "2/2" in progress_messages[1]
