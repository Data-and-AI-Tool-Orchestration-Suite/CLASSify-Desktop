"""Forecasting engine tests.

These tests require the Darts add-on (``pip install -e ".[forecast]"``)
and are skipped automatically when it is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("darts", reason="The Darts forecasting add-on is not installed")

pytestmark = pytest.mark.addon

from ml.args import ForecastArgs  # noqa: E402
from ml.forecast.engine import forecast_trainer  # noqa: E402
from ml.forecast.generate import run_forecast_generation  # noqa: E402
from storage.factory import reset_storage  # noqa: E402
from storage.local import LocalStorage  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "datasets" / "timeseries_daily.csv"

REPORT_ID = "forecast-test-report"


def _make_args(**overrides: Any) -> ForecastArgs:
    args = ForecastArgs(
        time_component="date",
        target=["sales"],
        grouping=[],
        future_covariates=["temperature"],
        past_covariates=[],
        static_covariates=[],
        train_group=["linear-regression"],
        test_size=0.2,
        lags=4,
        output_chunk_length=2,
        lags_future=0,
        epochs=1,
        visualize=True,
        report_uuid=REPORT_ID,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture()
def forecast_storage(tmp_path: Path) -> LocalStorage:
    reset_storage()
    storage = LocalStorage(tmp_path / "datasets")
    storage.put_text(f"{REPORT_ID}/original_file", FIXTURE.read_text(encoding="utf-8"))
    return storage


class TestForecastTrainer:
    def test_trains_and_writes_artifacts(self, forecast_storage: LocalStorage) -> None:
        progress: list[tuple[int, int, str]] = []
        log_lines: list[str] = []

        forecast_trainer(
            _make_args(),
            forecast_storage,
            on_progress=lambda c, t, m: progress.append((c, t, m)),
            log_cb=log_lines.append,
            cancel_token=None,
        )

        assert forecast_storage.exists(f"{REPORT_ID}/results")
        assert forecast_storage.exists(f"{REPORT_ID}/results.json")
        assert forecast_storage.exists(f"{REPORT_ID}/output_log")
        assert forecast_storage.exists(f"{REPORT_ID}/forecast_data_baseline")
        assert forecast_storage.exists(f"{REPORT_ID}/forecast_data_linear-regression")
        assert forecast_storage.exists(f"{REPORT_ID}/linear-regression_model.joblib")

        df = forecast_storage.read_csv(f"{REPORT_ID}/results")
        models = set(df["model"])
        assert "Baseline Average Prediction" in models
        assert "linear-regression" in models
        assert "Normalized MAE" in df.columns
        assert "De-Normalized MAE" in df.columns
        assert "MASE" in df.columns

        nested = json.loads(forecast_storage.get_text(f"{REPORT_ID}/results.json"))
        assert "linear-regression" in nested
        assert "Normalized MAE" in nested["linear-regression"]

        log_text = "\n".join(log_lines)
        assert "Evaluating model: linear-regression" in log_text
        assert "linear-regression training complete." in log_text

        assert len(progress) >= 2
        assert progress[-1][0] == progress[-1][1]

    def test_metrics_and_chart_data_shape(self, forecast_storage: LocalStorage) -> None:
        forecast_trainer(_make_args(), forecast_storage)

        viz = json.loads(forecast_storage.get_text(f"{REPORT_ID}/forecast_data_baseline"))
        series_block = viz[""]["sales"]
        assert len(series_block["train_y"]) > 0
        assert "val_x" in series_block
        assert len(series_block["val_x"]) == len(series_block["val_y"])
        assert len(series_block["pred_x"]) > 0

    def test_unknown_model_is_skipped(self, forecast_storage: LocalStorage) -> None:
        forecast_trainer(
            _make_args(train_group=["not-a-real-model"], visualize=False), forecast_storage
        )
        log_text = forecast_storage.get_text(f"{REPORT_ID}/output_log")
        assert "Skipping not-a-real-model" in log_text
        df = forecast_storage.read_csv(f"{REPORT_ID}/results")
        assert set(df["model"]) == {"Baseline Average Prediction"}


class TestForecastGeneration:
    def test_generates_future_forecasts(self, forecast_storage: LocalStorage) -> None:
        forecast_trainer(_make_args(visualize=False), forecast_storage)

        gen_args = _make_args(visualize=False)
        gen_args.mode = "forecast-generation"
        gen_args.forecast_horizon = 6
        gen_args.train_group = ["linear-regression"]
        run_forecast_generation(gen_args, forecast_storage)

        assert forecast_storage.exists(f"{REPORT_ID}/forecasts")
        payload = json.loads(forecast_storage.get_text(f"{REPORT_ID}/forecasts"))
        assert "linear-regression" in payload
        components = payload["linear-regression"]["components"]
        assert "sales" in components
        assert len(components["sales"]) == 6

    def test_missing_model_is_reported(self, forecast_storage: LocalStorage) -> None:
        forecast_trainer(_make_args(visualize=False), forecast_storage)

        gen_args = _make_args(visualize=False)
        gen_args.mode = "forecast-generation"
        gen_args.forecast_horizon = 3
        gen_args.train_group = ["arima"]
        with pytest.raises(ValueError, match="No forecasts were generated"):
            run_forecast_generation(gen_args, forecast_storage)
