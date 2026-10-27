"""Integration tests for forecasting-mode API endpoints.

These tests do not require the Darts add-on: the forecasting job start
endpoint is tested with the add-on availability monkeypatched.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from classify_api.db import reset_engine, run_migrations
from classify_api.main import create_app
from classify_api.settings import reset_settings
from storage.factory import get_storage, reset_storage

TS_CSV = b"""date,temperature,sales
2023-01-01,12.5,101.0
2023-01-02,11.1,98.2
2023-01-03,13.8,104.4
2023-01-04,14.2,110.1
2023-01-05,12.9,99.9
2023-01-06,10.5,95.5
2023-01-07,11.8,100.0
2023-01-08,13.1,105.7
2023-01-09,14.9,111.3
2023-01-10,15.2,112.8
"""

ROLES = {
    "roles": [
        {"column": "date", "role": "time-component"},
        {"column": "sales", "role": "target"},
        {"column": "temperature", "role": "future-covariate"},
    ]
}


def _setup(tmp_data_dir: object) -> tuple[TestClient, str]:
    reset_settings()
    reset_engine()
    reset_storage()
    run_migrations()
    app = create_app()
    client = TestClient(app)
    with client:
        resp = client.post(
            "/api/datasets/upload",
            files={"file": ("ts.csv", io.BytesIO(TS_CSV), "text/csv")},
        )
        report_id = resp.json()["report_id"]
    return client, report_id


class TestTsRoles:
    def test_save_roles_sets_forecasting_mode(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            resp = client.post(f"/api/datasets/{report_id}/ts-roles", json=ROLES)
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is True
            assert len(body["roles"]) == 3

            dataset = client.get(f"/api/datasets/{report_id}").json()
            assert dataset["mode"] == "forecasting"
            assert dataset["status"] == "Uploaded"

            saved = dataset["column_changes"]["ts_roles"]
            assert {"column": "date", "role": "time-component"} in saved

    def test_requires_exactly_one_time_component(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            resp = client.post(
                f"/api/datasets/{report_id}/ts-roles",
                json={
                    "roles": [
                        {"column": "sales", "role": "target"},
                        {"column": "temperature", "role": "future-covariate"},
                    ]
                },
            )
            assert resp.status_code == 400
            assert "exactly one" in resp.json()["detail"]

            resp = client.post(
                f"/api/datasets/{report_id}/ts-roles",
                json={
                    "roles": [
                        {"column": "date", "role": "time-component"},
                        {"column": "sales", "role": "time-component"},
                        {"column": "temperature", "role": "target"},
                    ]
                },
            )
            assert resp.status_code == 400

    def test_requires_at_least_one_target(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            resp = client.post(
                f"/api/datasets/{report_id}/ts-roles",
                json={
                    "roles": [
                        {"column": "date", "role": "time-component"},
                        {"column": "temperature", "role": "future-covariate"},
                    ]
                },
            )
            assert resp.status_code == 400
            assert "target" in resp.json()["detail"]

    def test_rejects_unknown_column(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            resp = client.post(
                f"/api/datasets/{report_id}/ts-roles",
                json={
                    "roles": [
                        {"column": "date", "role": "time-component"},
                        {"column": "nope", "role": "target"},
                    ]
                },
            )
            assert resp.status_code == 400

    def test_classification_changes_set_classification_mode(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            client.post(f"/api/datasets/{report_id}/ts-roles", json=ROLES)
            changes = {
                "data_types": [
                    {
                        "column": "date",
                        "data_type": "string",
                        "checked": True,
                        "missing": "",
                        "fill_value": "",
                        "is_class": True,
                    },
                    {
                        "column": "sales",
                        "data_type": "float",
                        "checked": True,
                        "missing": "",
                        "fill_value": "",
                        "is_class": False,
                    },
                    {
                        "column": "temperature",
                        "data_type": "float",
                        "checked": True,
                        "missing": "",
                        "fill_value": "",
                        "is_class": False,
                    },
                ]
            }
            resp = client.post(f"/api/datasets/{report_id}/column-changes", json=changes)
            assert resp.status_code == 200
            dataset = client.get(f"/api/datasets/{report_id}").json()
            assert dataset["mode"] == "classification"


class TestForecastJobStart:
    def test_requires_ts_roles(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs",
                json={"report_id": report_id, "options": [], "mode": "forecasting"},
            )
            assert resp.status_code == 400
            assert "time-series configuration" in resp.json()["detail"]

    def test_requires_addon(self, tmp_data_dir: object, monkeypatch: pytest.MonkeyPatch) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            client.post(f"/api/datasets/{report_id}/ts-roles", json=ROLES)
            monkeypatch.setattr("classify_api.routers.jobs.is_available", lambda name: False)
            resp = client.post(
                "/api/jobs",
                json={"report_id": report_id, "options": [], "mode": "forecasting"},
            )
            assert resp.status_code == 400
            assert "Forecasting add-on" in resp.json()["detail"]

    def test_enqueues_forecasting_job(
        self, tmp_data_dir: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            client.post(f"/api/datasets/{report_id}/ts-roles", json=ROLES)
            monkeypatch.setattr(
                "classify_api.routers.jobs.is_available", lambda name: name == "darts"
            )
            options = [
                {"name": "train_group", "value": "linear-regression"},
                {"name": "test_size", "value": "0.2"},
                {"name": "lags", "value": "4"},
                {"name": "output_chunk_length", "value": "2"},
                {"name": "epochs", "value": "1"},
            ]
            resp = client.post(
                "/api/jobs",
                json={"report_id": report_id, "options": options, "mode": "forecasting"},
            )
            assert resp.status_code == 200
            job = resp.json()
            assert job["state"] == "queued"
            args = job["args"]
            assert args["mode"] == "forecasting"
            assert args["time_component"] == "date"
            assert args["target"] == ["sales"]
            assert args["future_covariates"] == ["temperature"]
            assert args["train_group"] == ["linear-regression"]

            dataset = client.get(f"/api/datasets/{report_id}").json()
            assert dataset["status"] == "Queued"

            cancel = client.post(f"/api/jobs/{job['id']}/cancel")
            assert cancel.status_code == 200


class TestForecastGenerationEndpoint:
    def test_requires_models(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs/forecast-generation",
                data={"report_id": report_id, "forecast_horizon": "6", "models": ""},
            )
            assert resp.status_code == 400

    def test_enqueues_generation_job(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            client.post(f"/api/datasets/{report_id}/ts-roles", json=ROLES)
            resp = client.post(
                "/api/jobs/forecast-generation",
                data={
                    "report_id": report_id,
                    "forecast_horizon": "6",
                    "models": "linear-regression,arima",
                },
            )
            assert resp.status_code == 200
            job = resp.json()
            assert job["args"]["mode"] == "forecast-generation"
            assert job["args"]["forecast_horizon"] == 6
            assert job["args"]["train_group"] == ["linear-regression", "arima"]
            assert job["args"]["time_component"] == "date"
            assert job["args"]["target"] == ["sales"]
            assert job["args"]["future_covariates"] == ["temperature"]

            cancel = client.post(f"/api/jobs/{job['id']}/cancel")
            assert cancel.status_code == 200

    def test_requires_ts_roles(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs/forecast-generation",
                data={
                    "report_id": report_id,
                    "forecast_horizon": "6",
                    "models": "linear-regression",
                },
            )
            assert resp.status_code == 400
            assert "time-series configuration" in resp.json()["detail"]

    def test_future_covariates_upload_rejected_for_non_csv(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs/forecast-generation",
                data={
                    "report_id": report_id,
                    "forecast_horizon": "6",
                    "models": "linear-regression",
                },
                files={"future_covariates": ("cov.txt", io.BytesIO(b"x"), "text/plain")},
            )
            assert resp.status_code == 400


class TestForecastArtifactEndpoints:
    def test_forecast_data_and_forecasts_endpoints(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        storage = get_storage()
        storage.put_text(
            f"{report_id}/forecast_data_linear-regression",
            json.dumps({"": {"sales": {"train_x": [], "train_y": [], "pred_x": [], "pred_y": []}}}),
        )
        storage.put_text(f"{report_id}/forecasts", json.dumps({"arima": {"components": {}}}))

        with client:
            resp = client.get(f"/api/results/{report_id}/forecast-data/linear-regression")
            assert resp.status_code == 200
            assert "sales" in resp.json()[""]

            resp = client.get(f"/api/results/{report_id}/forecast-data/missing-model")
            assert resp.status_code == 404

            resp = client.get(f"/api/results/{report_id}/forecasts")
            assert resp.status_code == 200
            assert "arima" in resp.json()

            resp = client.get(
                f"/api/results/{report_id}/download",
                params={"suffix": "forecast_data_linear-regression"},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("application/json")

            resp = client.get(f"/api/results/{report_id}/download", params={"suffix": "forecasts"})
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("application/json")


class TestForecastOptions:
    def test_ml_options_forecasting(self, tmp_data_dir: object) -> None:
        client, _report_id = _setup(tmp_data_dir)
        with client:
            resp = client.get("/api/jobs/ml-options/forecasting")
            assert resp.status_code == 200
            options = resp.json()
            assert options["test_size"]["default"] == 0.2
            assert options["lags"]["default"] == 12
            assert options["arima_p"]["default"] == 12
            assert "tft" in options["train_group"]["default"]
            assert options["train_group"]["models"] == options["train_group"]["default"]

    def test_metric_defs_include_forecasting(self, tmp_data_dir: object) -> None:
        client, _report_id = _setup(tmp_data_dir)
        with client:
            resp = client.get("/api/system/metric-defs")
            defs = resp.json()
            assert "Normalized MAE" in defs
            assert "De-Normalized RMSE" in defs
            assert "MASE" in defs


class TestRetrainDataUpload:
    def test_upload_retrain_file(self, tmp_data_dir: object) -> None:
        client, report_id = _setup(tmp_data_dir)
        with client:
            resp = client.post(
                f"/api/datasets/{report_id}/retrain-data",
                files={"file": ("new.csv", io.BytesIO(TS_CSV), "text/csv")},
            )
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            assert get_storage().exists(f"{report_id}/retrain_file")

            resp = client.post(
                f"/api/datasets/{report_id}/retrain-data",
                files={"file": ("new.txt", io.BytesIO(b"x"), "text/plain")},
            )
            assert resp.status_code == 400
