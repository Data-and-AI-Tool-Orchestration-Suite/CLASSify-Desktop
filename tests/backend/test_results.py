"""Integration tests for the results router."""

from __future__ import annotations

import io
import time

from fastapi.testclient import TestClient

from classify_api.db import reset_engine, run_migrations
from classify_api.main import create_app
from classify_api.settings import reset_settings
from storage.factory import reset_storage

SMALL_CSV = b"""feature_1,feature_2,class
3.5,10,1
2.1,20,0
4.8,15,1
1.9,25,0
3.2,12,1
2.8,18,0
4.1,14,1
1.5,22,0
3.9,11,1
2.3,19,0
5.0,16,1
1.7,24,0
3.6,13,1
2.5,17,0
4.3,15,1
1.8,21,0
3.4,14,1
2.2,20,0
4.7,12,1
1.6,23,0
"""


def _setup() -> TestClient:
    """Create app with migrations, return TestClient (not yet entered)."""
    reset_settings()
    reset_engine()
    reset_storage()
    run_migrations()
    return TestClient(create_app())


def _upload_and_train(client: TestClient) -> str:
    """Upload, configure, and train a dataset. Returns report_id."""
    upload = client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
    )
    report_id = upload.json()["report_id"]

    changes = {
        "data_types": [
            {
                "column": "feature_1",
                "data_type": "float",
                "checked": True,
                "missing": "",
                "fill_value": "",
                "is_class": False,
            },
            {
                "column": "feature_2",
                "data_type": "integer",
                "checked": True,
                "missing": "",
                "fill_value": "",
                "is_class": False,
            },
            {
                "column": "class",
                "data_type": "integer",
                "checked": True,
                "missing": "",
                "fill_value": "",
                "is_class": True,
            },
        ]
    }
    client.post(f"/api/datasets/{report_id}/column-changes", json=changes)

    resp = client.post(
        "/api/jobs",
        json={
            "report_id": report_id,
            "options": [
                {"name": "supervised", "value": "True"},
                {"name": "train_group", "value": "randomforest"},
                {"name": "parameter_tune", "value": "False"},
                {"name": "shap_feature_explainability", "value": "True"},
                {"name": "visualize", "value": "True"},
                {"name": "test_size", "value": "0.3"},
                {"name": "random_state", "value": "42"},
                {"name": "class_column", "value": "class"},
            ],
        },
    )
    job_id = resp.json()["id"]

    for _ in range(120):
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["state"] in ("succeeded", "failed"):
            break
        time.sleep(1)

    assert status["state"] == "succeeded", f"Job failed: {status.get('error')}"
    return report_id


class TestGetResults:
    def test_get_results_table(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            report_id = _upload_and_train(client)
            resp = client.get(f"/api/results/{report_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["report_csv"]) >= 1
        assert "model" in body["columns"]
        assert body["report_csv"][0]["model"] == "randomforest"

    def test_get_results_nonexistent(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            resp = client.get("/api/results/nonexistent")
        assert resp.status_code == 404

    def test_get_results_untrained(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            upload = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload.json()["report_id"]
            resp = client.get(f"/api/results/{report_id}")
        assert resp.status_code == 200
        assert resp.json()["success"] is False


class TestVisualizations:
    def test_list_visualizations(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            report_id = _upload_and_train(client)
            resp = client.get(f"/api/results/{report_id}/viz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["visualizations"]) > 0

    def test_get_visualization_png(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            report_id = _upload_and_train(client)
            list_resp = client.get(f"/api/results/{report_id}/viz")
            viz_name = list_resp.json()["visualizations"][0]
            resp = client.get(f"/api/results/{report_id}/viz/{viz_name}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 0

    def test_get_nonexistent_viz(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            resp = client.get("/api/results/nonexistent/viz/nope.png")
        assert resp.status_code == 404


class TestShapRows:
    def test_get_shap_rows(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            report_id = _upload_and_train(client)
            resp = client.get(f"/api/results/{report_id}/shap-rows/randomforest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert len(body["rows"]) > 0

    def test_get_shap_rows_nonexistent_model(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            report_id = _upload_and_train(client)
            resp = client.get(f"/api/results/{report_id}/shap-rows/nonexistent")
        assert resp.status_code == 200
        assert resp.json()["success"] is False


class TestOutputLog:
    def test_get_output_log(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            report_id = _upload_and_train(client)
            resp = client.get(f"/api/results/{report_id}/output-log")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "Starting Training" in body["log"]


class TestDownload:
    def test_download_results_csv(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            report_id = _upload_and_train(client)
            resp = client.get(f"/api/results/{report_id}/download?suffix=results")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    def test_download_model_joblib(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            report_id = _upload_and_train(client)
            resp = client.get(f"/api/results/{report_id}/download?suffix=randomforest_model.joblib")
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_download_nonexistent(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            resp = client.get("/api/results/nonexistent/download?suffix=nope")
        assert resp.status_code == 404


class TestPrepareParams:
    def test_get_prepare_params(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            report_id = _upload_and_train(client)
            resp = client.get(f"/api/results/{report_id}/prepare-params")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["parameters"] is not None
        assert body["class_column"] == "class"

    def test_get_prepare_params_no_job(self, tmp_data_dir: object) -> None:
        client = _setup()
        with client:
            upload = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload.json()["report_id"]
            resp = client.get(f"/api/results/{report_id}/prepare-params")
        assert resp.status_code == 200
        assert resp.json()["success"] is False
