"""Tests for system router endpoints: first-run, metric-defs, update check, usage."""

from __future__ import annotations

from fastapi.testclient import TestClient

from classify_api.db import reset_engine, run_migrations
from classify_api.main import create_app
from classify_api.settings import reset_settings
from storage.factory import reset_storage


def _setup(tmp_data_dir: object) -> TestClient:
    reset_settings()
    reset_engine()
    reset_storage()
    run_migrations()
    return TestClient(create_app())


class TestFirstRunEndpoints:
    def test_first_run_status(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            resp = client.get("/api/system/first-run")
        assert resp.status_code == 200
        body = resp.json()
        assert body["first_run"] is True
        assert "data_dir" in body
        assert body["disk_free_bytes"] > 0
        assert body["cpu_count"] >= 1

    def test_complete_first_run(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            # Initially first run
            assert client.get("/api/system/first-run").json()["first_run"] is True
            # Complete it
            resp = client.post("/api/system/first-run/complete")
            assert resp.status_code == 200
            # No longer first run
            assert client.get("/api/system/first-run").json()["first_run"] is False

    def test_complete_first_run_idempotent(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            client.post("/api/system/first-run/complete")
            client.post("/api/system/first-run/complete")
            assert client.get("/api/system/first-run").json()["first_run"] is False


class TestMetricDefs:
    def test_metric_definitions(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            resp = client.get("/api/system/metric-defs")
        assert resp.status_code == 200
        defs = resp.json()
        assert "test_auc" in defs
        assert "test_acc" in defs
        assert "silhouette_score" in defs
        assert len(defs["test_auc"]) > 10  # Has a meaningful description
        assert "ROC" in defs["test_auc"]


class TestUpdateCheck:
    def test_check_updates_dev_mode(self, tmp_data_dir: object) -> None:
        """In dev mode, update check should return an error gracefully."""
        from classify_api.settings import get_settings

        client = _setup(tmp_data_dir)
        with client:
            settings = get_settings()
            if settings.dev_mode:
                resp = client.get("/api/system/check-updates")
                assert resp.status_code == 200
                body = resp.json()
                assert body["error"] is not None
                assert "dev mode" in body["error"]
