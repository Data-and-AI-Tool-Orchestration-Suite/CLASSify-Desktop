"""Integration tests for the job runner: enqueue, run, cancel, crash recovery."""

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


def _setup_and_upload(tmp_data_dir: object) -> tuple[TestClient, str]:
    """Create app, run migrations, upload a dataset, return (client, report_id)."""
    reset_settings()
    reset_engine()
    reset_storage()
    run_migrations()
    app = create_app()
    client = TestClient(app)
    with client:
        resp = client.post(
            "/api/datasets/upload",
            files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
        )
        report_id = resp.json()["report_id"]

        # Apply column changes to make it training-ready
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
    return client, report_id


class TestJobSubmission:
    def test_start_training(self, tmp_data_dir: object) -> None:
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [
                        {"name": "supervised", "value": "True"},
                        {"name": "train_group", "value": "randomforest"},
                        {"name": "parameter_tune", "value": "False"},
                        {"name": "shap_feature_explainability", "value": "False"},
                        {"name": "visualize", "value": "False"},
                        {"name": "test_size", "value": "0.3"},
                        {"name": "random_state", "value": "42"},
                        {"name": "class_column", "value": "class"},
                    ],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "queued"
        assert body["report_uuid"] == report_id

    def test_start_training_nonexistent_report(self, tmp_data_dir: object) -> None:
        client, _ = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs",
                json={"report_id": "nonexistent", "options": []},
            )
        assert resp.status_code == 404

    def test_get_job_status(self, tmp_data_dir: object) -> None:
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            create_resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [{"name": "supervised", "value": "True"}],
                },
            )
            job_id = create_resp.json()["id"]
            resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job_id

    def test_get_job_not_found(self, tmp_data_dir: object) -> None:
        client, _ = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.get("/api/jobs/nonexistent")
        assert resp.status_code == 404

    def test_list_jobs(self, tmp_data_dir: object) -> None:
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            client.post(
                "/api/jobs",
                json={"report_id": report_id, "options": []},
            )
            resp = client.get("/api/jobs")
        assert resp.status_code == 200
        assert len(resp.json()["jobs"]) >= 1


class TestMLOptions:
    def test_supervised_options(self, tmp_data_dir: object) -> None:
        client, _ = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.get("/api/jobs/ml-options/supervised")
        assert resp.status_code == 200
        body = resp.json()
        assert "train_group" in body
        assert "parameter_tune" in body
        assert "test_size" in body

    def test_unsupervised_options(self, tmp_data_dir: object) -> None:
        client, _ = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.get("/api/jobs/ml-options/unsupervised")
        assert resp.status_code == 200
        body = resp.json()
        assert "num_clusters" in body
        assert "clustering_parameter_goal" in body


class TestJobExecution:
    def test_job_runs_to_completion(self, tmp_data_dir: object) -> None:
        """Full end-to-end: upload → configure → submit → wait → verify results."""
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            # Submit training job
            resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [
                        {"name": "supervised", "value": "True"},
                        {"name": "train_group", "value": "randomforest"},
                        {"name": "parameter_tune", "value": "False"},
                        {"name": "shap_feature_explainability", "value": "False"},
                        {"name": "visualize", "value": "False"},
                        {"name": "test_size", "value": "0.3"},
                        {"name": "random_state", "value": "42"},
                        {"name": "class_column", "value": "class"},
                    ],
                },
            )
            job_id = resp.json()["id"]

            # Poll for completion (max 60 seconds)
            for _ in range(60):
                status_resp = client.get(f"/api/jobs/{job_id}")
                state = status_resp.json()["state"]
                if state in ("succeeded", "failed"):
                    break
                time.sleep(1)

            assert state == "succeeded", (
                f"Job ended in state: {state}, error: {status_resp.json().get('error')}"
            )

            # Verify report status updated
            report_resp = client.get(f"/api/datasets/{report_id}")
            assert report_resp.json()["status"] == "Processed"

    def test_cancel_queued_job(self, tmp_data_dir: object) -> None:
        """Cancel a job that's still in the queue."""
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            # Submit a job
            resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [{"name": "supervised", "value": "True"}],
                },
            )
            job_id = resp.json()["id"]

            # Cancel it immediately (should be queued, not running yet)
            cancel_resp = client.post(f"/api/jobs/{job_id}/cancel")
            assert cancel_resp.status_code == 200

            # Check it's failed
            status_resp = client.get(f"/api/jobs/{job_id}")
            assert status_resp.json()["state"] in ("failed", "cancelling")


class TestCrashRecovery:
    def test_recover_stale_jobs(self, tmp_data_dir: object) -> None:
        """Test that stale running jobs are marked as failed on recovery."""
        from classify_api import repositories as repo
        from classify_api.db import get_session_factory

        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            # Manually create a job in "running" state (simulating a crash)
            db_factory = get_session_factory()
            db = db_factory()
            try:
                job = repo.create_job(db, report_uuid=report_id, args={"supervised": True})
                repo.update_job_state(db, job.id, "running")
                db.commit()
                job_id = job.id
            finally:
                db.close()

            # Call recovery
            resp = client.post("/api/jobs/recover")
            assert resp.status_code == 200
            assert resp.json()["recovered"] >= 1

            # Verify the job is now failed
            status_resp = client.get(f"/api/jobs/{job_id}")
            assert status_resp.json()["state"] == "failed"
            assert "Interrupted" in status_resp.json()["error"]
