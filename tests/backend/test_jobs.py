"""Integration tests for the job runner: enqueue, run, cancel, crash recovery."""

from __future__ import annotations

import io
import json
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

    def test_start_training_supervised_requires_class_column(self, tmp_data_dir: object) -> None:
        """Supervised training without a class column is rejected with 400."""
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
                        {"name": "visualize", "value": "False"},
                    ],
                },
            )
        assert resp.status_code == 400
        assert "class" in resp.json()["detail"].lower()

    def test_start_training_unsupervised_without_class_column(self, tmp_data_dir: object) -> None:
        """Unsupervised training is accepted without any class column."""
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [
                        {"name": "supervised", "value": "False"},
                        {"name": "train_group", "value": "kmeans"},
                        {"name": "parameter_tune", "value": "False"},
                        {"name": "visualize", "value": "False"},
                    ],
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "queued"
        stored_args = body["args"] or {}
        assert "class_column" not in stored_args

    def test_unsupervised_after_class_mapping_flow(self, tmp_data_dir: object) -> None:
        """Regression: unsupervised training on a dataset that went through
        the categorical class-mapping flow must succeed.

        The mapping step stores a string '{class}_mapping' column; the
        unsupervised trainer used to leak it into the clustering features,
        crashing every model with 'could not convert string to float'.
        """
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            # Reconfigure with a categorical class column + mapping (the
            # standard supervised configuration flow)
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
                        "data_type": "categorical",
                        "checked": True,
                        "missing": "",
                        "fill_value": "",
                        "is_class": True,
                    },
                ]
            }
            resp = client.post(f"/api/datasets/{report_id}/column-changes", json=changes)
            assert resp.json()["success"] is True

            values = client.get(
                f"/api/datasets/{report_id}/class-values", params={"class_column": "class"}
            ).json()["class_values"]
            mapping = {v: str(i) for i, v in enumerate(sorted(values))}
            resp = client.post(
                f"/api/datasets/{report_id}/class-mapping",
                json={"class_column": "class", "mapping": mapping},
            )
            assert resp.json()["success"] is True

            # Now train unsupervised with the class column marked (to exclude
            # it from the clustering features) â€” exactly what the frontend sends
            resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [
                        {"name": "supervised", "value": "False"},
                        {"name": "train_group", "value": "kmeans"},
                        {"name": "parameter_tune", "value": "False"},
                        {"name": "visualize", "value": "False"},
                        {"name": "random_state", "value": "42"},
                        {"name": "num_clusters", "value": "2"},
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

            assert status["state"] == "succeeded", (
                f"Job ended in state: {status['state']}, error: {status.get('error')}"
            )
            results = client.get(f"/api/results/{report_id}").json()
            assert results["success"] is True
            assert results["report_csv"][0]["model"] == "kmeans"

    def test_start_training_lowercase_boolean_options(self, tmp_data_dir: object) -> None:
        """Regression: the frontend sends String(bool) â†’ lowercase 'false'/'true'.

        The parser only matched capitalized 'True'/'False', so 'false' fell
        through to the numeric-parse fallback and was stored as the STRING
        'false' â€” truthy! Unsupervised requests from the UI therefore ran the
        supervised trainer (or hit the class-column 400).
        """
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [
                        {"name": "supervised", "value": "false"},
                        {"name": "train_group", "value": "kmeans"},
                        {"name": "parameter_tune", "value": "false"},
                        {"name": "visualize", "value": "false"},
                    ],
                },
            )
            assert resp.status_code == 200, resp.json()
            body = resp.json()
            assert body["state"] == "queued"
            stored_args = body["args"] or {}
            assert stored_args["supervised"] is False
            assert stored_args["parameter_tune"] is False
            assert "class_column" not in stored_args

    def test_start_training_queued_behind_running_job(self, tmp_data_dir: object) -> None:
        """A second job for a DIFFERENT dataset queues instead of 409ing.

        Regression: any start while another job ran was rejected with 409,
        forcing the user to wait manually.  Jobs now queue FIFO and run
        sequentially.
        """
        client, report_id_a = _setup_and_upload(tmp_data_dir)
        # Second, independent dataset
        resp = client.post(
            "/api/datasets/upload",
            files={"file": ("second.csv", io.BytesIO(SMALL_CSV), "text/csv")},
        )
        assert resp.status_code == 200
        report_id_b = resp.json()["report_id"]

        with client:
            resp_a = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id_a,
                    "options": [
                        {"name": "supervised", "value": "false"},
                        {"name": "train_group", "value": "kmeans"},
                        {"name": "parameter_tune", "value": "false"},
                        {"name": "visualize", "value": "false"},
                    ],
                },
            )
            assert resp_a.status_code == 200

            resp_b = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id_b,
                    "options": [
                        {"name": "supervised", "value": "false"},
                        {"name": "train_group", "value": "kmeans"},
                        {"name": "parameter_tune", "value": "false"},
                        {"name": "visualize", "value": "false"},
                    ],
                },
            )
            assert resp_b.status_code == 200, resp_b.json()
            # Job B may be queued (A running) or already running â€” never 409
            assert resp_b.json()["state"] in ("queued", "running")

    def test_start_training_same_dataset_twice_rejected(self, tmp_data_dir: object) -> None:
        """A second job for the SAME dataset is still rejected with 409."""
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [
                        {"name": "supervised", "value": "false"},
                        {"name": "train_group", "value": "kmeans"},
                        {"name": "parameter_tune", "value": "false"},
                        {"name": "visualize", "value": "false"},
                    ],
                },
            )
            assert resp.status_code == 200

            resp2 = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [
                        {"name": "supervised", "value": "false"},
                        {"name": "train_group", "value": "kmeans"},
                        {"name": "parameter_tune", "value": "false"},
                        {"name": "visualize", "value": "false"},
                    ],
                },
            )
            assert resp2.status_code == 409
            detail = resp2.json()["detail"]
            assert "queued or running for this dataset" in detail

    def test_queued_job_sets_report_status_and_cancel_restores_it(
        self, tmp_data_dir: object
    ) -> None:
        """Enqueueing marks the dataset 'Queued'; cancelling restores the
        pre-queue status so the dataset list reflects the queue."""
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [
                        {"name": "supervised", "value": "false"},
                        {"name": "train_group", "value": "kmeans"},
                        {"name": "parameter_tune", "value": "false"},
                        {"name": "visualize", "value": "false"},
                    ],
                },
            )
            assert resp.status_code == 200
            job_id = resp.json()["id"]

            report = client.get(f"/api/datasets/{report_id}").json()
            assert report["status"] == "Queued"

            resp = client.post(f"/api/jobs/{job_id}/cancel")
            assert resp.status_code == 200

            report = client.get(f"/api/datasets/{report_id}").json()
            assert report["status"] == "Uploaded"

    def test_sse_events_include_args(self, tmp_data_dir: object) -> None:
        """The SSE progress stream must include job args so the frontend can
        show the full requested model list (regression: the list only showed
        models seen in the log so far, with totals like '3 of 6' for 10)."""
        client, report_id = _setup_and_upload(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/jobs",
                json={
                    "report_id": report_id,
                    "options": [
                        {"name": "supervised", "value": "false"},
                        {"name": "train_group", "value": "kmeans"},
                        {"name": "train_group", "value": "spectralclustering"},
                        {"name": "parameter_tune", "value": "false"},
                        {"name": "visualize", "value": "false"},
                    ],
                },
            )
            job_id = resp.json()["id"]

            with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
                event_payload = None
                for line in stream.iter_lines():
                    if line.startswith("data: "):
                        event_payload = json.loads(line[len("data: ") :])
                        break
            assert event_payload is not None
            args = event_payload.get("args") or {}
            assert args.get("train_group") == ["kmeans", "spectralclustering"]

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
                    "options": [
                        {"name": "supervised", "value": "True"},
                        {"name": "class_column", "value": "class"},
                    ],
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
                json={
                    "report_id": report_id,
                    "options": [{"name": "class_column", "value": "class"}],
                },
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
        """Full end-to-end: upload â†’ configure â†’ submit â†’ wait â†’ verify results."""
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
                    "options": [
                        {"name": "supervised", "value": "True"},
                        {"name": "class_column", "value": "class"},
                    ],
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
