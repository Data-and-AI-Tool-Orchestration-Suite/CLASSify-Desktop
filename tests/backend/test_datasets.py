"""Integration tests for the datasets router."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from classify_api.db import reset_engine, run_migrations
from classify_api.main import create_app
from classify_api.settings import reset_settings
from storage.factory import reset_storage


def _make_csv(content: str) -> bytes:
    return content.encode("utf-8")


SMALL_CSV = b"""feature_1,feature_2,class
3.5,10,yes
2.1,20,no
4.8,15,yes
1.9,25,no
3.2,12,yes
2.8,18,no
4.1,14,yes
1.5,22,no
3.9,11,yes
2.3,19,no
"""


def _setup_app(tmp_data_dir: object) -> TestClient:
    """Create app with migrations applied and return a TestClient."""
    reset_settings()
    reset_engine()
    reset_storage()
    run_migrations()
    app = create_app()
    return TestClient(app)


class TestUploadDataset:
    def test_upload_csv(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["report_id"] is not None
        assert body["filename"] == "test"
        assert "feature_1" in body["data_types"]
        assert body["data_types"]["feature_1"] == "float"
        assert body["data_types"]["feature_2"] == "integer"

    def test_upload_non_csv_rejected(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            )
        assert resp.status_code == 400

    def test_upload_yes_no_detected_as_bool(self, tmp_data_dir: object) -> None:
        csv = b"a,b,class\n1,10,yes\n2,20,no\n3,15,yes\n4,25,no\n"
        client = _setup_app(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/datasets/upload",
                files={"file": ("bool_test.csv", io.BytesIO(csv), "text/csv")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_types"]["class"] == "bool"

    def test_upload_duplicate_filename_gets_suffix(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            resp1 = client.post(
                "/api/datasets/upload",
                files={"file": ("dup.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            resp2 = client.post(
                "/api/datasets/upload",
                files={"file": ("dup.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
        assert resp1.json()["filename"] == "dup"
        assert resp2.json()["filename"] == "dup_1"


class TestColumnChanges:
    def test_apply_column_changes(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]

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
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True

    def test_drop_column(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]

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
                        "checked": False,
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
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_class_column_must_have_multiple_classes(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        csv = b"a,class\n1,1\n2,1\n3,1\n4,1\n"
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("single.csv", io.BytesIO(csv), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            changes = {
                "data_types": [
                    {
                        "column": "a",
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
            resp = client.post(f"/api/datasets/{report_id}/column-changes", json=changes)
        assert resp.status_code == 200
        assert resp.json()["success"] is False
        assert "more than one class" in resp.json()["message"]

    def test_404_for_nonexistent_report(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            resp = client.post(
                "/api/datasets/nonexistent-id/column-changes",
                json={"data_types": []},
            )
        assert resp.status_code == 404


class TestClassValues:
    def test_get_class_values(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            resp = client.get(f"/api/datasets/{report_id}/class-values?class_column=class")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert set(body["class_values"]) == {"yes", "no"}

    def test_class_values_nonexistent_column(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            resp = client.get(f"/api/datasets/{report_id}/class-values?class_column=nonexistent")
        assert resp.status_code == 400


class TestClassMapping:
    def test_set_class_mapping(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            resp = client.post(
                f"/api/datasets/{report_id}/class-mapping",
                json={"class_column": "class", "mapping": {"yes": 1, "no": 0}},
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


class TestTestsetUpload:
    def test_upload_testset(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        testset_csv = b"feature_1,feature_2,class\n5.0,30,yes\n1.0,5,no\n"
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            resp = client.post(
                f"/api/datasets/{report_id}/testset",
                files={"file": ("testset.csv", io.BytesIO(testset_csv), "text/csv")},
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


class TestDuplicate:
    def test_duplicate_dataset(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            resp = client.post(f"/api/datasets/{report_id}/duplicate")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert "copy" in resp.json()["message"]


class TestDelete:
    def test_delete_dataset(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            resp = client.delete(f"/api/datasets/{report_id}")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_delete_nonexistent_returns_404(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            resp = client.delete("/api/datasets/nonexistent-id")
        assert resp.status_code == 404


class TestListDatasets:
    def test_list_empty(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            resp = client.get("/api/datasets")
        assert resp.status_code == 200
        body = resp.json()
        assert body["recordsTotal"] == 0
        assert body["data"] == []

    def test_list_with_data(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            for i in range(3):
                client.post(
                    "/api/datasets/upload",
                    files={"file": (f"file_{i}.csv", io.BytesIO(SMALL_CSV), "text/csv")},
                )
            resp = client.get("/api/datasets")
        assert resp.status_code == 200
        body = resp.json()
        assert body["recordsTotal"] == 3
        assert len(body["data"]) == 3

    def test_list_with_search(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            client.post(
                "/api/datasets/upload",
                files={"file": ("alpha.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            client.post(
                "/api/datasets/upload",
                files={"file": ("beta.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            resp = client.get("/api/datasets?search=alpha")
        assert resp.status_code == 200
        body = resp.json()
        assert body["recordsFiltered"] == 1
        assert body["data"][0]["filename"] == "alpha"

    def test_list_pagination(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            for i in range(5):
                client.post(
                    "/api/datasets/upload",
                    files={"file": (f"file_{i:02d}.csv", io.BytesIO(SMALL_CSV), "text/csv")},
                )
            resp = client.get("/api/datasets?start=2&length=2")
        assert resp.status_code == 200
        body = resp.json()
        assert body["recordsTotal"] == 5
        assert len(body["data"]) == 2


class TestComment:
    def test_update_comment(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            resp = client.patch(
                f"/api/datasets/{report_id}/comment",
                json={"comments": "My test dataset"},
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_get_dataset_after_comment(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            client.patch(
                f"/api/datasets/{report_id}/comment",
                json={"comments": "Updated notes"},
            )
            resp = client.get(f"/api/datasets/{report_id}")
        assert resp.status_code == 200
        assert resp.json()["comments"] == "Updated notes"


class TestGetDataset:
    def test_get_single_dataset(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            upload_resp = client.post(
                "/api/datasets/upload",
                files={"file": ("test.csv", io.BytesIO(SMALL_CSV), "text/csv")},
            )
            report_id = upload_resp.json()["report_id"]
            resp = client.get(f"/api/datasets/{report_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "test"
        assert body["status"] == "Preview"

    def test_get_nonexistent_returns_404(self, tmp_data_dir: object) -> None:
        client = _setup_app(tmp_data_dir)
        with client:
            resp = client.get("/api/datasets/nonexistent-id")
        assert resp.status_code == 404
