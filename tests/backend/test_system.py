"""Tests for the system router (health, info, usage, cpu-count)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from classify_api.main import create_app


def test_health(tmp_data_dir: None) -> None:  # noqa: ARG001
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/system/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_info(tmp_data_dir: None) -> None:  # noqa: ARG001
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/system/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["app"] == "CLASSify Desktop"
    assert "version" in body
    assert "os" in body
    assert "data_dir" in body


def test_usage(tmp_data_dir: None) -> None:  # noqa: ARG001
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/system/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] > 0
    assert body["free"] > 0
    assert body["used"] >= 0


def test_cpu_count(tmp_data_dir: None) -> None:  # noqa: ARG001
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/api/system/cpu-count")
    assert resp.status_code == 200
    assert resp.json()["count"] >= 1
