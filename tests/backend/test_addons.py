"""Tests for the add-on system."""

from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from classify_api.db import reset_engine, run_migrations
from classify_api.main import create_app
from classify_api.services.addon_service import (
    BUILTIN_ADDONS,
    get_addon_dir,
    get_installed_addons,
    is_addon_installed,
    list_available_addons,
)
from classify_api.settings import reset_settings
from storage.factory import reset_storage


def _setup(tmp_data_dir: object) -> TestClient:
    reset_settings()
    reset_engine()
    reset_storage()
    run_migrations()
    return TestClient(create_app())


class TestAddonManifest:
    def test_builtin_addons_exist(self) -> None:
        assert "tabpfn" in BUILTIN_ADDONS
        assert "sdv" in BUILTIN_ADDONS

    def test_tabpfn_manifest(self) -> None:
        m = BUILTIN_ADDONS["tabpfn"]
        assert m.name == "tabpfn"
        assert "tabpfn" in m.provides
        assert "torch" in m.provides
        assert m.size_estimate_mb > 0

    def test_sdv_manifest(self) -> None:
        m = BUILTIN_ADDONS["sdv"]
        assert m.name == "sdv"
        assert "sdv" in m.provides
        assert "torch" in m.provides


class TestAddonRegistry:
    def test_list_available_addons(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            addons = list_available_addons()
        assert len(addons) == 2
        names = [a["name"] for a in addons]
        assert "tabpfn" in names
        assert "sdv" in names
        for a in addons:
            assert "installed" in a
            assert a["installed"] is False  # nothing installed in fresh env

    def test_is_addon_installed_false_initially(self, tmp_data_dir: object) -> None:
        _setup(tmp_data_dir)
        assert not is_addon_installed("tabpfn")
        assert not is_addon_installed("sdv")

    def test_get_installed_addons_empty_initially(self, tmp_data_dir: object) -> None:
        _setup(tmp_data_dir)
        assert get_installed_addons() == {}

    def test_addon_dir_created(self, tmp_data_dir: object) -> None:
        _setup(tmp_data_dir)
        addon_dir = get_addon_dir()
        assert addon_dir.exists()
        assert addon_dir.is_dir()


class TestAddonEndpoints:
    def test_list_addons_endpoint(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            resp = client.get("/api/addons")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["addons"]) == 2
        for a in body["addons"]:
            assert a["installed"] is False
            assert a["size_estimate_mb"] > 0

    def test_addon_status_endpoint(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            resp = client.get("/api/addons/tabpfn/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "tabpfn"
        assert body["installed"] is False
        assert "modules_available" in body

    def test_addon_status_unknown(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            resp = client.get("/api/addons/nonexistent/status")
        assert resp.status_code == 404

    def test_check_modules_endpoint(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            resp = client.get("/api/addons/modules/check")
        assert resp.status_code == 200
        body = resp.json()
        assert "modules" in body
        assert len(body["modules"]) >= 3  # tabpfn, sdv, torch

    def test_uninstall_not_installed(self, tmp_data_dir: object) -> None:
        client = _setup(tmp_data_dir)
        with client:
            resp = client.post("/api/addons/tabpfn/uninstall")
        assert resp.status_code == 200
        assert resp.json()["success"] is False


class TestAddonInstallMocked:
    """Test install/uninstall with mocked pip (no actual downloads)."""

    def test_install_unknown_addon(self, tmp_data_dir: object) -> None:
        from classify_api.services.addon_service import install_addon

        _setup(tmp_data_dir)
        result = install_addon("nonexistent")
        assert result["success"] is False
        assert "Unknown" in result["message"]

    def test_uninstall_unknown_addon(self, tmp_data_dir: object) -> None:
        from classify_api.services.addon_service import uninstall_addon

        _setup(tmp_data_dir)
        result = uninstall_addon("nonexistent")
        assert result["success"] is False

    def test_install_addon_mocked_pip(self, tmp_data_dir: object) -> None:
        """Mock pip to test the install flow without downloading torch."""
        from classify_api.services import addon_service
        from classify_api.services.addon_service import _run_install, get_install_status

        _setup(tmp_data_dir)

        # Set up initial status (normally done by install_addon)
        addon_service._install_status["tabpfn"] = addon_service.InstallStatus(
            addon="tabpfn", state="installing", progress=[]
        )

        mock_result = type("MockResult", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with (
            patch("subprocess.run", return_value=mock_result),
            patch.object(addon_service, "_verify_in_subprocess", return_value=(True, None)),
            patch.object(addon_service, "_clear_addon_dir", return_value=True),
        ):
            _run_install("tabpfn")

        status = get_install_status("tabpfn")
        assert status["state"] == "succeeded"
        assert len(status["progress"]) > 0

    def test_manual_registry_write_read(self, tmp_data_dir: object) -> None:
        """Test the installed.json registry by writing and reading it directly."""
        from classify_api.services.addon_service import get_installed_addons_file

        _setup(tmp_data_dir)
        registry = get_installed_addons_file()
        registry.write_text(json.dumps({"tabpfn": "2.0.0"}))

        installed = get_installed_addons()
        assert "tabpfn" in installed
        assert installed["tabpfn"] == "2.0.0"
        assert is_addon_installed("tabpfn")
