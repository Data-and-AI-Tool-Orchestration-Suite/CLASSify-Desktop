"""Tests for the first-run service."""

from __future__ import annotations

from classify_api.db import reset_engine, run_migrations
from classify_api.services.first_run import (
    get_wizard_state,
    is_first_run,
    mark_initialized,
)
from classify_api.settings import reset_settings
from storage.factory import reset_storage


def _setup(tmp_data_dir: object) -> None:
    reset_settings()
    reset_engine()
    reset_storage()
    run_migrations()


class TestFirstRun:
    def test_is_first_run_true_initially(self, tmp_data_dir: object) -> None:
        _setup(tmp_data_dir)
        assert is_first_run() is True

    def test_mark_initialized(self, tmp_data_dir: object) -> None:
        _setup(tmp_data_dir)
        assert is_first_run() is True
        mark_initialized()
        assert is_first_run() is False

    def test_mark_initialized_idempotent(self, tmp_data_dir: object) -> None:
        _setup(tmp_data_dir)
        mark_initialized()
        mark_initialized()
        assert is_first_run() is False

    def test_get_wizard_state_first_run(self, tmp_data_dir: object) -> None:
        _setup(tmp_data_dir)
        state = get_wizard_state()
        assert state["first_run"] is True
        assert "data_dir" in state
        assert state["disk_free_bytes"] > 0
        assert state["cpu_count"] >= 1
        assert "has_addons" in state

    def test_get_wizard_state_after_init(self, tmp_data_dir: object) -> None:
        _setup(tmp_data_dir)
        mark_initialized()
        state = get_wizard_state()
        assert state["first_run"] is False

    def test_wizard_state_has_addon_status(self, tmp_data_dir: object) -> None:
        _setup(tmp_data_dir)
        state = get_wizard_state()
        assert "tabpfn" in state["has_addons"]
        assert "sdv" in state["has_addons"]
        # In a base install, neither should be available
        assert state["has_addons"]["tabpfn"] is False
        assert state["has_addons"]["sdv"] is False
