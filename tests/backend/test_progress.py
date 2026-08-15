"""Tests for the progress tracking module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runner.progress import ProgressUpdate, append_log, read_progress, write_progress
from storage.local import LocalStorage


@pytest.fixture()
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "datasets")


class TestProgressWriteRead:
    def test_write_and_read_progress(self, storage: LocalStorage) -> None:
        storage.put_text("report1/placeholder", "")
        write_progress(storage, "report1", 5, 10, "5/10 Processed")
        result = read_progress(storage, "report1")
        assert result is not None
        assert result.completed == 5
        assert result.total == 10
        assert result.message == "5/10 Processed"
        assert result.timestamp > 0

    def test_read_progress_nonexistent(self, storage: LocalStorage) -> None:
        result = read_progress(storage, "nonexistent")
        assert result is None

    def test_write_progress_overwrites(self, storage: LocalStorage) -> None:
        write_progress(storage, "r", 1, 10, "first")
        write_progress(storage, "r", 5, 10, "second")
        result = read_progress(storage, "r")
        assert result is not None
        assert result.completed == 5
        assert result.message == "second"

    def test_progress_json_format(self, storage: LocalStorage) -> None:
        write_progress(storage, "r", 3, 7, "3/7")
        raw = storage.get_text("r/progress.json")
        data = json.loads(raw)
        assert data["completed"] == 3
        assert data["total"] == 7
        assert data["message"] == "3/7"
        assert "timestamp" in data


class TestAppendLog:
    def test_append_to_empty_log(self, storage: LocalStorage) -> None:
        append_log(storage, "r", "First line")
        assert storage.get_text("r/output_log") == "First line\n"

    def test_append_multiple_lines(self, storage: LocalStorage) -> None:
        append_log(storage, "r", "Line 1")
        append_log(storage, "r", "Line 2")
        append_log(storage, "r", "Line 3")
        log = storage.get_text("r/output_log")
        assert "Line 1" in log
        assert "Line 2" in log
        assert "Line 3" in log
        assert log.count("\n") == 3

    def test_append_log_nonexistent_storage(self, storage: LocalStorage) -> None:
        append_log(storage, "r", "Creates the log")
        assert storage.exists("r/output_log")


class TestProgressUpdate:
    def test_progress_update_dataclass(self) -> None:
        update = ProgressUpdate(completed=10, total=20, message="Halfway", timestamp=12345.0)
        assert update.completed == 10
        assert update.total == 20
        assert update.message == "Halfway"
        assert update.timestamp == 12345.0
