"""Tests for shared logging setup (windowed/frozen stream handling)."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog

from classify_api.logging_setup import configure_logging, reset_logging_state


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    yield
    structlog.reset_defaults()
    reset_logging_state()


def test_windowed_logging_writes_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    log_file = tmp_path / "logs" / "classify.log"

    configure_logging(dev=False, log_file=log_file)
    structlog.get_logger().info("shell.starting", port=1234)

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    event = json.loads(lines[-1])
    assert event["event"] == "shell.starting"
    assert event["port"] == 1234


def test_windowed_logging_falls_back_to_devnull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    blocked = tmp_path / "not-a-dir.txt"
    blocked.write_text("", encoding="utf-8")
    log_file = blocked / "sub" / "classify.log"

    configure_logging(dev=False, log_file=log_file)
    structlog.get_logger().info("still.works")


def test_console_logging_does_not_create_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdout", sys.__stdout__)
    monkeypatch.setattr(sys, "stderr", sys.__stderr__)
    log_file = tmp_path / "logs" / "classify.log"

    configure_logging(dev=True, log_file=log_file)
    structlog.get_logger().info("boot")

    assert not log_file.exists()
