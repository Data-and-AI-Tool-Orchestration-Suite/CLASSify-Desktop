"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from classify_api.settings import reset_settings

if TYPE_CHECKING:
    from classify_api.settings import Settings


@pytest.fixture()
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the app's data directory to a temporary path."""
    data_dir = tmp_path / "appdata"
    monkeypatch.setenv("CLASSIFY_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CLASSIFY_DEV_MODE", "false")
    reset_settings()
    return data_dir


@pytest.fixture()
def settings(tmp_data_dir: Path) -> Settings:
    """Yield settings pointed at the temp data dir, with dirs created."""
    from classify_api.settings import get_settings

    s = get_settings()
    s.ensure_dirs()
    return s
