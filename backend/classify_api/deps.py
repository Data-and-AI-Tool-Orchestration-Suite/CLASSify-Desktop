"""Shared FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator

from classify_api.settings import Settings, get_settings


def get_app_settings() -> Iterator[Settings]:
    """Yield the application settings singleton."""
    yield get_settings()
