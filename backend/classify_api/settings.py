"""Application configuration loaded from settings.json + environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_data_dir() -> Path:
    """Platform-appropriate default application data directory."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(base) / "CLASSify"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "CLASSify"
    base = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    return Path(base) / "CLASSify"


class Settings(BaseSettings):
    """Runtime settings.

    On first launch the app writes a ``settings.json`` into the data
    directory.  Subsequent launches read it back.  Environment variables
    (prefix ``CLASSIFY_``) override file values for testing / headless use.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLASSIFY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ──
    data_dir: Path = Field(default_factory=_default_data_dir)

    # ── Server ──
    host: str = "127.0.0.1"
    port: int = 0  # 0 = pick a free port at runtime
    dev_mode: bool = False

    # ── Compute ──
    n_jobs: int = Field(default_factory=lambda: os.cpu_count() or 1)
    max_upload_mb: int = 500

    # ── Security ──
    encryption_enabled: bool = False
    db_passphrase: str | None = None  # set via keyring at runtime, not persisted

    # ── UI ──
    theme: str = "default"

    # ── Add-ons ──
    addon_python_libs: Path | None = None  # resolved in property below

    @computed_field  # type: ignore[prop-decorator]
    @property
    def db_path(self) -> Path:
        return self.data_dir / "classify.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def addon_dir(self) -> Path:
        return self.data_dir / "addons" / "pythonlibs"

    def ensure_dirs(self) -> None:
        """Create all required directories on disk."""
        for d in (
            self.data_dir,
            self.datasets_dir,
            self.cache_dir,
            self.logs_dir,
            self.addon_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Clear the cached settings (used by tests)."""
    global _settings
    _settings = None
