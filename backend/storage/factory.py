"""Storage factory — resolves the right Storage implementation from settings."""

from __future__ import annotations

from classify_api.settings import get_settings
from storage.base import Storage
from storage.local import LocalStorage

_storage: Storage | None = None


def get_storage() -> Storage:
    """Return the cached Storage singleton.

    When encryption is enabled in settings, wraps LocalStorage with
    EncryptedStorage.  Otherwise returns a plain LocalStorage.
    """
    global _storage
    if _storage is None:
        settings = get_settings()
        base = LocalStorage(settings.datasets_dir)
        if settings.encryption_enabled and settings.db_passphrase:
            from storage.encrypted import EncryptedStorage

            _storage = EncryptedStorage(base, settings.db_passphrase)
        else:
            _storage = base
    return _storage


def reset_storage() -> None:
    """Clear the cached storage instance (used by tests)."""
    global _storage
    _storage = None
