"""Storage protocol — the abstraction layer that replaces S3/boto3.

Every ML engine and API service talks to ``Storage`` instead of
``S3_CONNECTION['client']``.  Keys mirror the web app's S3 layout:

    <report_id>/file                  — processed dataset CSV
    <report_id>/original_file         — untouched upload
    <report_id>/testset               — separate test set
    <report_id>/retest                — re-test dataset
    <report_id>/report.csv            — metrics report
    <report_id>/results.json          — full results JSON
    <report_id>/output_log            — training stdout/stderr
    <report_id>/<model>_model.joblib  — saved model
    <report_id>/scaler.joblib         — fitted scaler
    <report_id>/shap_rows_<model>     — per-row SHAP CSV
    <report_id>/viz/<CHART>_<model>   — visualization PNGs
    <report_id>/progress.json         — live job progress
"""

from __future__ import annotations

from typing import IO, Protocol, runtime_checkable


class StorageError(Exception):
    """Base error for storage operations."""


class KeyNotFound(StorageError):
    """Raised when a requested key does not exist."""


class InsufficientSpace(StorageError):
    """Raised when there is not enough disk space for a write."""


@runtime_checkable
class Storage(Protocol):
    """File/blob storage interface used throughout the app.

    Keys use ``/`` as a separator (same as S3 keys).  Implementations
    map keys to filesystem paths internally.
    """

    def get_bytes(self, key: str) -> bytes:
        """Read a key and return its raw bytes.  Raise ``KeyNotFound`` if missing."""
        ...

    def put_bytes(self, key: str, data: bytes) -> None:
        """Write raw bytes to a key.  Atomic write (temp + rename)."""
        ...

    def get_text(self, key: str, encoding: str = "utf-8") -> str:
        """Read a key as a decoded string."""
        ...

    def put_text(self, key: str, text: str, encoding: str = "utf-8") -> None:
        """Write a string to a key."""
        ...

    def read_csv(self, key: str, **kwargs: object) -> object:
        """Read a key as a pandas DataFrame (lazy import of pandas)."""
        ...

    def write_csv(self, key: str, df: object, **kwargs: object) -> None:
        """Write a DataFrame to a key as CSV (lazy import of pandas)."""
        ...

    def list(self, prefix: str) -> list[str]:
        """List all keys that start with ``prefix``.  Returns sorted keys."""
        ...

    def exists(self, key: str) -> bool:
        """Return ``True`` if the key exists."""
        ...

    def delete(self, key: str) -> None:
        """Delete a single key.  No error if missing."""
        ...

    def delete_prefix(self, prefix: str) -> int:
        """Delete all keys under ``prefix``.  Returns count deleted."""
        ...

    def copy(self, src: str, dst: str) -> None:
        """Copy ``src`` key to ``dst`` key."""
        ...

    def open_read(self, key: str) -> IO[bytes]:
        """Open a key for binary reading.  Caller must close."""
        ...

    def open_write(self, key: str) -> IO[bytes]:
        """Open a key for binary writing.  Caller must close.  Atomic on close."""
        ...
