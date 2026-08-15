"""Local filesystem storage — the default Storage implementation.

Maps S3-style keys (``report_id/viz/ROC_model``) to filesystem paths
(``<appdata>/datasets/report_id/viz/ROC_model``).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import IO, Any

from storage.base import InsufficientSpace, KeyNotFound, StorageError

# Minimum free disk space required for writes (100 MB)
_MIN_FREE_BYTES = 100 * 1024 * 1024


class LocalStorage:
    """Filesystem-backed storage under a root directory.

    Each key component after the first ``/`` becomes a subdirectory,
    matching the web app's S3 key structure exactly.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _key_to_path(self, key: str) -> Path:
        """Convert a storage key to a filesystem path."""
        if not key:
            raise StorageError("Empty key")
        # Prevent path traversal: reject keys with .. or absolute paths
        if ".." in key.split("/"):
            raise StorageError(f"Invalid key (path traversal): {key}")
        return self._root / key

    def _check_disk_space(self, estimated_bytes: int = 0) -> None:
        """Raise if there isn't enough free disk space."""
        target = self._root
        while not target.exists():
            target = target.parent
        usage = shutil.disk_usage(str(target))
        if usage.free - estimated_bytes < _MIN_FREE_BYTES:
            raise InsufficientSpace(
                f"Only {usage.free / 1e6:.0f} MB free; need at least {_MIN_FREE_BYTES / 1e6:.0f} MB"
            )

    def get_bytes(self, key: str) -> bytes:
        path = self._key_to_path(key)
        if not path.is_file():
            raise KeyNotFound(f"Key not found: {key}")
        return path.read_bytes()

    def put_bytes(self, key: str, data: bytes) -> None:
        self._check_disk_space(len(data))
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file in same dir, then rename
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def get_text(self, key: str, encoding: str = "utf-8") -> str:
        return self.get_bytes(key).decode(encoding)

    def put_text(self, key: str, text: str, encoding: str = "utf-8") -> None:
        self.put_bytes(key, text.encode(encoding))

    def read_csv(self, key: str, **kwargs: Any) -> Any:
        import pandas as pd

        path = self._key_to_path(key)
        if not path.is_file():
            raise KeyNotFound(f"Key not found: {key}")
        return pd.read_csv(path, **kwargs)

    def write_csv(self, key: str, df: Any, **kwargs: Any) -> None:
        import io

        buf = io.StringIO()
        df.to_csv(buf, **kwargs)
        self.put_text(key, buf.getvalue())

    def list(self, prefix: str) -> list[str]:
        prefix_path = self._key_to_path(prefix)
        if not prefix_path.exists():
            return []
        if prefix_path.is_file():
            return [prefix]
        keys: list[str] = []
        root_str = str(self._root) + os.sep
        for path in sorted(prefix_path.rglob("*")):
            if path.is_file():
                rel = str(path)
                if rel.startswith(root_str):
                    rel = rel[len(root_str) :]
                keys.append(rel.replace(os.sep, "/"))
        return keys

    def exists(self, key: str) -> bool:
        return self._key_to_path(key).is_file()

    def delete(self, key: str) -> None:
        path = self._key_to_path(key)
        path.unlink(missing_ok=True)
        # Clean up empty parent dirs (but not the root)
        self._cleanup_empty_dirs(path.parent)

    def delete_prefix(self, prefix: str) -> int:
        prefix_path = self._key_to_path(prefix)
        if not prefix_path.exists():
            return 0
        count = 0
        if prefix_path.is_file():
            prefix_path.unlink()
            return 1
        for path in sorted(prefix_path.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
                count += 1
            elif path.is_dir() and path != prefix_path:
                with contextlib.suppress(OSError):
                    path.rmdir()
        # Remove the prefix dir itself
        if prefix_path.is_dir():
            with contextlib.suppress(OSError):
                prefix_path.rmdir()
        return count

    def copy(self, src: str, dst: str) -> None:
        src_path = self._key_to_path(src)
        if not src_path.is_file():
            raise KeyNotFound(f"Source key not found: {src}")
        dst_path = self._key_to_path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        self._check_disk_space(src_path.stat().st_size)
        shutil.copy2(src_path, dst_path)

    def open_read(self, key: str) -> IO[bytes]:
        path = self._key_to_path(key)
        if not path.is_file():
            raise KeyNotFound(f"Key not found: {key}")
        return open(path, "rb")

    def open_write(self, key: str) -> IO[bytes]:
        self._check_disk_space()
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return _AtomicWriter(path)  # type: ignore[return-value]

    def _cleanup_empty_dirs(self, dir_path: Path) -> None:
        """Remove empty parent directories up to (but not including) root."""
        try:
            current = dir_path.resolve()
            root = self._root.resolve()
            while current != root and current.parent != current:
                current.rmdir()
                current = current.parent
        except OSError:
            pass


class _AtomicWriter:
    """File-like wrapper that writes to a temp file and atomically renames on close."""

    def __init__(self, target: Path) -> None:
        self._target = target
        self._tmp: Path | None = None
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        self._fd = fd
        self._tmp = Path(tmp)
        self._closed = False

    def write(self, data: bytes) -> int:
        return os.write(self._fd, data)

    def flush(self) -> None:
        os.fsync(self._fd)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._fd)
        if self._tmp is not None:
            os.replace(self._tmp, self._target)
            self._tmp = None

    def __enter__(self) -> _AtomicWriter:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if exc_type is not None:
            # On error, clean up temp file (close fd first — required on Windows)
            if not self._closed:
                os.close(self._fd)
                self._closed = True
            if self._tmp and self._tmp.exists():
                self._tmp.unlink(missing_ok=True)
        else:
            self.close()

    def __del__(self) -> None:
        if not self._closed:
            self.close()
