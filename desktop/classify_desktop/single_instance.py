"""Single-instance enforcement — prevents multiple app instances.

Uses a lock file on Linux/macOS and a named mutex on Windows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from classify_api.settings import get_settings

_lock_fd: int | None = None
_lock_path: Path | None = None


def acquire_lock() -> bool:
    """Try to acquire the single-instance lock.

    Returns True if this is the first instance, False if another is running.
    """
    global _lock_fd, _lock_path

    settings = get_settings()
    _lock_path = settings.data_dir / ".classify.lock"

    if sys.platform == "win32":
        return _acquire_windows_lock()
    return _acquire_unix_lock()


def _acquire_windows_lock() -> bool:
    """Windows: use a named mutex via ctypes."""
    import ctypes
    import ctypes.wintypes

    mutex_name = "CLASSifyDesktopSingleInstance"

    kernel32 = ctypes.windll.kernel32

    CREATE_MUTEX = kernel32.CreateMutexW
    CREATE_MUTEX.restype = ctypes.wintypes.HANDLE
    CREATE_MUTEX.argtypes = [ctypes.wintypes.LPCVOID, ctypes.wintypes.BOOL, ctypes.wintypes.LPCWSTR]

    mutex = CREATE_MUTEX(None, False, mutex_name)

    last_error = kernel32.GetLastError()
    if last_error == 183:
        kernel32.CloseHandle(mutex)
        return False

    acquire_lock._win_mutex = mutex  # noqa: SLF001
    return True


def _acquire_unix_lock() -> bool:
    """Unix: use a file lock (flock)."""
    global _lock_fd, _lock_path

    assert _lock_path is not None

    _lock_path.parent.mkdir(parents=True, exist_ok=True)
    _lock_fd = os.open(str(_lock_path), os.O_CREAT | os.O_RDWR, 0o644)

    try:
        import fcntl

        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(_lock_fd, f"{os.getpid()}\n".encode())
        return True
    except (OSError, ImportError):
        os.close(_lock_fd)
        _lock_fd = None
        return False


def release_lock() -> None:
    """Release the single-instance lock (on shutdown)."""
    global _lock_fd, _lock_path

    if sys.platform == "win32":
        mutex = getattr(acquire_lock, "_win_mutex", None)  # noqa: SLF001
        if mutex:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.CloseHandle(mutex)
        return

    if _lock_fd is not None:
        try:
            import fcntl

            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
        except (OSError, ImportError):
            pass
        os.close(_lock_fd)
        _lock_fd = None

    if _lock_path and _lock_path.exists():
        _lock_path.unlink(missing_ok=True)
