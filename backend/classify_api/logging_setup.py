"""Logging setup shared by the API server and the desktop shell.

Windowed (frozen) builds have ``sys.stdout`` and ``sys.stderr`` set to
``None``. structlog's default ``PrintLoggerFactory`` then fails with
``TypeError: cannot create weak reference to 'NoneType' object`` because
``PrintLogger`` falls back to the (``None``) captured ``stdout``. In that
case all output is redirected to a log file, or ``os.devnull`` as a last
resort.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TextIO

import structlog

_log_file_handle: TextIO | None = None


def _open_log_file(log_file: Path | None) -> TextIO:
    """Open *log_file* for appending, falling back to ``os.devnull``."""
    global _log_file_handle
    if _log_file_handle is not None and not _log_file_handle.closed:
        return _log_file_handle

    handle: TextIO | None = None
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handle = open(log_file, "a", encoding="utf-8")  # noqa: SIM115
        except OSError:
            handle = None
    if handle is None:
        handle = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115

    _log_file_handle = handle
    return handle


def reset_logging_state() -> None:
    """Close the redirect log file and clear cached state (used by tests)."""
    global _log_file_handle
    if _log_file_handle is not None and not _log_file_handle.closed:
        _log_file_handle.close()
    _log_file_handle = None


def configure_logging(dev: bool, log_file: Path | None = None) -> None:
    """Configure structlog + stdlib logging for every entry point.

    In console sessions output goes to stdout/stderr as usual. In windowed
    builds (no console streams) everything is written to *log_file*.
    """
    level = logging.DEBUG if dev else logging.INFO
    windowed = sys.stdout is None or sys.stderr is None

    root = logging.getLogger()
    if windowed:
        stream = _open_log_file(log_file)
        if not root.handlers:
            root.addHandler(logging.StreamHandler(stream))
            root.setLevel(level)
    else:
        logging.basicConfig(level=level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            (structlog.dev.ConsoleRenderer() if dev else structlog.processors.JSONRenderer()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(
            file=_open_log_file(log_file) if windowed else None
        ),
        cache_logger_on_first_use=True,
    )
