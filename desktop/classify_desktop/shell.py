"""Desktop shell — pywebview window, system tray, and lifecycle management.

This is the core of the desktop app: it starts the API server, opens a
native window with the SPA, manages the system tray, and handles the
close-with-active-jobs prompt.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Any

import structlog

log = structlog.get_logger()


class DesktopShell:
    """Manages the pywebview window, tray icon, and app lifecycle."""

    def __init__(self, base_url: str, port: int, token: str) -> None:
        self._base_url = base_url
        self._port = port
        self._token = token
        self._window: Any = None
        self._tray: Any = None
        self._quitting = False

    def run(self) -> None:
        """Start the webview and block until the app quits."""
        import webview

        url = self._build_url()

        self._window = webview.create_window(
            title="CLASSify Desktop",
            url=url,
            width=1280,
            height=800,
            min_size=(1024, 600),
            text_select=False,
        )
        self._window.events.closing += self._on_window_closing

        webview.start(
            debug=self._is_dev(),
            http_server=False,
            func=self._on_started,
        )

    def _build_url(self) -> str:
        """Build the URL to load in the webview."""
        from classify_desktop.server import get_spa_url

        return get_spa_url(self._base_url, self._token)

    def _is_dev(self) -> bool:
        """Check if running in development mode."""
        return (
            os.environ.get("CLASSIFY_DEV_MODE", "").lower() in ("true", "1")
            and not self._is_frozen()
        )

    def _is_frozen(self) -> bool:
        return getattr(sys, "frozen", False)

    def _on_started(self) -> None:
        """Called after the webview window is created."""
        log.info("shell.window_ready")

        if self._is_dev():
            self._window.evaluate_js("document.title = 'CLASSify Desktop (Dev)'")

    def _on_window_closing(self) -> bool:
        """Handle window close — prompt if jobs are running.

        Returns False to prevent closing (minimize to tray instead),
        or True to allow the close.
        """
        if self._quitting:
            return True

        has_active = self._has_active_jobs()

        if has_active:
            choice = self._show_close_prompt()
            if choice == "tray":
                self._minimize_to_tray()
                return False
            elif choice == "quit":
                self._quit()
                return True
            else:
                return False
        else:
            # No active jobs — quit cleanly.
            self._quit()
            return True

    def _has_active_jobs(self) -> bool:
        """Check if any training jobs are running or queued."""
        try:
            from classify_api.db import get_session_factory
            from classify_api.repositories import get_next_queued_job, get_running_job

            db_factory = get_session_factory()
            db = db_factory()
            try:
                running = get_running_job(db)
                queued = get_next_queued_job(db)
                return running is not None or queued is not None
            finally:
                db.close()
        except Exception:
            return False

    def _show_close_prompt(self) -> str:
        """Show a dialog asking what to do with active jobs.

        Returns "tray", "quit", or "cancel".
        """
        try:
            result = self._window.create_confirmation_dialog(
                "Training in Progress",
                "A training job is still running.\n\n"
                "Keep running in the system tray, or stop jobs and quit?",
                ["Keep running (tray)", "Stop & quit", "Cancel"],
            )
            if result == "Keep running (tray)":
                return "tray"
            elif result == "Stop & quit":
                return "quit"
            return "cancel"
        except Exception:
            return "tray"

    def _minimize_to_tray(self) -> None:
        """Hide the window (minimize to tray)."""
        try:
            self._window.hide()
            self._show_tray_notification(
                "CLASSify Desktop", "Running in the background. Click the tray icon to show."
            )
        except Exception:
            pass

    def _show_tray_notification(self, title: str, message: str) -> None:
        """Show a tray notification (platform-specific)."""
        try:
            if sys.platform == "win32":
                import ctypes

                NIM_ADD = 0x00000000
                NIF_INFO = 0x00000010
                hwnd = self._window.gui.hwnd if hasattr(self._window, "gui") else 0
                if hwnd:

                    class NOTIFYICONDATA(ctypes.Structure):
                        _fields_ = [
                            ("cbSize", ctypes.c_uint32),
                            ("hWnd", ctypes.c_void_p),
                            ("uID", ctypes.c_uint32),
                            ("uFlags", ctypes.c_uint32),
                            ("uCallbackMessage", ctypes.c_uint32),
                            ("hIcon", ctypes.c_void_p),
                            ("szTip", ctypes.c_wchar * 128),
                            ("dwState", ctypes.c_uint32),
                            ("dwStateMask", ctypes.c_uint32),
                            ("szInfo", ctypes.c_wchar * 256),
                            ("uTimeout", ctypes.c_uint32),
                            ("szInfoTitle", ctypes.c_wchar * 64),
                            ("dwInfoFlags", ctypes.c_uint32),
                        ]

                    data = NOTIFYICONDATA()
                    data.cbSize = ctypes.sizeof(NOTIFYICONDATA)
                    data.hWnd = hwnd
                    data.uFlags = NIF_INFO
                    data.szInfo = message
                    data.szInfoTitle = title
                    ctypes.windll.shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(data))
        except Exception:
            pass

    def _quit(self) -> None:
        """Full quit — cancel jobs, stop server, exit."""
        self._quitting = True
        log.info("shell.quitting")

        self._cancel_all_jobs()

        from runner.manager import stop_manager

        stop_manager()

        from classify_desktop.single_instance import release_lock

        release_lock()

        try:
            if self._window:
                self._window.destroy()
        except Exception:
            pass

    def _cancel_all_jobs(self) -> None:
        """Cancel any running or queued jobs."""
        try:
            from classify_api.db import get_session_factory
            from classify_api.repositories import get_running_job, update_job_state
            from runner.cancellation import set_cancel_flag
            from storage.factory import get_storage

            db_factory = get_session_factory()
            db = db_factory()
            storage = get_storage()
            try:
                running = get_running_job(db)
                if running:
                    set_cancel_flag(storage, running.report_uuid)
                    update_job_state(db, running.id, "cancelling")
                    db.commit()
                    time.sleep(2)
            finally:
                db.close()
        except Exception:
            pass
        from runner.manager import get_manager

        with contextlib.suppress(Exception):
            get_manager().cancel_current_job()

    def show_window(self) -> None:
        """Show the window from tray."""
        with contextlib.suppress(Exception):
            self._window.show()


def run_shell() -> None:
    """Entry point — start server, acquire lock, run shell."""
    from classify_desktop.server import start_server
    from classify_desktop.single_instance import acquire_lock

    if not acquire_lock():
        print("CLASSify Desktop is already running.", file=sys.stderr)
        sys.exit(0)

    log.info("shell.starting")

    base_url, port, token = start_server()
    shell = DesktopShell(base_url, port, token)
    shell.run()
