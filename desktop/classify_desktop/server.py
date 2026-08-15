"""Server launcher — starts the FastAPI backend on a random localhost port.

Generates a random bearer token for local-only auth, waits for the health
endpoint to respond, then returns the URL for the webview to load.
"""

from __future__ import annotations

import secrets
import socket
import sys
import time
from pathlib import Path
from typing import Any

import structlog
import uvicorn

from classify_api.settings import get_settings

log = structlog.get_logger()


def find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def generate_token() -> str:
    """Generate a random bearer token for local API auth."""
    return secrets.token_urlsafe(32)


def wait_for_health(url: str, timeout: float = 30.0) -> bool:
    """Poll the health endpoint until it responds or timeout."""
    import urllib.request

    health_url = f"{url}/api/system/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def start_server(port: int | None = None) -> tuple[str, int, str]:
    """Start the FastAPI server in a background thread.

    Returns (base_url, port, token).  The caller is responsible for
    keeping the main thread alive while the server runs.
    """
    import threading

    settings = get_settings()
    settings.ensure_dirs()

    if port is None:
        port = find_free_port()

    token = generate_token()

    config = uvicorn.Config(
        "classify_api.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        reload=False,
        access_log=False,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn")
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    if not wait_for_health(base_url):
        raise RuntimeError("Server failed to start within timeout")

    log.info("shell.server_started", port=port)
    return base_url, port, token


def get_spa_url(base_url: str, token: str) -> str:
    """Build the URL for the webview to load.

    The shell always loads the SPA from the backend, which serves
    frontend/dist/ as static files.  For pure frontend development
    (hot-reload), run Vite separately and open a browser.
    """
    return f"{base_url}/"


def is_frozen() -> bool:
    """Check if we're running inside a PyInstaller bundle."""
    import sys

    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_app_icon() -> str | None:
    """Return the path to the app icon, if available."""
    base = Path(sys._MEIPASS) if is_frozen() else Path(__file__).resolve().parent.parent / "assets"  # noqa: SLF001, SIM108

    for ext in (".png", ".ico", ".icns"):
        icon = base / f"classify_icon{ext}"
        if icon.exists():
            return str(icon)
    return None


# Keep a reference to prevent garbage collection
_server_instance: Any = None
