"""Desktop shell entry point — boots the API server and opens the webview.

Usage:
    python -m classify_desktop          # dev mode (needs CLASSIFY_VITE_PORT for hot reload)
    classify                            # console script (production, frozen)
"""

from __future__ import annotations

import sys


def main() -> None:
    """Console-script entry point (``classify``)."""
    try:
        from classify_desktop.shell import run_shell

        run_shell()
    except ImportError as e:
        if "webview" in str(e):
            print(
                "pywebview is not installed. Install with: pip install classify-desktop[desktop]",
                file=sys.stderr,
            )
        else:
            raise
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
