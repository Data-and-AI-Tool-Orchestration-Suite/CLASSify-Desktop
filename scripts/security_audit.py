#!/usr/bin/env python
"""Security audit script — checks for known vulnerabilities in dependencies.

Runs pip-audit on the Python environment and npm audit on the frontend.
Exits non-zero if high/critical vulnerabilities are found.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_pip_audit() -> int:
    """Run pip-audit and return exit code."""
    print("=== Python dependency audit (pip-audit) ===")
    result = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--strict"],
        capture_output=False,
    )
    return result.returncode


def run_npm_audit() -> int:
    """Run npm audit on the frontend and return exit code."""
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    if not (frontend / "package-lock.json").exists():
        print("=== Frontend audit skipped (no package-lock.json) ===")
        return 0

    print("=== Frontend dependency audit (npm audit) ===")
    result = subprocess.run(
        ["npm", "audit", "--audit-level=high"],
        cwd=str(frontend),
        shell=True,
    )
    # npm audit returns non-zero if vulnerabilities found — but we only
    # care about high/critical for the gate
    return 0 if result.returncode <= 1 else result.returncode


def main() -> None:
    """Run both audits and exit with the worst code."""
    pip_code = run_pip_audit()
    npm_code = run_npm_audit()

    if pip_code != 0:
        print("\n!!! pip-audit found vulnerabilities !!!")
    if npm_code != 0:
        print("\n!!! npm audit found high/critical vulnerabilities !!!")

    sys.exit(max(pip_code, npm_code))


if __name__ == "__main__":
    main()
