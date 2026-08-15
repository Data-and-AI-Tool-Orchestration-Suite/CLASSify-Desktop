"""Generate the latest.json update manifest for a release.

Usage:
    python scripts/generate_manifest.py --version 1.0.0 --repo org/CLASSify-app

Reads artifact info from the GitHub release API (or CLI args) and produces
a latest.json file that the in-app update check fetches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(
    version: str,
    repo: str,
    artifacts: dict[str, Path],
    notes: str = "",
    channel: str = "stable",
    minimum_upgrade_from: str = "0.0.0",
) -> dict:
    """Generate the latest.json manifest dict."""
    assets: dict[str, dict] = {}
    for platform_key, filepath in artifacts.items():
        if not filepath.exists():
            print(f"Warning: {filepath} does not exist, skipping", file=sys.stderr)
            continue
        sha = compute_sha256(filepath)
        size = filepath.stat().st_size
        url = f"https://github.com/{repo}/releases/download/v{version}/{filepath.name}"
        assets[platform_key] = {
            "url": url,
            "sha256": sha,
            "size": size,
        }

    manifest = {
        "version": version,
        "released_at": datetime.now(UTC).isoformat(),
        "channel": channel,
        "minimum_upgrade_from": minimum_upgrade_from,
        "assets": assets,
        "notes": notes,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate latest.json update manifest")
    parser.add_argument("--version", required=True, help="Release version (e.g. 1.0.0)")
    parser.add_argument("--repo", required=True, help="GitHub repo (org/name)")
    parser.add_argument("--output", default="latest.json", help="Output file path")
    parser.add_argument("--channel", default="stable", choices=["stable", "beta"])
    parser.add_argument("--minimum-upgrade-from", default="0.0.0")
    parser.add_argument("--notes", default="", help="Release notes")
    parser.add_argument("--windows-exe", help="Path to Windows installer .exe")
    parser.add_argument("--macos-dmg", help="Path to macOS .dmg")
    parser.add_argument("--linux-appimage", help="Path to Linux .AppImage")
    parser.add_argument("--linux-deb", help="Path to Linux .deb")
    args = parser.parse_args()

    artifacts: dict[str, Path] = {}
    if args.windows_exe:
        artifacts["windows-x64"] = Path(args.windows_exe)
    if args.macos_dmg:
        artifacts["macos-universal2"] = Path(args.macos_dmg)
    if args.linux_appimage:
        artifacts["linux-x86_64-appimage"] = Path(args.linux_appimage)
    if args.linux_deb:
        artifacts["linux-x86_64-deb"] = Path(args.linux_deb)

    manifest = generate_manifest(
        version=args.version,
        repo=args.repo,
        artifacts=artifacts,
        notes=args.notes,
        channel=args.channel,
        minimum_upgrade_from=args.minimum_upgrade_from,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(manifest, indent=2))
    print(f"Written {output_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
