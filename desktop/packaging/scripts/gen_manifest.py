"""Generate latest.json for the CLASSify Desktop release pipeline.

Called by .github/workflows/release.yml after all platform builds finish.
"""

from __future__ import annotations

import argparse
import datetime
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="tag name, e.g. v1.1.0")
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--win-sha", required=True)
    parser.add_argument("--win-size", required=True, type=int)
    parser.add_argument("--mac-sha", required=True)
    parser.add_argument("--mac-size", required=True, type=int)
    parser.add_argument("--lin-sha", required=True)
    parser.add_argument("--lin-size", required=True, type=int)
    parser.add_argument("--out", default="latest.json")
    args = parser.parse_args()

    base = f"https://github.com/{args.repo}/releases/download/{args.version}"
    manifest = {
        "version": args.version.lstrip("v"),
        "released_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "channel": "stable",
        "minimum_upgrade_from": "0.0.0",
        "assets": {
            "windows-x64": {
                "url": f"{base}/CLASSify-Setup-{args.version}-x64.exe",
                "sha256": args.win_sha,
                "size": args.win_size,
            },
            "macos-universal2": {
                "url": f"{base}/CLASSify-{args.version}-macos.zip",
                "sha256": args.mac_sha,
                "size": args.mac_size,
            },
            "linux-x86_64-appimage": {
                "url": f"{base}/CLASSify-{args.version}-linux.tar.gz",
                "sha256": args.lin_sha,
                "size": args.lin_size,
            },
        },
        "notes": f"CLASSify Desktop {args.version}",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out} for {args.version}")


if __name__ == "__main__":
    main()
