"""Generate latest.json for the CLASSify Desktop release pipeline.

Two modes:
  --from-release  Read asset names/sizes/sha256 digests from the GitHub
                  release (used by .github/workflows/release.yml).
  explicit        Pass hashes/sizes via flags (local testing).

The GitHub API exposes a computed ``digest`` (``sha256:<hex>``) for every
uploaded asset, so the build jobs never need to pass hashes around.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import urllib.request

ASSET_NAMES = {
    "windows-x64": "CLASSify-Setup-{tag}-x64.exe",
    "macos-universal2": "CLASSify-{tag}-macos.zip",
    "linux-x86_64-appimage": "CLASSify-{tag}-linux.tar.gz",
}


def fetch_release_assets(repo: str, tag: str) -> dict[str, dict[str, object]]:
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as resp:
        release = json.loads(resp.read().decode("utf-8"))
    return {a["name"]: a for a in release.get("assets", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="tag name, e.g. v1.1.0")
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--from-release", action="store_true")
    parser.add_argument("--out", default="latest.json")
    parser.add_argument("--win-sha")
    parser.add_argument("--win-size", type=int)
    parser.add_argument("--mac-sha")
    parser.add_argument("--mac-size", type=int)
    parser.add_argument("--lin-sha")
    parser.add_argument("--lin-size", type=int)
    args = parser.parse_args()

    tag = args.version
    base = f"https://github.com/{args.repo}/releases/download/{tag}"

    assets: dict[str, dict[str, object]] = {}
    if args.from_release:
        release_assets = fetch_release_assets(args.repo, tag)
        for platform, name_template in ASSET_NAMES.items():
            name = name_template.format(tag=tag)
            asset = release_assets.get(name)
            if asset is None:
                raise SystemExit(f"asset {name!r} not found on release {tag}")
            digest = str(asset.get("digest", ""))
            assets[platform] = {
                "url": f"{base}/{name}",
                "sha256": digest.split(":", 1)[1] if ":" in digest else "",
                "size": int(asset.get("size", 0)),
            }
    else:
        for platform, sha_flag, size_flag, size_value in (
            ("windows-x64", args.win_sha, "--win-size", args.win_size),
            ("macos-universal2", args.mac_sha, "--mac-size", args.mac_size),
            ("linux-x86_64-appimage", args.lin_sha, "--lin-size", args.lin_size),
        ):
            if sha_flag is None or size_flag is None:
                raise SystemExit(f"{sha_flag or size_flag} is required without --from-release")
            name = ASSET_NAMES[platform].format(tag=tag)
            assets[platform] = {
                "url": f"{base}/{name}",
                "sha256": sha_flag,
                "size": size_value,
            }

    manifest = {
        "version": tag.lstrip("v"),
        "released_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "channel": "stable",
        "minimum_upgrade_from": "0.0.0",
        "assets": assets,
        "notes": f"CLASSify Desktop {tag}",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out} for {tag}")


if __name__ == "__main__":
    main()
