# CLASSify Desktop — PyInstaller Packaging

This directory contains the PyInstaller spec files and installer scripts
for building platform-specific CLASSify Desktop installers.

## Build (per OS — PyInstaller cannot cross-compile)

### Windows
```powershell
pip install -e ".[dev,ml,desktop]" pyinstaller
cd frontend && npm ci && npm run build && cd ..
pyinstaller desktop/packaging/pyinstaller/windows.spec --noconfirm --distpath dist --workpath build
# Then compile the installer:
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" desktop\packaging\installers\windows\classify.iss
# Output: dist\installers\CLASSify-Setup-<version>-x64.exe
```

### macOS (build on both arm64 + x86_64, then lipo-merge)
```bash
pip install -e ".[dev,ml,desktop]" pyinstaller
cd frontend && npm ci && npm run build && cd ..
pyinstaller desktop/packaging/pyinstaller/macos.spec --noconfirm --distpath dist --workpath build
```

### Linux
```bash
pip install -e ".[dev,ml,desktop]" pyinstaller
cd frontend && npm ci && npm run build && cd ..
pyinstaller desktop/packaging/pyinstaller/linux.spec --noconfirm --distpath dist --workpath build
```

## CI

Builds are automated via GitHub Actions:
- `.github/workflows/release.yml` — tag-triggered release build (Win + macOS + Linux)
  - Triggers on `git push origin v*` tags or manual dispatch
  - Builds PyInstaller bundle + platform installer per OS
  - Uploads all artifacts + `latest.json` manifest to the GitHub release
  - `latest.json` is consumed by the in-app update checker (`GET /api/system/check-updates`)

See `CI_CD.md` for the full pipeline design.

## Release process

1. Bump version in `pyproject.toml` and `backend/classify_api/routers/system.py` (`APP_VERSION`)
2. Commit: `git commit -m "chore: bump version to X.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push origin main && git push origin vX.Y.Z`
5. The `release.yml` workflow builds all three OS installers and publishes them + `latest.json` to the release
6. Once the repo is public, the update checker fetches `latest.json` from:
   `https://github.com/<org>/<repo>/releases/latest/download/latest.json`

For local-only builds (no CI), run the Windows steps above, then upload
the `.exe` and `latest.json` to the GitHub release manually.

## Update checker

The app's `GET /api/system/check-updates` endpoint fetches `latest.json`
from the stable release URL and compares the `version` field against the
installed `APP_VERSION`. If newer, it returns the download URL for the
user's platform and release notes.

**Important:** The repo must be public (or the release assets must be
publicly accessible) for the update checker to work. If the repo is
private, the manifest URL returns 404.

## Signing secrets (GitHub Actions)

| Secret | Purpose |
|---|---|
| `WINDOWS_CERT_BASE64` | Authenticode PFX cert (base64) |
| `WINDOWS_CERT_PASSWORD` | PFX password |
| `APPLE_DEVELOPER_ID_P12_BASE64` | macOS Developer ID cert (base64) |
| `APPLE_P12_PASSWORD` | macOS cert password |
| `APPLE_API_KEY_BASE64` | Notarization API key |
| `APPLE_API_KEY_ID` | Notarization key ID |
| `APPLE_API_ISSUER` | Notarization issuer ID |
| `GPG_PRIVATE_KEY` | Linux artifact signing (optional) |
| `GPG_PASSPHRASE` | GPG key passphrase |

All signing secrets are only used in the `release` environment, which
requires manual approval before the signing step runs.
