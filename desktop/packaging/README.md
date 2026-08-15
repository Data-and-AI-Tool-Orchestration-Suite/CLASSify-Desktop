# CLASSify Desktop — PyInstaller Packaging

This directory contains the PyInstaller spec files and installer scripts
for building platform-specific CLASSify Desktop installers.

## Build (per OS — PyInstaller cannot cross-compile)

### Windows
```powershell
pip install -e ".[dev,ml,desktop]"
cd frontend && npm ci && npm run build && cd ..
pyinstaller desktop/packaging/pyinstaller/windows.spec --noconfirm
# Then compile the installer:
iscc desktop/packaging/installers/windows/classify.iss
```

### macOS (build on both arm64 + x86_64, then lipo-merge)
```bash
pip install -e ".[dev,ml,desktop]"
cd frontend && npm ci && npm run build && cd ..
pyinstaller desktop/packaging/pyinstaller/macos.spec --noconfirm
```

### Linux
```bash
pip install -e ".[dev,ml,desktop]"
cd frontend && npm ci && npm run build && cd ..
pyinstaller desktop/packaging/pyinstaller/linux.spec --noconfirm
```

## CI

Builds are automated via GitHub Actions:
- `.github/workflows/release.yml` — tag-triggered signed release
- `.github/workflows/nightly.yml` — nightly signed builds

See `CI_CD.md` for the full pipeline design.

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
