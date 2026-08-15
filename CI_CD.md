# CI/CD & Testing — CLASSify Desktop

Detailed companion to `IMPLEMENTATION_CHECKLIST.md` (see Phase M and the new **Phase CI**). Covers environments, pipelines, the test pyramid, coverage/quality gates, cross-OS packaging with signing/notarization, and the release process.

Principles:
- **Every PR runs** lint + typecheck + unit + integration tests on all 3 OSes (Win/macOS-arm/macOS-intel via runners; Linux Ubuntu).
- **No torch in base CI** for unit/integration — torch-gated ML regression runs in a separate, slower job that installs the addon deps.
- **Gates are hard**: a red job blocks merge; coverage below threshold blocks merge.
- **Reproducible**: pinned deps, lock files, cached wheels; a release build can be re-run from a tag and produce byte-identical-sha artifacts (best effort; native-build nondeterminism is tracked).
- **Signing is part of CI**, not a manual post-step — secrets live in GitHub Actions, never on a dev machine.

---

## 1. Environments

| Environment | Purpose | Where | Secrets |
|---|---|---|---|
| `dev` | Local development | developer machine | none (local SQLite, no signing) |
| `ci` | Pull-request & branch validation | GitHub Actions runners | none (no signing) |
| `nightly` | Full packaging build + smoke | GitHub Actions runners, scheduled | signing secrets (read-only) |
| `release` | Tagged release → signed installers | GitHub Actions runners, on tag | signing secrets + notarization creds |

App runtime environments (the app itself, not CI):
- **dev build**: FastAPI auto-reload + Vite dev server + webview DevTools; verbose logs; encryption off; temp data dir.
- **staging/preview build**: frozen app pointing at a `CLASSify-preview` data dir; for manual QA before release.
- **production build**: frozen, minified SPA, no DevTools, logs to file, encryption configurable.

---

## 2. Repository branching & versioning

- **Trunk-based**: `main` is always shippable. Short-lived feature branches (`feat/…`, `fix/…`) PR into `main`.
- **Conventional Commits** (`feat:`, `fix:`, `chore:`, `test:`, `refactor:`, `docs:`, `perf:`) → drive semantic versioning via a release tool.
- **Versioning**: CalVer+SemVer hybrid — `v<MAJOR.MINOR.PATCH>` (e.g. `v1.0.0`). Tags trigger release. Internal build metadata: `<version>+<shortsha>.<os><arch>` (e.g. `1.0.0+a1b2c3c.darwin-universal2`).
- **Release branches**: for patch fixes only, `release/1.0.x`, cherry-pick from `main`, tag `v1.0.1`.
- Pre-release tags: `v1.1.0-rc.1` → marked as pre-release on GitHub, `latest.json` not updated until stable.

---

## 3. Workflow files (`.github/workflows/`)

### 3.1 `ci.yml` — runs on every PR and push to `main`
Triggers: `pull_request`, `push: [main]`. Concurrency: cancel superseded runs.

Jobs (matrix `os: [windows-latest, ubuntu-latest, macos-14]`):
1. **backend-quality** (no matrix needed, ubuntu only): ruff format --check, ruff check, mypy --strict backend/, pip-audit.
2. **frontend-quality** (ubuntu only): npm ci, eslint, prettier --check, svelte-check, npm audit.
3. **backend-tests** (matrix): setup-python 3.12, pip cache, `pip install -e ".[dev]"` (base deps only, no torch), pytest (unit + integration, marker `-m "not ml_regression and not addon and not e2e"`), upload coverage to Codecov.
4. **frontend-tests** (ubuntu): vitest run + coverage, upload coverage.
5. **openapi-sync** (ubuntu): start backend, generate `frontend/src/lib/api/types.ts` from OpenAPI, `git diff --exit-code` — fails if generated types are stale (forces regen on API changes).
6. **ml-regression** (ubuntu only, allowed to be slow): install addon deps (torch CPU + tabpfn + sdv) in a separate venv, run `pytest -m ml_regression` against golden datasets. ~15–25 min. Allowed to be a separate required check.
7. **build-smoke** (matrix): PyInstaller `--onedir` build (unsigned), then run a headless smoke test (Phase M2/K10 mini) to confirm the frozen app boots and the job subprocess entry point works. macOS matrix uses both `macos-14` (arm) and `macos-13` (intel) to validate both arches.

Quality gates (branch protection, required):
- backend-quality, frontend-quality, openapi-sync, backend-tests (all 3 OS), frontend-tests, ml-regression, build-smoke (all matrix legs).

### 3.2 `nightly.yml` — scheduled full packaging
Triggers: `schedule: 0 5 * * *` + `workflow_dispatch`.
Jobs:
- For each `os/arch` leg: full PyInstaller build → sign (Win/mac) → notarize (mac) → run the full clean-VM smoke suite → upload artifact with 7-day retention.
- Aggregates a `nightly-summary` job posting pass/fail to a Slack webhook (optional).
Purpose: catch signing/notarization rot and native-build regressions before release day.

### 3.3 `release.yml` — on tag `v*`
Triggers: `push: tags: ['v*']`.
Jobs (parallel per OS/arch, then aggregate):
1. **build-and-sign** per OS — same as nightly but promotes artifacts.
2. **notarize** (macOS) — `xcrun notarytool submit` + staple; fail on rejection.
3. **smoke-test** signed artifacts on clean runner images.
4. **release-publish** (single job, `needs: [all build-and-sign + smoke]`):
   - Create/Update the GitHub Release (softprops/action-gh-release) with artifacts.
   - Compute `sha256sum` for every artifact; upload `<artifact>.sha256`.
   - Generate `latest.json` manifest (schema in §6) and upload as a release asset; only for stable (non-prerelease) tags.
   - Tag the Docker image of the CI runner version for traceability (internal).
5. **post-release-verify**: download the published artifacts fresh, re-run smoke, confirm SHA256 matches → confirms release assets are what we signed.

---

## 4. Secrets management

Stored as GitHub Actions repository secrets (encrypted at rest by GitHub):
- `WINDOWS_CERT_BASE64` / `WINDOWS_CERT_PASSWORD` — Authenticode PFX.
- `APPLE_DEVELOPER_ID_P12_BASE64` / `APPLE_P12_PASSWORD` — Developer ID Application cert.
- `APPLE_ID` / `APPLE_APP_SPECIFIC_PASSWORD` / `APPLE_TEAM_ID` — notarization credentials.
- `APPLE_API_KEY_BASE64` / `APPLE_API_KEY_ID` / `APPLE_API_ISSUER` — preferred notarytool API-key auth (store-batch).
- `GPG_PRIVATE_KEY` / `GPG_PASSPHRASE` — Linux artifact signing (optional).
- `SLACK_WEBHOOK` — release/nightly notifications (optional).
- `CODECOV_TOKEN` — coverage upload.

Rules:
- Signing secrets only referenced by `nightly.yml` and `release.yml` (environment `release` with required reviewers for `release.yml`).
- Never echo secrets; use `::add-mask::` if a derived value is printed.
- Certs rotate on their own schedule; track expiry in a `SECURITY.md` note + a scheduled reminder issue.
- No secrets needed for the unsigned `ci.yml` builds.

---

## 5. Caching strategy

- **pip**: `actions/setup-python` built-in cache keyed on `pyproject.toml` + `requirements/*.txt`. Separate cache key suffix for `+torch` (ml-regression job) to avoid poisoning.
- **npm**: `actions/setup-node` cache keyed on `frontend/package-lock.json`.
- **PyInstaller build artifacts**: cache `build/` per OS + Python version to speed incremental builds (keyed on `pyproject.toml` hash + spec hash). Bust on dependency changes.
- **torch wheels** (ml-regression + addon builds): cache the `pip download` of CPU torch (large) to avoid re-fetching every run.
- **HF model weights**: cache `~/.cache/huggingface` for TabPFN smoke tests to avoid re-downloading.
- **Homebrew** (macOS): cache `~/Library/Caches/Homebrew`.

Cache restore keys use a prefix + hash fallback so partial cache hits still help.

---

## 6. Release artifacts & `latest.json` manifest

Artifacts per release:
- `CLASSify-Setup-<version>-x64.exe` (Windows installer, signed)
- `CLASSify-<version>-x64.msi` (optional alt)
- `CLASSify-<version>-universal2.dmg` + `CLASSify-<version>-universal2.zip` (macOS, notarized + stapled)
- `CLASSify-<version>-x86_64.AppImage` (Linux)
- `classify-<version>-amd64.deb` (Linux)
- `classify-<version>-linux-x86_64.tar.gz` (Linux portable)
- `latest.json`, plus `<artifact>.sha256` for each
- `RELEASE_NOTES.md` (generated from conventional commits since last tag)

`latest.json` schema (consumed by the in-app "Check for updates" now, and the future auto-updater):
```json
{
  "version": "1.0.0",
  "released_at": "2026-08-15T12:00:00Z",
  "channel": "stable",
  "minimum_upgrade_from": "0.9.0",
  "assets": {
    "windows-x64": {
      "url": "https://github.com/<org>/<repo>/releases/download/v1.0.0/CLASSify-Setup-1.0.0-x64.exe",
      "sha256": "…",
      "size": 723456789
    },
    "macos-universal2": { "url": "…", "sha256": "…", "size": 912345678 },
    "linux-x86_64-appimage": { "url": "…", "sha256": "…", "size": 689000000 }
  },
  "notes": "…changelog markdown…"
}
```
The app's update check fetches `latest.json` from a stable raw URL (GitHub release asset or a CDN mirror), compares `version` to installed, and if newer links the user to the asset for their platform. (Real auto-update is a roadmap item.)

---

## 7. Test pyramid

```
        ┌───────────┐
        │   smoke    │  clean-VM launch+train on each signed artifact (release/nightly)
        ├───────────┤
        │    e2e     │  Playwright full flow on a frozen dev build (CI build-smoke leg)
        ├───────────┤
        │ ML regre-  │  golden-dataset training; artifact + metric asserts (ml-regression job)
        │  ssion     │
        ├───────────┤
        │ integration│  routers + storage + runner with real SQLite + temp FS (backend-tests)
        ├───────────┤
        │   unit     │  pure functions, schemas, storage, queue logic (backend-tests + frontend-tests)
        └───────────┘
```

| Layer | Scope | Count target | Speed budget |
|---|---|---|---|
| unit | `column_types`, `options`, queue/manager logic, pydantic schemas, frontend pure utils + components | grows with features | <90s total |
| integration | each FastAPI router against temp appdata + SQLite; runner subprocess on a tiny job; storage round-trips | one+ per endpoint | <3 min |
| ML regression | RF/LogReg/XGBoost/KMeans × {binary, multiclass, clustering, missing, categorical} golden sets; SHAP + viz artifact asserts; retest | ~20 cases | <25 min (parallelized) |
| e2e | upload → configure → map → options → train → results → re-test → download | 1 core flow + 2 variants | <6 min |
| smoke | frozen signed app boots on clean OS image, trains RF, views results | 1 per OS/arch | <10 min each |

### 7.1 Backend testing detail
- **pytest** layout mirrors `backend/`: `tests/backend/{unit,integration,ml_regression}`.
- **Fixtures**: `tmp_appdata` (monkeypatch `resolve_appdata` to a tmp path), `isolated_storage`, `fresh_db` (in-memory or tmp SQLite + `alembic upgrade head`), `sample_csv` (the golden datasets under `tests/fixtures/datasets/`).
- **Markers**: `@pytest.mark.ml_regression`, `@pytest.mark.addon`, `@pytest.mark.slow`, `@pytest.mark.e2e`. Default `pytest` excludes `ml_regression`+`addon`+`e2e` so unit/integration stay fast.
- **HTTP**: use `httpx.AsyncClient` with FastAPI `TestClient` (sync) for routers.
- **Subprocess runner tests** (`tests/backend/integration/test_runner.py`): enqueue → assert `running` → assert `succeeded` + artifacts; cancel path; crash path (worker exits non-zero → `failed` + partial artifacts).
- **Coverage gate**: `--cov=backend --cov-fail-under=80` on the unit+integration run; `ml/` measured but torch-gated lines excluded via `# pragma: no cover` + branch exclusions.

### 7.2 ML regression detail
- Golden datasets versioned in `tests/fixtures/datasets/` with a `README.md` documenting expected properties.
- Each case asserts: `report.csv` non-empty, `results.json` has expected metric keys, expected `viz/` PNGs exist and are non-zero, joblib model loadable, SHAP rows CSV present (when enabled), and key metrics within documented ranges (e.g. RF accuracy > 0.6 on the binary fixture). Ranges are intentionally loose to avoid flakiness.
- Run on CPU only. TabPFN/SDV cases are `@pytest.mark.addon` and run only in the ml-regression job.
- Snapshot the `results.json` metric keys (not values) to catch silent schema regressions.

### 7.3 Frontend testing detail
- **vitest** + `@testing-library/svelte` for components (ColumnPreview, ClassMapper, MLOptionsForm, ResultsTable, JobStatus).
- **MSW (Mock Service Worker)** to mock the typed API client in component tests — keeps them decoupled from a running backend.
- **Visual regression**: `@vitest/snapshot` on the rendered design-system storybook page (or a `+design` route) to catch CSS drift from the web version.
- **a11y**: `axe-svelte` checks in component tests for the mapper modal and options form (preserve web ARIA work).
- **Coverage gate**: vitest coverage `--coverage.thresholds.lines=75`.

### 7.4 e2e (Playwright)
- One spec: `full-flow.spec.ts` — launch the **frozen dev build** (built in CI build-smoke leg), upload `tiny.csv`, pick class column, accept defaults, train RF, wait for `Processed` via SSE, open results, assert table + at least one viz renders, run re-test, download report.csv.
- Linux runs under `xvfb-run`; Windows/macOS run headless via Playwright's bundled browsers (note: tests target the *system webview* build, so we run a Chromium fallback e2e against the FastAPI server directly for cross-browser logic, plus a lighter native-webview smoke).
- Two variants: `multiclass-flow`, `clustering-flow`.
- e2e is a required check on `main` but allowed to be `optional` on draft PRs.

### 7.5 Clean-VM smoke (release/nightly)
- Run on GitHub-hosted clean runner images (no prior app install).
- Steps: download artifact, install, launch, first-run wizard auto-accepted via a `--smoke` flag, upload `tiny.csv`, train RF, assert results, install TabPFN addon, train TabPFN, assert results, capture logs on failure and attach to the run.
- This is the final gate before a release is published.

---

## 8. Quality gates (branch protection — all required on `main`)

- `backend-quality` (ruff + mypy strict + pip-audit clean)
- `frontend-quality` (eslint + prettier + svelte-check + npm audit)
- `openapi-sync` (generated types current)
- `backend-tests` (×3 OS, coverage ≥80%)
- `frontend-tests` (coverage ≥75%)
- `ml-regression` (all golden cases pass)
- `build-smoke` (×3 OS frozen app boots)
- `e2e` (Playwright full flow)
- `lint-pr-title` (conventional commit) — non-blocking warning
- Code owner review on `backend/ml/` and `desktop/packaging/`

`release.yml` additionally requires the `release` environment's required reviewers to approve before signing secrets are used.

---

## 9. Packaging pipeline detail (per OS)

### 9.1 Common
- `python -m pip install -r requirements/runtime.txt` (locked, no torch) into a clean venv.
- `cd frontend && npm ci && npm run build` → `frontend/dist/` mounted as FastAPI static.
- `pyinstaller desktop/packaging/pyinstaller/<os>.spec --noconfirm --distpath dist/<os>`.
- Three entry points declared in the spec: `CLASSify`, `classify-jobworker`, `classify-api`.
- Audit hidden imports: run a post-build import-probe that imports every top-level module path used at runtime; fail on `ModuleNotFoundError`.

### 9.2 Windows (`windows-latest`)
- Spec builds `--onedir` then Inno Setup compiles `CLASSify-Setup-<ver>-x64.exe`.
- Sign `CLASSify.exe`, the jobworker exe, and the installer with `signtool` using the PFX from secrets (`WINDOWS_CERT_BASE64`).
- Verify: `signtool verify /pa /v` on each signed binary.
- File association: `.csv` → opens CLASSify (optional).

### 9.3 macOS (`macos-14` arm + `macos-13` intel → Universal2)
- Build arm64 and x86_64 `--onedir` artifacts separately (PyInstaller can't cross-compile).
- `lipo`-merge the main binary + jobworker + api binaries into universal2.
- Assemble `CLASSify.app` bundle (`Info.plist`, icon, entitlements plist: only network for update-check + HF weights; no camera/mic).
- `codesign --deep --options runtime --entitlements … --sign "Developer ID Application: …"` each binary + bundle.
- `xcrun notarytool submit … --keychain-profile classify` (API key auth) → poll `--wait`; on rejection, fetch the JSON log and fail the job with it attached.
- `xcrun stapler staple CLASSify.app`.
- Verify: `spctl --assess --verbose=4 --type execute CLASSify.app` and `codesign --verify --deep --strict`.
- Build the `.dmg` (`create-dmg` or `hdiutil`) from the stapled `.app`; also zip for direct download.
- Both `.dmg` and `.zip` are notarization-relevant; staple the `.app` before packaging into dmg.

### 9.4 Linux (`ubuntu-22.04` for broad glibc compat)
- `--onedir`; build AppImage via `appimagetool` with a `.AppDir` (`.desktop` + icon).
- Build `.deb` with `fpm`; build `.tar.gz` portable.
- Optional GPG sign all three with `GPG_PRIVATE_KEY`.
- Verify: `dpkg-deb -I` for the deb; AppImage `--appimage-extract-and-run` smoke.

---

## 10. Dependency & supply-chain hygiene

- **Lock files**: `requirements/runtime.txt` + `requirements/dev.txt` pinned (`pip-compile`), and `requirements/addon-tabpfn.txt` / `addon-sdv.txt` pinned for addons. `package-lock.json` committed.
- **pip-audit / npm audit** in CI; new advisories fail the build (or open a tracked issue for `dev`-only advisories).
- **SBOM**: generate CycloneDX SBOM (`cyclonedx-bom` for py, `@cyclonedx/cyclonedx-npm` for npm) per release; attach to the GitHub release.
- **Provenance**: enable GitHub Actions artifact attestations (`actions/attest-build-provenance`) on release artifacts for SLSA L3 build provenance.
- **PyInstaller isolation**: build in a minimal venv to avoid accidentally bundling dev tools.

---

## 11. Observability of CI itself

- Each workflow job sets a clear step name; failures attach the relevant log excerpt as a job summary (`$GITHUB_STEP_SUMMARY`).
- `nightly.yml` posts a markdown summary table (OS × gate) to the run.
- Track flaky tests: if an e2e/ml-regression test fails on `main` (which should be stable), auto-open an issue with the run link.
- Build-time dashboard: track per-OS build duration over time to catch dependency bloat (e.g. a new dep doubling build time).

---

## 12. Rollback / hotfix

- A bad release: delete/convert the GitHub release to a draft (keeps the tag), regenerate `latest.json` to point back at the prior good version's assets, republish. The app's "Check for updates" then sees the older-but-current version.
- Hotfix: branch `release/1.0.x` from the bad tag, fix, tag `v1.0.1`, run `release.yml`.
- Keep the last 3 minor versions' installers downloadable for users who can't upgrade immediately.

---

## 13. Local CI parity

- A `make ci` target (and `scripts/ci-local.ps1` for Windows) runs the same lint/typecheck/test commands as `ci.yml` so devs can reproduce CI locally before pushing.
- Act-based local workflow runner (`act -W .github/workflows/ci.yml`) documented for testing workflow changes without a push.
