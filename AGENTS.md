# AGENTS.md — CLASSify Desktop

Quick reference for AI agents working on this codebase.

## Project overview

CLASSify Desktop is a cross-platform (Win/Linux/macOS) local-first desktop
version of the CLASSify-2 web app. It lets users upload CSV data, configure
features, train classification/clustering models, and explore results — all
on local compute with local file storage. See `IMPLEMENTATION_CHECKLIST.md`
for the full build plan, `CI_CD.md` for pipelines, and `ROADMAP.md` for
post-v1 items.

## Tech stack

- **Backend**: Python 3.11+, FastAPI, SQLAlchemy 2.0, Alembic, Pydantic v2,
  structlog. Located in `backend/`.
- **Frontend**: Svelte 4 + Vite + TypeScript, Bootstrap 5, TanStack Table.
  Located in `frontend/`.
- **Desktop shell**: pywebview (system webview). Located in `desktop/`.
- **ML engine**: forked from CLASSify-2, in `backend/ml/`. No S3, no ClearML.

## Common commands

> **Windows / PowerShell**: environment variables must be set with
> `$env:VAR="value";` before the command — the `VAR=value command` syntax
> is bash-only.  All examples below show the bash form; use the PowerShell
> form on Windows, e.g. `$env:CLASSIFY_DEV_MODE="true"; python -m classify_api`.

### Backend

```bash
# Install (editable, with dev deps — API + tooling only, no ML libs)
pip install -e ".[dev]"

# Install with ML deps for local engine development
pip install -e ".[dev,ml]"

# Install with desktop shell (pywebview)
pip install -e ".[dev,desktop]"

# Run the API server (dev mode, port 8000)
CLASSIFY_DEV_MODE=true python -m classify_api
# PowerShell: $env:CLASSIFY_DEV_MODE="true"; python -m classify_api

# Run the desktop shell (native window — needs frontend built first)
cd frontend && npm run build && cd ..
python -m classify_desktop

# Lint + typecheck
ruff format --check backend desktop tests
ruff check backend desktop tests
mypy backend desktop

# Run tests (excludes slow/addon/e2e by default)
pytest tests/backend
# PowerShell: $env:MPLBACKEND="Agg"; pytest tests/backend
```

### Frontend

```bash
cd frontend
npm ci
npm run dev          # Vite dev server on :5173, proxies /api to :8000
npm run build        # production build → frontend/dist/
npm run check        # svelte-check typecheck
npm run lint         # eslint
npm run format:check # prettier
npm test             # vitest
npm run gen-types    # regenerate API types from OpenAPI (needs backend running)
```

### Full local CI

```bash
make ci    # runs lint + typecheck + tests for both backend and frontend
```

### Install ML add-on deps (TabPFN/SDV — not in base)

```bash
pip install -e ".[tabpfn]"   # pulls torch + tabpfn
pip install -e ".[sdv]"      # pulls torch + sdv
```

## Code layout

```
backend/
  classify_api/   → FastAPI app, routers, schemas, ORM, settings
  ml/             → training engine, evaluation, SHAP, synthesis, viz, retest
  storage/        → LocalStorage (replaces S3) + optional encryption
  runner/         → SQLite job queue + subprocess manager (replaces ClearML)
desktop/
  classify_desktop/ → pywebview shell, tray, lifecycle
frontend/
  src/routes/     → Svelte page components
  src/lib/        → API client, shared components, stores
tests/
  backend/        → pytest (unit + integration)
  frontend/       → vitest
  e2e/            → Playwright
```

## Conventions

- Python: ruff format (double quotes), mypy strict, no bare `except`, type
  all function signatures.
- Frontend: prettier (double quotes, 100 cols), eslint, TypeScript strict.
- Commits: conventional commits (`feat:`, `fix:`, `test:`, `chore:`, etc.).
- No comments in code unless asked.
- Tests must pass before merge; coverage gates: backend ≥80%, frontend ≥75%.

## ML engine note

The ML code in `backend/ml/` is a fork of CLASSify-2's
`backend/ml_classifier_trainer/`. The modeling math and defaults should NOT
be changed — only the I/O layer (S3 → local storage, ClearML → local runner).
When porting functions, preserve the exact algorithm logic.

## Frozen builds & add-ons (PyInstaller pitfalls — all hit in production)

These broke v1.2.0 and v1.1.x releases. Read before touching the frozen
build, the jobworker, or the add-on system.

- **Never run `sys.executable -m <module>` or `-c` in frozen apps.**
  `sys.executable` is the app bootloader — it relaunches the whole app and
  ignores the args. The single-instance mutex then exits it with code 0,
  which the caller reads as *success*. This silently faked add-on installs.
  Fix pattern: run pip **in-process** (`pip._internal.cli.main.main(args)`)
  in frozen apps; keep subprocess isolation in dev (`getattr(sys, "frozen")`).
- **Bundling pip**: pip is excluded from analysis, so the bundle needs it
  shipped as *filesystem data* (`collect_data_files("pip", include_py_files=True)`
  + `"pip"` in excludes — NOT collect_submodules; distlib's script templates
  and certifi's CA bundle must be on disk). It also imports stdlib modules
  dynamically — the specs compute pip's full stdlib import set at build time
  (`_collect_stdlib_modules()` in the specs) and bundle every stdlib module +
  submodule via `sys.stdlib_module_names`. If a frozen process reports
  `No module named 'http.client'` / `'unittest.result'` / `'compileall'`,
  this is why.
- **NEVER override `sys._MEIPASS` and never create junctions inside it.**
  Three separate failures came from that idea: (1) `server.py` resolves the
  SPA static mount via `sys._MEIPASS` at request time → overwriting it makes
  the webview load nothing and the app exit silently after `server_started`;
  (2) the frozen importer resolves modules through it *live*, so spawned
  children lose their stdlib; (3) a filesystem junction into `_MEIPASS`
  poisons the whole bundle tree (WinError 448 "untrusted mount point") —
  jobworkers crash at startup with **no output log**. faker instead resolves
  its data paths via `sys.frozen` checks: job-side processes set
  `sys.frozen = False` (jobworker training phase, verification child) so
  faker resolves from the add-on dir. The main app must stay frozen (static
  mount) and never imports faker at module level.
- **jobworker is a separate process.** It must call
  `init_addons()` itself (add-on dir on `sys.path`) and re-apply add-on env
  settings — the main app's boot-time init does not carry over.
- **Add-on dirs are shared.** torch is installed once into
  `<data>/addons/pythonlibs`. Installing one add-on must NOT wipe the dir if
  others are installed; uninstalling one keeps files while others need them
  (the uninstall response explains this). pip runs WITHOUT `--no-cache-dir`
  so the torch wheel is downloaded once across add-ons and reinstalls.
- **Missing output log = the jobworker died before starting.** Check the
  `error` column in `classify.db`'s `jobs` table and
  `%APPDATA%/CLASSify/logs/classify.log`. A job that fails with
  "No models were trained" means the engine skipped every requested model —
  the reason is in that run's output log.

## Releases (how not to break the pipeline again)

- **Version lives in 4 places** — bump all: `pyproject.toml`,
  `APP_VERSION` in `backend/classify_api/routers/system.py`,
  `frontend/package.json` (npm version), `MyAppVersion` in
  `desktop/packaging/installers/windows/classify.iss`.
- **Releases are tag-driven**: pushing a `v*` tag triggers `release.yml`
  (3-OS builds → installer → GitHub release → `latest.json` asset).
  The installer filename is `CLASSify-Setup-<tag>-x64.exe` — the `.iss`
  bakes the tag's `v` prefix in (`OutputBaseFilename`), so the app version
  and the tag must stay in sync.
- **Sync `latest.json`** (repo root mirror) from the generated release
  asset after the workflow finishes.
- **Tool pins**: `ruff` and `mypy` are pinned to tested minors in `dev`.
  Unpinned, a new ruff fails `format --check` on unchanged files and a new
  mypy changes the error set. Bump the pin deliberately, reformat, commit.
- **`dev` extra includes `[ml]`** (self-referential extra). The API imports
  numpy/pandas/joblib/charset-normalizer at module level — those are BASE
  dependencies, not ml-extra. Heavy viz/model imports in the API
  (`ml.retest`, `ml.shap_explain` in `routers/results.py`) must stay
  function-level lazy.
- **`filterwarnings = ["error", ...]`** escalates new deprecation warnings
  into test *collection* errors (this broke CI for starlette's testclient
  change). Add a targeted `ignore::` line in `pyproject.toml`, not CLI
  flags.
- **pip-audit** audits a `pip freeze --exclude-editable` export — it cannot
  handle the project's own editable install under `--strict`.
- **CodeQL**: the repo uses GitHub's *default* CodeQL setup; do not re-add
  a `codeql.yml` workflow (duplicate-setup conflict fails Code Scanning).
- **OpenAPI sync gate**: changing any API schema requires regenerating
  `frontend/src/lib/api/types.ts` (`npm run gen-types` pattern — see
  `ci.yml` openapi-sync job) or CI fails on `git diff --exit-code`.

## Windows / tooling gotchas

- **Never rewrite UTF-8 source files with PowerShell** (`Get-Content -Raw |
  Set-Content`, `>>` redirects): it misreads BOM-less UTF-8 as ANSI —
  double-encoding every em-dash/arrow into visible `â€"` mojibake — and
  writes BOMs (a BOM in `pyproject.toml` breaks `pip install -e .` TOML
  parsing). Use the edit tool or a small Python script.
  `scripts/fix_mojibake.py` repairs the damage repo-wide.
- **PowerShell session variables persist across tool commands** — a stale
  `$port`/`$tab` from an earlier command can silently fake a verification.
  Re-read state (`Get-Process -Id`, fresh HTTP calls) before concluding;
  check `Get-Process` StartTime to know which build a running app is.
- **Replicate CI locally** in a fresh dev-only venv (CI installs `.[dev]`,
  no ml) before pushing: ruff/mypy/pytest versions and warning escalation
  differ from a full dev machine. Keep `.venv-ci/` for this (gitignored).
- **Verify which process serves a port**: `Get-NetTCPConnection -LocalPort
  <p>` → owning PID → `Path`/`StartTime`. During rapid rebuild/relaunch
  cycles a stale instance can outlive its replacement and serve old code.
- **Frozen app debugging**: data dir is `%APPDATA%/CLASSify` — `classify.db`
  (`jobs.error` column), `logs/classify.log`, `addons/pythonlibs`,
  `installed.json`, `addon_settings.json` (TABPFN_TOKEN lives here). Run
  `dist/CLASSify/CLASSify.exe` directly to test a release build locally.
