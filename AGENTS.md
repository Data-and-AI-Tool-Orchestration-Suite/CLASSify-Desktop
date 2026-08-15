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
