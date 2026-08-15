# CLASSify Desktop

A cross-platform (Windows / Linux / macOS) desktop application for training
classification and clustering models on tabular data — a local-first version
of the [CLASSify-2](../CLASSify-2) web application.

## Key differences from the web version

| | Web (CLASSify-2) | Desktop |
|---|---|---|
| Storage | S3 | Local filesystem |
| Job orchestration | ClearML queue | In-process subprocess runner |
| Database | PostgreSQL | SQLite |
| Frontend | PHP / Apache | Svelte SPA |
| Backend | Flask + PHP | FastAPI |
| Auth | CiLogon SSO + multi-tenant | Single user (no auth) |
| Compute | DGX GPU cluster | Local CPU (GPU add-on planned) |

## Documentation

- [`IMPLEMENTATION_CHECKLIST.md`](IMPLEMENTATION_CHECKLIST.md) — full build plan
- [`CI_CD.md`](CI_CD.md) — CI/CD pipelines, test pyramid, signing & release
- [`ROADMAP.md`](ROADMAP.md) — post-v1 deferred features
- [`AGENTS.md`](AGENTS.md) — quick reference for developers and AI agents

## Quick start (development)

### Prerequisites

- Python 3.11+
- Node.js 20+
- A C/C++ compiler (for hdbscan, pyamg, numba native builds)

### Backend

```bash
pip install -e ".[dev]"
CLASSIFY_DEV_MODE=true python -m classify_api
# PowerShell: $env:CLASSIFY_DEV_MODE="true"; python -m classify_api
# API now running at http://127.0.0.1:8000
```

### Frontend (development with hot-reload)

```bash
cd frontend
npm ci
npm run dev
# SPA at http://localhost:5173, proxies /api → :8000
```

### Desktop shell (native window)

```bash
cd frontend && npm run build        # build SPA first
python -m classify_desktop          # opens native window
```

### Windows convenience script

```powershell
.\scripts\dev.ps1              # backend only
.\scripts\dev.ps1 -Frontend    # backend + Vite (hot-reload in browser)
.\scripts\dev.ps1 -Shell       # native desktop window (needs frontend built)
```

### Frontend

```bash
cd frontend
npm ci
npm run dev
# SPA now running at http://localhost:5173 (proxies /api → :8000)
```

### Run tests

```bash
make ci   # lint + typecheck + test (backend + frontend)
```

## License

GPL-3.0-or-later (inherited from CLASSify-2).
