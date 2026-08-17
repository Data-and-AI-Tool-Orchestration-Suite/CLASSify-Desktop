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

## Acknowledgements

CLASSify Desktop was developed at the **Center for Applied AI at the University of Kentucky**.

### Lead Developer

**Aaron D. Mullen** — [LinkedIn](https://www.linkedin.com/in/aaron-mullen-5706761b8)

### Publication

This work is described in:

> Mullen AD, Armstrong SE, Talbert J, Bumgardner VKC. **CLASSify: A Web-Based Tool for Machine Learning.** *AMIA Jt Summits Transl Sci Proc.* 2024 May 31;2024:364–373. PMID: 38827105.
>
> [https://pmc.ncbi.nlm.nih.gov/articles/PMC11141843/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11141843/)

If you use CLASSify in your research, please cite the publication above.

### Funding & Support

This project was supported by:

- **University of Kentucky Center for Clinical and Translational Science (CCTS)** — NIH Clinical and Translational Science Award (CTSA) grant [UL1TR001998](https://reporter.nih.gov/search/EW3ePp7pbU2r6Q1Y5mWRYg/project-details/10548576)
- **University of Kentucky College of Medicine Office of Research**

## License

GPL-3.0-or-later (inherited from CLASSify-2).
