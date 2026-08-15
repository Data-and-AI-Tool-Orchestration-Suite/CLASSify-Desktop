# CLASSify Desktop — Implementation Checklist

Authoritative build plan. Decisions are locked at the top; everything below follows from them.
Track progress by checking items off. Each phase lists tasks + an acceptance gate.

Companion docs: **`CI_CD.md`** (pipelines, test pyramid, signing/notarization, release process) and **`ROADMAP.md`** (post-v1 deferred items).

---

## 0. Locked decisions

| Area | Decision |
|---|---|
| Frontend | **Full Svelte + Vite SPA** (TypeScript). Reuse web visual design/CSS so it feels the same; rebuild interaction layer properly. matplotlib→PNG chart generation stays server-side. |
| Backend | **FastAPI** + Uvicorn (localhost, single worker). SQLAlchemy 2.0 (sync) + Alembic. pydantic-settings. structlog. Pydantic v2 schemas. |
| ML engine | **Fork/copy** CLASSify-2's `models.py` / `synthesize.py` / `visualization.py` / `retest_file.py` into `backend/ml/`, refactored: S3→local FS, ClearML removed. Accepted drift from web. |
| Heavy deps | **TabPFN + SDV are downloadable add-ons** (pull torch). Base installer has the 10 sklearn/xgboost/shap/hdbscan models only. Torch-gated import guards + addon registry. |
| GPU | **CPU-only v1.** GPU torch = a later add-on. |
| Storage | Local FS under appdata. **Optional encryption at rest, off by default** (SQLCipher DB + optional AES-GCM data dir). |
| Jobs | SQLite-backed queue; **one job at a time**; each job runs in an **isolated subprocess** (clean cancel + crash isolation). SSE progress. |
| Shell | **pywebview** (system WebView2/WKWebView/WebKitGTK). **Tray-resident**: window close with active jobs → prompt "Keep running in tray / Stop jobs & quit"; tray menu has full Quit (confirms if jobs active). |
| Packaging | PyInstaller `--onedir`. **macOS Universal2.** Authenticode (Win) + Developer ID notarize (mac). **Manual update-check for v1** (links to releases); real auto-update post-v1. |
| Code quality | ruff + mypy (backend); eslint + prettier + vitest (frontend); pytest + pytest-asyncio; Playwright e2e. GitHub Actions matrix CI. |

### Stack & versions (pin in `pyproject.toml` / `package.json`)
- Python 3.11+ (3.12 target). Node 20 LTS.
- fastapi, uvicorn[standard], sqlalchemy>=2, alembic, pydantic>=2, pydantic-settings, structlog, python-multipart.
- scikit-learn, scipy, numpy, pandas, xgboost, shap, hdbscan, pyamg, numba, matplotlib, plotly, imbalanced-learn, optuna, joblib, charset-normalizer, reportlab, xlsxwriter, openapi-typescript (dev).
- **No** boto3, s3transfer, clearml, clearml-agent, alembic-only-for-clearml, flask (in runtime).
- Frontend: svelte 5, vite, typescript, @tanstack/table-svelte, svelte-multiselect, bootstrap 5 (CSS only), sass. Dev: vitest, @testing-library/svelte, eslint, prettier, openapi-typescript.

---

```
CLASSify-app/
├─ desktop/                       # pywebview shell + packaging
│  ├─ classify_desktop/
│  │  ├─ __main__.py              # entry: boot server + webview + tray
│  │  ├─ shell.py                 # window/tray/lifecycle (close prompt)
│  │  ├─ server.py                # spawn uvicorn on 127.0.0.1:<randport>, bearer token
│  │  └─ single_instance.py       # lock file / named mutex per OS
│  └─ packaging/
│     ├─ pyinstaller/<os>.spec
│     └─ installers/{nsis, inno, dmg, appimage}
├─ backend/
│  ├─ classify_api/
│  │  ├─ main.py                  # FastAPI app, routers, SPA static mount, CORS off (local)
│  │  ├─ settings.py              # pydantic-settings -> settings.json
│  │  ├─ db.py                    # engine, Session, sqlite+SQLCipher
│  │  ├─ orm/                     # Report, Result, Job, Setting, Action (local history)
│  │  ├─ schemas/                 # pydantic request/response models
│  │  ├─ routers/                 # datasets, jobs, results, settings, addons, ml_options, system
│  │  ├─ services/                # dataset_proc, storage_facade, results_reader
│  │  └─ deps.py
│  ├─ ml/
│  │  ├─ engine.py                # trainer() (refactored models.py)
│  │  ├─ evaluate.py              # getmodelstats/clusteringstats/CV
│  │  ├─ tuning.py                # optuna objective()
│  │  ├─ shap_explain.py          # SHAP beeswarm + per-row CSV
│  │  ├─ synthesize.py            # SDV (addon-gated)
│  │  ├─ visualize.py             # matplotlib chart generators
│  │  ├─ retest.py                # retest_file.py refactored
│  │  ├─ options.py               # ml-options dict (supervised + unsupervised)
│  │  ├─ backends.py              # addon registry: is_available(), require()
│  │  ├─ args.py                  # typed TrainingArgs dataclass
│  │  └─ column_types.py          # get_column_types_internal + change logic
│  ├─ storage/
│  │  ├─ base.py                  # Storage Protocol
│  │  ├─ local.py                 # LocalStorage (dir-per-report)
│  │  └─ encrypted.py             # AES-GCM streaming wrapper (optional)
│  ├─ runner/
│  │  ├─ queue.py                 # SQLite job queue (enqueue, peek, status)
│  │  ├─ manager.py               # spawns subprocess, monitors, cancels
│  │  ├─ jobworker.py             # subprocess entry: loads args, calls engine.trainer
│  │  ├─ progress.py              # progress file + SSE source
│  │  └─ cancellation.py          # cancel flag (file/Event), process-group kill
│  └─ migrations/                 # alembic versions
├─ frontend/
│  ├─ src/
│  │  ├─ routes/                  # +datasets, +datasets/[id]/prepare, +results, +results/[id], +settings, +addons
│  │  ├─ lib/
│  │  │  ├─ api/                  # generated types + typed client (fetch)
│  │  │  ├─ components/           # ColumnPreview, ClassMapper, MLOptionsForm, ResultsTable, VizGallery, ShapRowExplorer, OutputLog, JobStatus
│  │  │  └─ stores/               # jobs, datasets, settings, toasts
│  │  └─ app.css / theme          # ported CSS variables from web
│  ├─ static/{css,img}
│  └─ package.json, vite.config.ts, svelte.config.js
├─ addons/
│  ├─ tabpfn_addon/               # manifest + install script (pip --target)
│  └─ sdv_addon/
├─ tests/{backend,frontend,e2e}
├─ scripts/{build,sign,release}
├─ .github/workflows/{ci,nightly,release}.yml
├─ pyproject.toml                 # ruff, mypy, pytest config
├─ IMPLEMENTATION_CHECKLIST.md    # this file
├─ CI_CD.md                       # pipelines, test pyramid, signing/release
├─ ROADMAP.md                     # post-v1 deferred items
└─ AGENTS.md / README.md
```

---

## Phase A — Repo bootstrap & tooling

- [ ] A1. Init `CLASSify-app/` git repo, `.gitignore` (pyenv, node_modules, build/, dist/, appdata-runtime/, *.joblib, *.spec out).
- [ ] A2. `pyproject.toml`: project metadata, ruff (line-length 100, select E/F/I/UP/B/SIM), mypy (strict on `backend/`), pytest, pytest-asyncio.
- [ ] A3. Backend package skeleton: `backend/classify_api/main.py` with a health `GET /api/system/health` returning `{app, version, os}`. Uvicorn runnable via `python -m classify_api` on `127.0.0.1`.
- [ ] A4. Frontend skeleton: `npm create vite@latest frontend -- --template svelte-ts`. Add bootstrap CSS, sass, @tanstack/table-svelte, svelte-multiselect. `npm run dev` proxies `/api` to backend.
- [ ] A5. OpenAPI type generation: `openapi-typescript` script pulls `http://localhost:8000/openapi.json` → `frontend/src/lib/api/types.ts`. Wire into a typed `apiClient` (fetch wrapper with base URL + optional bearer token).
- [ ] A6. CI skeleton: GitHub Actions workflow `ci.yml` matrix `windows-latest / ubuntu-latest / macos-14`. Jobs: lint (ruff+eslint), typecheck (mypy+svelte-check), unit tests (pytest+vitest). Cache pip + npm.
- [ ] A7. Pre-commit: ruff format+check, mypy, eslint, prettier, end-of-file-fixer.
- [ ] A8. `AGENTS.md` with build/test/lint commands so future agents know them.

**Gate A:** `make lint typecheck test` passes on all 3 OSes; health endpoint returns 200 from the SPA dev server.

---

## Phase B — Storage layer (replaces S3)

- [ ] B1. Define `Storage` Protocol (`base.py`): `get_bytes(key) -> bytes`, `put_bytes(key, bytes)`, `get_text`, `put_text`, `read_csv(key) -> pd.DataFrame`, `write_csv`, `list(prefix) -> list[key]`, `exists`, `delete(prefix)`, `copy(src,dst)`, `put_object_png` (viz). Keys mirror web S3 layout: `<report_id>/file`, `<report_id>/original_file`, `<report_id>/testset`, `<report_id>/retest`, `<report_id>/<model>_model.joblib`, `<report_id>/scaler.joblib`, `<report_id>/report.csv`, `<report_id>/results.json`, `<report_id>/output_log`, `<report_id>/viz/<CHART>_<model>`, `<report_id>/shap_rows_<model>`.
- [ ] B2. `LocalStorage`: dir-per-report under `<appdata>/datasets/<report_id>/...`. Implement all Protocol methods against the filesystem. Atomic writes (temp file + os.replace).
- [ ] B3. `resolve_appdata()`: `%APPDATA%/CLASSify` (win), `~/.local/share/CLASSify` (linux), `~/Library/Application Support/CLASSify` (mac). Honor `settings.data_dir` override + portable mode (data next to binary).
- [ ] B4. Optional encryption wrapper (`encrypted.py`): when enabled, wrap `LocalStorage` with AES-GCM streaming (cryptography lib) for file bodies + use SQLCipher for DB. Passphrase held in OS keyring (keyring lib) when encryption on. Off by default.
- [ ] B5. Disk-space guard: refuse writes when free space < threshold; expose `GET /api/system/usage` (used/free bytes).
- [ ] B6. Unit tests: round-trip every Storage method on LocalStorage; encryption on/off parity.

**Gate B:** Storage unit tests green; a dataset can be written, listed, read back, copied, deleted identically with encryption on and off.

---

## Phase C — Database & migrations

- [ ] C1. SQLAlchemy 2.0 `db.py`: engine factory `sqlite:///<appdata>/classify.db` (or SQLCipher URL when encryption on), `sessionmaker`, `get_session` FastAPI dependency.
- [ ] C2. ORM models (`orm/`): `Report` (uuid, filename, original_filename, status, task→job_id, column_changes(JSON), comments, created_at), `Result` (uuid, report_uuid, date_processed), `Job` (id, report_uuid, state queued/running/cancelling/succeeded/failed, args(JSON), progress, error, created/started/finished), `Setting` (k/v), `Action` (local history, optional).
- [ ] C3. Alembic init + first migration creating all tables. `alembic upgrade head` runs on first launch.
- [ ] C4. Repository functions: `reports.create/with_uuid/list_for_table/update_status/update_comments/delete`, `results.create/with_report`, `jobs.enqueue/peek/running/set_state/update_progress`.
- [ ] C5. Seed defaults into `Setting` on first run (n_jobs=os.cpu_count(), theme, encryption=off, max_upload_mb=500).
- [ ] C6. Tests: CRUD on each repo; migration up/down idempotent.

**Gate C:** App boots on a clean machine, creates DB, runs migrations, settings seeded.

---

## Phase D — ML engine fork & refactor

Goal: take CLASSify-2's `models.py`, `synthesize.py`, `visualization.py`, `retest_file.py` verbatim as a starting copy, then refactor I/O and orchestration — **do not** change the modeling math/defaults.

- [ ] D1. Copy the four files into `backend/ml/`; strip `clearml`, `boto3`/`s3transfer` imports; remove `Task`/`Dataset`/`Logger`/`enqueue` calls; remove SLURM/`epilog` paths.
- [ ] D2. Define typed `TrainingArgs` dataclass (`args.py`) replacing the `dotdict` from `utils.py`. Include every parameter from `api.py` `/get-ml-options` (supervised + unsupervised sets).
- [ ] D3. Refactor `trainer(args, storage, dataset_df, testset_df, on_progress, cancel_token) -> results_dict`. Replace every `S3_CONNECTION['client'].put_object/get_object/...` with `storage.*`. Replace ClearML logging with `on_progress(completed, total)` callback + `log_to_file` to `output_log`.
- [ ] D4. `backends.py` addon registry: `is_available(name)` via guarded import; `require(name)` raises a clean `AddonMissingError` with install instructions. Wrap TabPFN (`from tabpfn import TabPFNClassifier`) and SDV (`from sdv...`) usages so the engine imports and runs without them installed; `train_group` containing `tabpfn` is rejected upfront with a friendly message if the addon is absent. Synthesis options (`synthesize_original`/`synthesize_new`) similarly gated on the SDV addon.
- [ ] D5. `column_types.py`: port `get_column_types_internal`, `createMappingColumn`, `change_column_types` logic, `upload_testset` transform logic — all using `storage` instead of S3, returning pydantic schemas.
- [ ] D6. `options.py`: produce the supervised + unsupervised ml-options dicts as static typed structures (port from `api.py` `/get-ml-options`). One source used by both the `/api/ml-options` endpoint and frontend.
- [ ] D7. `retest.py`: port `retest_file.retest()` to use `storage` + `joblib.load` from local files; gated model loading.
- [ ] D8. Keep `matplotlib.use('Agg')`; ensure all chart functions write PNGs via `storage.put_bytes(..., ContentType png)`.
- [ ] D9. Tests: golden dataset (small CSV) → train RandomForest only → assert `report.csv` + `results.json` + `viz/ROC_*` + joblib produced on disk and metrics within sane ranges. Repeat with a multiclass + a clustering (kmeans) dataset.

**Gate D:** End-to-end train of RandomForest/LogReg/KMeans on a sample dataset via a CLI harness (`python -m ml.cli train ...`) produces the same artifacts the web version does, with no torch installed.

---

## Phase E — Dataset processing endpoints (FastAPI)

- [ ] E1. `POST /api/datasets/upload` (multipart): save uploaded CSV → `storage` as `<report_id>/file` + `original_file`; run `column_types.get_column_types_internal`; create `Report` row (status `Preview`); return `{report_id, column_types, missing_values}`. Reuse web validation (csv extension, filename ≤100 chars, ≤500MB, spaces→underscores).
- [ ] E2. `POST /api/datasets/{id}/column-changes`: port `change_column_types` — apply drops, type casts, missing-value strategies (drop/constant/synthetic IterativeImputer), one-hot encode low-cardinality categoricals (reject high-cardinality), class-column validation (≥2 classes, not float). Persist transformed `file`. Store `column_changes` JSON on Report.
- [ ] E3. `GET /api/datasets/{id}/class-values?class_column=` → unique class labels (for the mapping modal).
- [ ] E4. `POST /api/datasets/{id}/class-mapping`: port `set_class_column_mapping` — add `<class>_mapping` column via `createMappingColumn`.
- [ ] E5. `POST /api/datasets/{id}/testset` (multipart): port `upload_testset` — apply same transforms, validate columns/categories, write `<report_id>/testset`.
- [ ] E6. `POST /api/datasets/{id}/duplicate`: port `copy_dataset` (suffix logic, original-file retention).
- [ ] E7. `DELETE /api/datasets/{id}`: port `delete_dataset` (keep original_file if other copies exist).
- [ ] E8. `GET /api/datasets` (DataTable-style query: start/length/search/order) → list for the results table.
- [ ] E9. `PATCH /api/datasets/{id}/comment`.
- [ ] E10. Request/response pydantic schemas for all of the above; 422 on bad input; consistent `{success, message}` error envelope.
- [ ] E11. Tests: each endpoint happy path + key error cases (missing class column, high-cardinality, mismatched testset columns, unexpected categories).

**Gate E:** A dataset can be uploaded, columns configured, class mapped, testset added, duplicated, and deleted via the API alone.

---

## Phase F — Job runner (replaces ClearML)

- [ ] F1. `queue.py`: SQLite-backed FIFO. `enqueue(report_id, args) -> job_id`; `next_pending()`; states `queued/running/cancelling/succeeded/failed`. Only one `running` at a time (advisory lock).
- [ ] F2. `manager.py`: background thread (started at app boot) that pulls the next queued job and spawns `jobworker` as a **subprocess** (frozen entry point `classify-jobworker` under PyInstaller; `python -m runner.jobworker` in dev). Monitor via poll + progress file. On success/fail, set Report status + Result row. On child crash, mark failed, write traceback to `output_log`, still emit partial `report.csv`/`results.json` if the engine produced any (replaces `epilog.py`).
- [ ] F3. `jobworker.py`: subprocess entry. Load `TrainingArgs` from job row, resolve `storage`, call `engine.trainer(args, storage, df, testset_df, on_progress=write_progress, cancel_token=read_cancel_flag)`. Write `report.csv`, `results.json`, joblibs, viz, shap_rows, `output_log`. Exit code 0/1.
- [ ] F4. `progress.py`: worker writes `<report_id>/progress.json` (`{completed,total,message,updated_at}`) + appends to `output_log` live. Manager reads; `GET /api/jobs/{id}/events` streams via **SSE** (FastAPI `StreamingResponse`). Reuse web "n/total Processed" status string.
- [ ] F5. `cancellation.py`: `POST /api/jobs/{id}/cancel` sets cancel flag (file + DB state `cancelling`); manager sends SIGTERM to the process **group** (use `os.setsid`/creationflags CREATE_NEW_PROCESS_GROUP on win) so Optuna/numpy children die; SIGKILL after grace timeout. Job → `failed` with status `Cancelled`.
- [ ] F6. `POST /api/jobs` (start training): validate options against `options.py` + addon availability; build `TrainingArgs`; set Report status `Processing`; enqueue. Return job_id. Port the web `train()` option-parsing (bool/float/int/list, `train_group` flattening, defaults for `n_jobs`/`max_features`/`min_samples_*`/`bootstrap`).
- [ ] F7. `GET /api/jobs/{id}` status; `GET /api/jobs` list.
- [ ] F8. Crash recovery on app start: any `running` job whose subprocess is gone → mark `failed` (interrupted). Re-queue nothing automatically; let user retry.
- [ ] F9. Tests: enqueue 2 jobs → run sequentially; cancel a running job → process group dies within timeout; crash a worker → status failed + partial artifacts present.

**Gate F:** Submit a training job via API, watch live SSE progress, cancel it cleanly, and survive a worker crash — all without the UI.

---

## Phase G — Results & re-test endpoints

- [ ] G1. `GET /api/results/{report_id}`: return `results.json` (metrics) + `report.csv` rows (paginated) for the Results Table. Include metric tooltip metadata (port the web tooltip dict) as a static `/api/system/metric-defs` so the frontend can render tooltips.
- [ ] G2. `GET /api/results/{report_id}/viz`: list `viz/` objects (names + URLs). Serve each via `GET /api/results/{report_id}/viz/{name}` streaming the PNG (replaces presigned URLs).
- [ ] G3. `GET /api/results/{report_id}/shap-rows/{model}`: serve the per-row SHAP CSV (for Prediction Insights table).
- [ ] G4. `GET /api/results/{report_id}/shap-row-graph?model=&row_num=&train_test=`: port `download-shap-row-graph` matplotlib builder → PNG (reuse `api.py` logic verbatim, swap S3 reads for `storage`).
- [ ] G5. `POST /api/results/{report_id}/retest` (multipart testset): port `retest_model` flow — load joblibs, run `retest.retest()`, return retest metrics.
- [ ] G6. `GET /api/results/{report_id}/output-log`: stream the log text.
- [ ] G7. `GET /api/results/{report_id}/download?suffix=`: stream any artifact (dataset, report.csv, results.json, joblib, scaler, log) for Download Data.
- [ ] G8. `GET /api/datasets/{id}/prepare-params`: return previous job's `TrainingArgs` (for rerun-with-same-params) + detected class_column — read from local Job row, not ClearML.
- [ ] G9. Tests: each results endpoint against a fixture trained report.

**Gate G:** All results tabs' data is fetchable via API on a trained report.

---

## Phase H — Frontend SPA (Svelte)

Design principle: **visual parity with the web app** — port the CSS variables (`variables.css`), navbar, switches, modals, and layout; rebuild the JS interaction in idiomatic Svelte. Charts stay server-rendered PNGs (`<img>`), with optional Plotly upgrade later.

- [ ] H1. SvelteKit-style routing (Svelte + routify or svelte-spa-router; no SSR — pure SPA). Routes: `/` (datasets/home), `/datasets/:id/prepare`, `/results`, `/results/:id`, `/settings`, `/addons`.
- [ ] H2. Port the web design system: `app.css` with CSS variables (navbar color, button palette, title text), Bootstrap 5 CSS import, `navbar.css`/`switches.css`/`global.css` equivalents. No jQuery.
- [ ] H3. **Home/Datasets page**: "Add Data File" + "View All Data" cards; upload modal → file pick (native dialog) → upload to E1; results table via **TanStack Table** (replaces DataTables) with search/sort/pagination matching web columns; per-row actions (prepare, duplicate, delete, comment).
- [ ] H4. **Column Preview modal**: rebuild `showColumns` — per-column include checkbox, type radios (integer/float/bool/categorical), missing-value strategy dropdown (drop/constant/synthetic) + fill-value input, class-column select (svelte-multiselect single). Same enable/disable logic as web.
- [ ] H5. **Class Mapping modal**: rebuild the drag-drop string→integer mapper (HTML5 drag, Svelte store for order). Same UX as web `mappingModal`.
- [ ] H6. **Prepare page**: rebuild `parseMLOpts` renderer — supervised/unsupervised toggle, auto-determine-clusters toggle, dynamic option sections (General/Model Parameters), `.toggle-parameter` enable-by-selected-models logic (`change_blocked`), tooltips, reset-to-defaults, separate-testset upload, rerun-with-previous-params. Pull options from `/api/ml-options`.
- [ ] H7. **Results detail page**: tabbed (Results Table / Visualizations / Download Data / Re-Test / Prediction Insights / Output Log).
  - Results Table: TanStack Table over `report.csv` with metric-def tooltips.
  - Visualizations: gallery of PNGs from G2 (ROC, confusion, SHAP beeswarm per class, cluster plots, heatmaps, train/test/CV comparison).
  - Download Data: buttons per artifact (G7).
  - Re-Test: testset upload → G5 → show retest metrics.
  - Prediction Insights: per-row SHAP table (G3) + row selector → SHAP impact bar chart PNG (G4).
  - Output Log: live-updating log viewer (SSE or poll G6).
- [ ] H8. **Job status**: global jobs store; SSE subscription; toasts on completion/failure; status badges on the datasets table; cancel button.
- [ ] H9. **Settings page**: data dir, theme colors, n_jobs, max upload size, encryption toggle (with passphrase set flow), addon management, "Check for updates".
- [ ] H10. **Add-ons page**: list installed/available addons (TabPFN, SDV); install button → triggers addon installer (Phase J); show download size + disk space; offline handling.
- [ ] H11. Typed API client (`lib/api/`): generated types from OpenAPI (A5); one `apiClient` with methods per endpoint; bearer token from the shell's launch token.
- [ ] H12. Accessibility: preserve web ARIA roles, tooltips, focus handling, skip-link, keyboard nav for the mapper.
- [ ] H13. Frontend tests: vitest + @testing-library/svelte for Column Preview, Class Mapper, MLOptionsForm rendering/interactions; snapshot the design system.

**Gate H:** A user can upload a CSV, configure columns, map a class, set options, train, and browse all results tabs — entirely in the SPA, matching the web look.

---

## Phase I — pywebview shell, tray, lifecycle

- [ ] I1. `server.py`: pick free port on `127.0.0.1`, generate a random bearer token, start Uvicorn in a background thread, wait for health endpoint, return `http://127.0.0.1:<port>/?token=...`. Bind only loopback.
- [ ] I2. `shell.py`: open pywebview window (min size 1200×800), load the token URL. Window title + app icon per OS. DevTools available in dev builds only.
- [ ] I3. **Tray**: system tray icon with menu — Show window, Running jobs (count), Check for updates, Settings, Quit. Minimize-to-tray on window close.
- [ ] I4. **Close-with-jobs prompt**: on window close, if a job is `running`/`queued`, show a native dialog (pywebview `window.create_confirmation_dialog` or custom): "A training job is running. Keep running in the background?" → [Keep running (tray)] [Stop jobs & quit]. If no jobs → minimize to tray.
- [ ] I5. **Quit flow**: tray Quit → if jobs active, confirm "Stop running jobs and quit?" → cancel jobs (F5) → wait grace period → stop server → exit. If idle → stop server → exit.
- [ ] I6. Single-instance enforcement: lock file (linux/mac) / named mutex (win); second launch focuses the existing window.
- [ ] I7. Clean shutdown: stop runner manager, finish/cancel in-flight subprocess handling, close DB, flush logs.
- [ ] I8. Tests: manual + a smoke test harness that boots the shell headless (xvfb on linux) and asserts the window loads.

**Gate I:** App launches, runs a job, closes to tray with the prompt, quits cleanly from tray.

---

## Phase J — Add-on system (TabPFN + SDV)

- [ ] J1. Addon manifest format (`addons/<name>/manifest.json`): name, version, pip deps, size estimate, min app version, entry module(s) it provides (`tabpfn`, `sdv`).
- [ ] J2. Addon installer service: `POST /api/addons/{name}/install` → download wheels via `pip install --target=<appdata>/addons/pythonlibs --no-deps <deps>` (resolve deps with deps), show progress, verify import succeeds, register in `Setting`/`installed_addons` table. `POST /api/addons/{name}/uninstall` → remove dir.
- [ ] J3. Runtime: at boot, prepend `<appdata>/addons/pythonlibs` to `sys.path`; `backends.is_available()` then succeeds for installed addons. Re-check on install without restart where possible (importlib).
- [ ] J4. TabPFN addon: torch (CPU wheel) + tabpfn + huggingface-hub. First TabPFN use downloads model weights into `<appdata>/cache/huggingface` — surface progress + disk check + offline error. Engine: `tabpfn` allowed in `train_group` only when addon installed.
- [ ] J5. SDV addon: torch + sdv + ctgan + copulagan + tvae + rdt + deepecho + copulas. Engine: synthesis options gated.
- [ ] J6. GPU-addon stub (post-v1): same mechanism with CUDA torch wheel + CUDA runtime check; document only for v1.
- [ ] J7. Tests: install/uninstall cycle on each OS; train TabPFN after install; train SDV synthesis after install; engine rejects TabPFN/SDV before install with friendly message.

**Gate J:** With only the base installer, TabPFN/SDV are unavailable with a clear message; after installing the addon, they work.

---

## Phase K — Packaging & distribution

- [ ] K1. PyInstaller `--onedir` spec per OS. Hooks for numba, torch-free deps, shap, xgboost native libs, matplotlib fonts, pyamg. Hidden imports audited via runtime smoke test.
- [ ] K2. **Exclude** boto3/s3transfer/clearml/clearml-agent/flask. (Base bundle must NOT contain torch.)
- [ ] K3. Bundle the built SPA (`vite build`) as static assets served by FastAPI; confirm it loads from the frozen app with no dev server.
- [ ] K4. Multiple entry points in the bundle: `CLASSify` (main app) + `classify-jobworker` (job subprocess) + `classify-api` (optional headless).
- [ ] K5. macOS Universal2: build arm64 + x86_64, `lipo`-merge binaries, create `.app` bundle, sign with Developer ID Application, **notarize** (`notarytool`) + staple. Verify `spctl --assess`.
- [ ] K6. Windows: Authenticode-sign `CLASSify.exe` + installer (OV cert; EV if budget). Build `.exe` installer (Inno Setup or NSIS) with file associations (.csv) + start menu shortcuts + uninstaller.
- [ ] K7. Linux: AppImage (universal glibc) + `.deb` + `.tar.gz`. Optional GPG sign.
- [ ] K8. First-run wizard (see Phase L) baked into installers; installers place data dir default under appdata.
- [ ] K9. CI release workflow: on tag, build all OSes in parallel, sign/notarize, upload artifacts to the GitHub release, generate SHA256 + a `latest.json` manifest (for the future auto-updater + the manual "Check for updates").
- [ ] K10. Smoke test each artifact on a clean VM (Win10/11, Ubuntu 22.04, macOS 13 intel + macOS 14 arm): launch, upload sample CSV, train RandomForest, view results, install TabPFN addon, train TabPFN.

**Gate K:** Signed/notarized installers for all 3 OSes that pass the clean-VM smoke test; base installer has no torch.

---

## Phase CI — CI/CD infrastructure

Full detail in **`CI_CD.md`**. This phase stands up the pipelines early so every later phase runs under them from day one.

- [ ] CI1. `ci.yml`: PR/push-to-main workflow, matrix `windows-latest / ubuntu-latest / macos-14`. Jobs: backend-quality (ruff format+check, mypy strict, pip-audit), frontend-quality (eslint, prettier, svelte-check, npm audit), openapi-sync (regen types + `git diff --exit-code`), backend-tests (×3 OS), frontend-tests, ml-regression (torch addon venv), build-smoke (×3 OS frozen boot).
- [ ] CI2. Branch protection on `main`: all `ci.yml` jobs required; conventional-commit PR title check; code-owner review on `backend/ml/` + `desktop/packaging/`.
- [ ] CI3. Concurrency control (cancel superseded runs); pip/npm/PyInstaller/HF-weight caching with hashed keys (CI_CD.md §5).
- [ ] CI4. `make ci` + `scripts/ci-local.ps1` parity target so devs reproduce CI locally; document `act` for workflow testing.
- [ ] CI5. `nightly.yml`: scheduled full signed+notarized packaging build + clean-runner smoke per OS/arch; posts a summary table.
- [ ] CI6. `release.yml` (on tag `v*`): per-OS build+sign+notarize+staple+smoke, then `release-publish` (GitHub release + `sha256` + `latest.json` + SBOM + build provenance attestation), then `post-release-verify` (re-download + re-smoke + sha check).
- [ ] CI7. Secrets in GitHub Actions (CI_CD.md §4): Windows PFX, Apple p12 + notarization creds (API-key auth), GPG key, Codecov token. `release` environment with required reviewers for `release.yml`.
- [ ] CI8. Quality gates wired as required checks (CI_CD.md §8); coverage gates `--cov-fail-under=80` (backend), vitest lines ≥75% (frontend).
- [ ] CI9. Dependency hygiene: pinned `requirements/*.txt` (pip-compile), `package-lock.json`, CycloneDX SBOM generation per release, artifact attestations (SLSA L3).
- [ ] CI10. `latest.json` manifest generator + stable raw URL for the in-app update check.

**Gate CI:** A PR cannot merge unless all gates pass on all 3 OSes; a tag produces signed installers + manifest end-to-end with no manual steps.

---

## Phase L — First-run wizard & UX polish

- [ ] L1. First-run wizard: welcome → choose data dir (default + portable option) → optional encryption setup (skip by default) → CPU/disk check → optional addon offer (TabPFN/SDV, "install now or later") → done.
- [ ] L2. Training-time estimates: based on n_rows × n_models × tuning, show a rough ETA + a "this may take a while" warning for expensive combos (tabpfn large N, neuralnetwork high n_iter, spectral clustering big data).
- [ ] L3. Memory/size guards: estimate one-hot expansion + IterativeImputer memory; warn before training if dataset is large.
- [ ] L4. Error surfaces: engine exceptions → user-friendly messages (port the web's feature-count-mismatch messaging for retest, high-cardinality, unexpected categories, etc.).
- [ ] L5. In-app user guide page (replaces the `user_guide` plugin) — markdown rendered locally.
- [ ] L6. Theme: port tenant styling defaults (title text, navbar color, button palette) as user-editable settings.
- [ ] L7. "Check for updates": fetch `latest.json`, compare versions, open release page in the system browser.
- [ ] L8. Logging UI: a diagnostics view (last errors, log file path) to help support.

**Gate L:** A brand-new user can install, run the wizard, and train a model without reading docs.

---

## Phase M — Testing & quality

Full detail in **`CI_CD.md`** (test pyramid §7, coverage gates §8, supply-chain §10). This phase implements the test suites the CI pipeline runs.

- [ ] M1. Backend unit tests ≥80% on `ml/` (excluding torch-gated paths via `# pragma: no cover`), `storage/`, `runner/`, `routers/`, `services/`. Fixtures: `tmp_appdata`, `isolated_storage`, `fresh_db` (alembic upgrade head on tmp SQLite), `sample_csv`. Markers: `ml_regression`, `addon`, `slow`, `e2e` (default run excludes the latter three).
- [ ] M2. ML regression suite: golden datasets (binary, multiclass, clustering, missing-values, categorical) under `tests/fixtures/datasets/`; per case assert `report.csv`/`results.json`/`viz` PNGs/joblib/`shap_rows_*` presence + metric keys (snapshot) + loose metric ranges for RF/LogReg/XGBoost/KMeans. TabPFN/SDV cases under `@pytest.mark.addon`. CPU only.
- [ ] M3. Frontend unit tests (vitest + @testing-library/svelte + MSW for the typed API client) for Column Preview, Class Mapper, MLOptionsForm, ResultsTable, JobStatus; snapshot the design system; axe-svelte a11y checks on mapper + options form. Coverage lines ≥75%.
- [ ] M4. Playwright e2e: full flow upload→configure→train→results on a tiny dataset on the frozen dev build (xvfb on linux); `multiclass-flow` + `clustering-flow` variants.
- [ ] M5. Clean-VM smoke: on each signed artifact (release/nightly), download → install → `--smoke` first-run → upload `tiny.csv` → train RF → assert results → install TabPFN addon → train TabPFN → attach logs on failure.
- [ ] M6. mypy strict on `backend/`; `svelte-check` clean; ruff/eslint clean — all enforced as required CI gates.
- [ ] M7. Security: pip-audit + npm audit in CI (fail on new advisories), no secrets in bundle, local-only binding verified by a port-scan test, CycloneDX SBOM + SLSA build-provenance attestations per release.

**Gate M:** All CI quality gates green on all 3 OSes; coverage thresholds met; e2e + smoke stable on `main`.

---

## Phase N — v1 release

- [ ] N1. Freeze `requirements.txt` + `package.json` versions; reproducible build.
- [ ] N2. Release notes + in-app guide + README install instructions per OS.
- [ ] N3. Signing/notarization final check on shipped artifacts.
- [ ] N4. Publish `latest.json` manifest + SHA256s to the release.
- [ ] N5. Post-v1 backlog queued in **`ROADMAP.md`** (Tier 1 first): real auto-update (R1), GPU add-on (R2), interactive Plotly charts (R3).

**Gate N:** v1 installers published and smoke-tested on all 3 OSes.

---

## Feature-parity matrix (web → desktop)

| Web capability | Desktop source | Status target |
|---|---|---|
| CSV upload + column type auto-detect | E1 + `column_types.py` | parity |
| Column include/exclude + type change | E2 | parity |
| Missing-value strategies (drop/constant/synthetic) | E2 | parity |
| Categorical one-hot (low-card) + high-card reject | E2 | parity |
| Class-column select + drag-drop int mapping | E3/E4 + H5 | parity |
| Separate testset upload + transform | E5 | parity |
| Dataset duplicate / delete (original retention) | E6/E7 | parity |
| Supervised/unsupervised + auto-clusters + full ML options | F6 + `options.py` + H6 | parity |
| 12 supervised + kmeans/spectral/hdbscan | `engine.py` | parity |
| TabPFN + SDV synthesis | Phase J addons | add-on gated |
| Optuna tuning + CV with CIs | `tuning.py`/`evaluate.py` | parity |
| Training status + progress | F4 SSE | parity (+cancel) |
| Results table w/ metric tooltips | G1 + H7 | parity |
| Visualizations (ROC/confusion/SHAP/cluster/heatmap/compare) | G2 + `visualize.py` | parity |
| Download Data | G7 | parity |
| Re-test models on new testset | G5 | parity |
| Prediction Insights (per-row SHAP + impact chart) | G3/G4 + H7 | parity |
| Output log | G6 + H7 | parity |
| Rerun with previous params | G8 | parity |
| Comments | E9 | parity |
| SSO / multi-tenant / plugins / admin report | — | **dropped** (single-user) |
| S3 / ClearML / Postgres / PHP / Docker | — | **replaced** (local FS / local runner / SQLite / FastAPI-Svelte / native installers) |

---

## Notes on best-practice choices made (no need to revisit)

- **FastAPI + Pydantic v2**: typed contract; OpenAPI → TS types keeps frontend/backend in sync.
- **SQLAlchemy 2.0 sync + Alembic**: migrations as code; SQLCipher swap for encryption.
- **Subprocess-per-job**: crash isolation + guaranteed-clean cancellation (process-group kill) — strictly better than the web version's non-cancellable ClearML jobs.
- **SSE for progress**: simple, FastAPI-native, no websocket complexity for a local single-user app.
- **Addon = `pip install --target` dir + sys.path**: works inside a PyInstaller bundle; keeps the base installer torch-free.
- **System webview (pywebview)**: no bundled Chromium; native menus/tray/dialogs.
- **structlog + mypy strict + ruff**: long-term maintainability.
