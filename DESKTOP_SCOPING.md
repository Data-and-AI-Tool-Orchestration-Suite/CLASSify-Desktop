# CLASSify Desktop — Production Scoping

Scope for shipping a cross-platform (Windows / Linux / macOS) **desktop** edition of CLASSify-2 that preserves the existing web UX, runs entirely on local compute + local file storage, and removes the S3 / ClearML / Postgres / PHP-Apache / SSO dependencies.

---

## 1. Executive summary

CLASSify-2 today is a 3-tier web app:

| Layer | Today (web) | Desktop target |
|---|---|---|
| UI | PHP (LAPP) on Apache, Bootstrap 5 + jQuery + DataTables | Same HTML/CSS/JS, served by a local Python web server inside a desktop webview |
| ML API | Flask `api.py` | Local FastAPI/Flask service, in-process |
| Job orchestration | ClearML queue + `agent_local.py` / `agent_dgx.py` / SLURM | In-process local job queue + worker thread(s) |
| Storage | S3 (boto3) for datasets/models/viz/logs | Local filesystem under app data dir |
| Metadata DB | PostgreSQL (users, tenants, reports, results, actions, plugins) | SQLite (reports, results, jobs, settings only) |
| Auth | CiLogon SSO + multi-tenant RBAC | Single-user, no auth (optional local password/PIN) |
| Compute | DGX GPU cluster via ClearML | User's local CPU (GPU optional) |

**Recommendation:** Ship a **Python-native desktop app** = `pywebview` shell + a local Flask/FastAPI server + SQLite + local FS, packaged with **PyInstaller**, with the PHP views ported 1:1 to **Jinja2 templates** so the Bootstrap UI is preserved verbatim. This maximizes reuse of the existing Python ML engine (`models.py`, `retest_file.py`, `synthesize.py`, `visualization.py`) and the existing HTML/JS, and avoids dragging in a second runtime (Node/Electron) or a new language (Rust/Tauri). The ML dependency footprint (torch, tabpfn, sdv, xgboost, shap) dominates installer size anyway, so Electron's overhead buys nothing here.

Alternatives considered in §3.

---

## 2. Current system (what exists, mapped to desktop relevance)

### 2.1 Frontend (`frontend/`, PHP)
- `routes.php` — AltoRouter routes; ~all logic is AJAX → JSON.
- Views that matter for desktop:
  - `home.php` — upload CSV → column preview modal → class-column selection → drag-drop **string→integer mapping modal** → submit.
  - `reports/prepare.php` — ML options form: supervised/unsupervised toggle, auto-determine-clusters, `train_group` (12 models), parameter-tuning ranges, SHAP, synthesis, sampling, testset upload.
  - `results/details.php` — tabbed results: **Results Table, Visualizations, Download Data, Re-Test Models, Prediction Insights (SHAP per-row), Output Log**.
  - `results/index.php` — list of datasets (DataTables).
- `_header.php` / `_menu.php` / `_modals.php` — navbar, tenant switcher, site banner.
- Heavy use of: Bootstrap 5.3, jQuery 3.7, DataTables (with buttons/colvis/searchpanes), bootstrap-select, Font Awesome, Toastify, daterangepicker, pickr.

### 2.2 ML trainer (`backend/ml_classifier_trainer/`, Python)
- `api.py` — Flask API. Endpoints relevant to desktop: `get_column_types`, `change_column_types`, `set_class_column_mapping`, `get_class_column_values`, `upload_testset`, `get-ml-options` (+ `-uns`), `train`, `update_training`, `finished_training`, `download-shap-row-graph`, `retest_model`, `delete_dataset`, `copy_dataset`, `get_parameters`.
- `run_job.py` — ClearML task entry: reads args, pulls dataset from S3, calls `trainer()`, uploads results to S3, calls `/finished_training`.
- `agent_local.py` / `agent_dgx.py` / `agent_cronjob.py` / `check_queue.sh` — ClearML queue pollers. **Drop entirely.**
- `models.py` — the core: `trainer()`, `estimatorevaluation()`, `objective()` (Optuna), `getmodelstats()`, `clusteringstats()`, SHAP beeswarm + per-row CSV, `write_report()`, model `.joblib` save, ROC/confusion/cluster/heatmap viz.
- `retest_file.py` — re-test saved models on a new testset.
- `synthesize.py` — SDV synthetic data (tabular/ctgan/copulagan/tvae).
- `visualization.py` — matplotlib chart generators.
- `epilog.py` — post-job crash recovery (SLURM error capture + partial-report write). Replace with local try/except + partial-result save.
- `create_report.py` — admin usage PDF/Excel (pulls users/actions from the web API). **Single-user desktop → drop or repurpose as a per-job results report.**
- 12 models: randomforest, neuralnetwork, xgboost, gradientboosting, histgradientboosting, bagging, sgdclassifier, logisticregression, kneighbors, tabpfn, spectralclustering, kmeans (+ hdbscan unsupervised).

### 2.3 Storage & DB
- S3 bucket `classify` stores (per `user_uuid/filename/...`): `file`, `original_file`, `testset`, `retest`, `*_model.joblib`, `scaler.joblib`, `report.csv`, `results.json`, `output_log`, `viz/ROC_*`, `viz/SHAP_*`, `viz/...`, `shap_rows_<model>`.
- Postgres tables: `users`, `user_sessions`, `tenants*`, `plugins`, `resources*`, **`Report`**, **`Results`**, `user_actions`. Only `Report` + `Results` (+ a `jobs` concept) are needed for desktop.

### 2.4 Auth / tenancy / plugins
- CiLogon OAuth2, multi-tenant RBAC, Row-Level Security, plugins (api_keys, projects, S3, LLM, user_guide, user_agreement, site_banner). **All dropped** for a single-user desktop app.

---

## 3. Architecture decision: shell options

| Option | Shell | Frontend | Pros | Cons |
|---|---|---|---|---|
| **A. pywebview + Flask (recommended)** | pywebview (OS WebView2/WebKit) | Jinja2 port of PHP views | Single language (Python); maximal ML-code reuse; tiny shell; reuses Flask `api.py` | Still bundles full Python runtime |
| B. Tauri 2 + Python sidecar | Rust shell | JS framework or static HTML | Smallest shell; modern | Adds Rust + IPC + sidecar management; two toolchains |
| C. Electron + bundled Python | Electron | Reuse HTML/JS directly | Max browser compat; mature | ~150 MB Chromium overhead for nothing; two runtimes |
| D. Flask + system browser (no shell) | none | local web page | Simplest | Not a "real" app; no tray/installer integration; poor UX |

**Go with A.** The frontend is already jQuery+Bootstrap talking to a JSON API; porting the ~6 PHP views to Jinja2 is mechanical, and the entire ML engine is reused as-is with only its I/O layer swapped (S3 → local FS, ClearML → local queue).

### 3.1 Target runtime topology

```
CLASSify.exe (PyInstaller bundle)
 ├── pywebview window  ──> http://127.0.0.1:<port>/   (OS webview)
 ├── Flask/FastAPI server (localhost)        ← reused api.py logic
 │     ├── /  Jinja2 views (ported from PHP)
 │     └── /api/*  JSON endpoints            ← ported controllers
 ├── Local job runner (replaces ClearML)
 │     ├── SQLite-backed queue
 │     └── worker thread → models.trainer()  ← reused, I/O patched
 ├── Storage adapter (replaces boto3/S3)     ← local FS under appdata
 └── SQLite (replaces Postgres)              ← Report/Results/Jobs/Settings
```

---

## 4. Component migration map

| Web component | Desktop replacement | Effort |
|---|---|---|
| `api.py` Flask endpoints | Port to local Flask app; remove `@apiKey_required` + S3, point at `Storage` adapter | Medium |
| `models.py` `trainer()` | Reuse directly; replace `S3_CONNECTION` calls with `Storage` calls; remove `clearml` Logger/Task calls | Medium |
| `run_job.py` | Delete; worker calls `trainer()` directly with a dict args object | Small |
| `agent_*.py`, `check_queue.sh`, `epilog.py` | Delete; replace with `LocalJobRunner` (queue + thread pool + status callbacks) | Medium (new) |
| `retest_file.py` | Reuse; S3→local FS | Small |
| `synthesize.py`, `visualization.py` | Reuse as-is (already pure functions writing to buffers) | Trivial |
| `create_report.py` | Drop (admin usage report) — or repurpose as per-job PDF | Small |
| `ReportsController` / `ResultsController` | Port PHP→Python route handlers (logic is thin; mostly forwards to `api.py`, which is now in-process) | Medium |
| `UsersController/TenantsController/PluginsController/...` | **Delete** | — |
| PHP views (`home`, `prepare`, `results/*`) | Port to Jinja2 templates (same HTML/JS; `<?= $rootURL ?>` → `{{ url_for(...) }}` or root-relative) | Medium |
| `_header/_menu/_modals` | Port; drop tenant switcher/banner; keep navbar + flash messages | Small |
| Postgres `Report`/`Results` tables | SQLite tables (same columns, drop `tenant_id`) | Small |
| Postgres users/tenants/plugins/actions | Delete (optional: keep `jobs` + `settings` + `user_actions` for local history) | — |
| S3 (`S3Utility`, boto3) | `LocalStorage` class: `get/put/list/delete/copy` against `<appdata>/datasets/<report_id>/...` | Medium (new) |
| `config.ini` / `config.php` | Single `settings.json` (data dir, n_jobs, theme, optional model-weight cache) | Small |
| CiLogon SSO / sessions | Delete; optional local PIN/password lock | Small |

---

## 5. What is removed

- **S3** (boto3, s3transfer, presigned URLs, `S3Utility.php`) → local filesystem.
- **ClearML** (`clearml`, `clearml-agent`, queues, `Task.create/enqueue/get_task/mark_*`) → local in-process runner.
- **PostgreSQL** (+ RLS, tenants, triggers, `init.sql`) → SQLite.
- **PHP + Apache + Composer + Guzzle** → Python (Flask + Jinja2).
- **CiLogon OAuth2 / SAML / multi-tenant RBAC / sessions** → single user.
- **Plugins** (api_keys, projects, S3, LLM, user_guide, user_agreement, site_banner) → not applicable.
- **DGX/SLURM agent** (`agent_dgx.py`, `epilog.py` SLURM error capture) → local exceptions.
- **Admin usage report** (`create_report.py`) → drop or repurpose.
- **Docker compose** → native installers.

---

## 6. What is added (new work)

1. **`LocalStorage` adapter** — drop-in replacement for the S3 calls inside `api.py`/`models.py`/`retest_file.py`. Same conceptual keys (`<report_id>/file`, `<report_id>/viz/ROC_<model>`, `<report_id>/shap_rows_<model>`, `<report_id>/*_model.joblib`, etc.), backed by directories. Keep the `S3_CONNECTION` dict shape (`client.put_object/get_object/...`) or introduce a clean `Storage` interface and refactor call sites.
2. **`LocalJobRunner`** — SQLite-backed FIFO queue + a single worker thread (CPU-bound; one job at a time). Responsibilities: take args, set Report status `Processing`, call `trainer()`, stream progress to `Report.status` (reuse the `update_training` "completed/total" message pattern), write `output_log`, set status `Processed`/`Failed`, capture exceptions + partial results (replaces `epilog.py`). Must support **cancellation** (threading.Event / process kill) — the web version cannot cancel; desktop should.
3. **SQLite schema** — `reports`, `results`, `jobs`, `settings`, optionally `user_actions` (local history). Migrations via a simple versioned schema bootstrap.
4. **Flask app + Jinja2 views** — port the 4-6 PHP views and the route handlers. The AJAX contract is already JSON, so most JS is reusable unchanged.
5. **`pywebview` shell** — boot Flask on `127.0.0.1:<random-port>`, open webview, handle lifecycle (quit → stop server → cancel jobs), tray icon, deep links.
6. **Settings UI** — data directory, theme, default `n_jobs`, max parallel jobs, optional GPU toggle, TabPFN weights cache location.
7. **Packaging pipeline** — PyInstaller spec per OS, code signing, notarization, auto-update (see §8).
8. **First-run / dependency bootstrap** — TabPFN model-weight download prompt, optional GPU/torch variant, disk-space check.
9. **Offline-friendly ML options endpoint** — the `/get-ml-options` dict is currently generated server-side; ship it as a static JSON/config so no server round-trip is needed (or keep the endpoint, it's trivial).

---

## 7. Feature-parity checklist (must match web)

- [ ] CSV upload + **column type auto-detection** (int/float/bool/categorical, yes/no & true/false → bool, dash-as-missing) — `get_column_types_internal`.
- [ ] Column preview modal: include/exclude, change type, missing-value strategy (drop / constant fill / synthetic impute via `IterativeImputer`).
- [ ] Class-column selection; categorical class → **drag-drop string→integer mapping modal**; binary yes/no/true/false auto-map.
- [ ] One-hot encoding for low-cardinality categoricals; reject high-cardinality; integer/float/bool casts.
- [ ] Separate testset upload (same transforms applied; missing-column validation; unexpected-category validation).
- [ ] Prepare page: supervised/unsupervised toggle, auto-determine-clusters, full ML-options form (train_group, parameter-tune ranges, SHAP, synthesis model/method, sampling, folds/repeats, test_size, scaler, random_state, …).
- [ ] Training submission + **status polling** (Preview → Processing "n/total" → Processed/Failed).
- [ ] Results Table (report.csv via DataTables with metric tooltips).
- [ ] Visualizations (ROC, confusion, SHAP beeswarm per class, cluster plots, metric heatmaps, train/test/CV comparison).
- [ ] Download Data (dataset, report.csv, results.json, joblib models, scaler, output log).
- [ ] **Re-Test Models** on a new testset (`retest_file.py`), with feature-count mismatch messaging.
- [ ] **Prediction Insights** — per-row SHAP impact bar chart (`download-shap-row-graph`) + `shap_rows_<model>` table.
- [ ] Output Log viewer.
- [ ] Dataset duplicate (`copy_dataset`) + delete (`delete_dataset`) with original-file retention logic.
- [ ] Comments on reports.
- [ ] Rerun with previous parameters (`prepareParameters` / `get_parameters` → now read from local job record, not ClearML).
- [ ] All 12 supervised + 3 unsupervised (kmeans/spectral/hdbscan) models, Optuna tuning, CV with confidence intervals.
- [ ] Synthetic data generation (SDV tabular/ctgan/copulagan/tvae) — original augmentation and/or new dataset.

---

## 8. Cross-platform packaging & distribution

### 8.1 Build matrix (PyInstaller cannot cross-compile)
- **Windows 10/11 x64** — built on a Windows runner.
- **Linux x64** (Ubuntu 20.04+ glibc) — built on Linux; AppImage + `.deb` + `.tar.gz`.
- **macOS** — separate **arm64 (Apple Silicon)** and **x86_64 (Intel)** builds, or a universal2 build. Notarize + staple.

Recommended CI: GitHub Actions with `windows-latest`, `ubuntu-latest`, `macos-14` (arm64) + `macos-13` (intel) runners. Each runner `pip install`s the pinned `requirements.txt` and runs `pyinstaller classify.spec`.

### 8.2 Code signing & trust (required for "production ready")
- **Windows**: Authenticode signing cert (OV/EV). Unsigned → SmartScreen warnings. EV avoids some warnings.
- **macOS**: Apple Developer ID Application cert + **notarization** (`notarytool`) + stapling. Required or Gatekeeper blocks launch.
- **Linux**: optional GPG-signed repos; AppImage is self-contained.

### 8.3 Auto-update
Options:
- `pyupdater` (PyInstaller-aware, supports delta patches) — recommended if you want in-app updates.
- Or a lightweight custom check: app pings a release JSON, downloads the new installer, prompts user to install.
- macOS: consider Sparkle (via a thin objc bridge) for native update UX.
At minimum ship a "Check for updates" menu item.

### 8.4 Data directory layout
```
<appdata>/CLASSify/
 ├── classify.db                 (SQLite)
 ├── datasets/<report_id>/
 │    ├── file, original_file, testset, retest
 │    ├── report.csv, results.json, output_log
 │    ├── <model>_model.joblib, scaler.joblib
 │    ├── shap_rows_<model>
 │    └── viz/ROC_*, viz/SHAP_*, ...
 ├── cache/                       (TabPFN weights, matplotlib font cache, etc.)
 ├── logs/
 └── settings.json
```
`<appdata>` = `%APPDATA%` (Win), `~/.local/share` (Linux), `~/Library/Application Support` (mac). Offer a "portable mode" (data next to binary) for USB installs.

---

## 9. Heavy-dependency & installer-size strategy (the hard part)

Pinned deps from `requirements.txt` that dominate size:
- `torch==2.8.0` — ~2 GB (CPU) / 2.5 GB+ (CUDA). **Largest single cost.**
- `tabpfn==2.1.3` — depends on torch + downloads model weights from HuggingFace Hub at first use (hundreds of MB).
- `sdv`/`ctgan`/`copulagan`/`tvae` — also pull torch (shared).
- `xgboost`, `shap`, `scikit-learn`, `hdbscan`, `pyamg`, `numba`, `matplotlib`, `plotly`, `reportlab`.

Realistic bundled size: **~2.5–4 GB** per platform. Mitigations:
1. **CPU-only torch wheel** by default (skip CUDA) — cuts ~1 GB and avoids driver issues. Offer a separate "GPU build" or a downloadable GPU-torch plugin for users with NVIDIA cards.
2. **TabPFN weights not bundled** — prompt on first TabPFN use to download from HF into `<appdata>/cache` (with disk-space + offline handling). Or pre-bundle as an optional "TabPFN add-on" installer.
3. **PyInstaller `--onedir`** (not `--onefile`) — faster startup, smaller per-file overhead, easier updates.
4. **Exclude unused** clearml/clearml-agent/boto3/s3transport/alembic/sqlalchemy (if SQLite uses stdlib `sqlite3`) — removes several hundred MB.
5. Consider a **"Lite" build** without `sdv`+`tabpfn` (no synthesis, no TabPFN model) for a ~600 MB installer, with these as optional downloads.
6. Strip test data, `__pycache__`, and `.pyc`-compile for a small size win.

Note: TabPFN and SDV both require torch — if you keep either, torch stays. This is the single biggest packaging decision.

---

## 10. Compute & UX considerations

- **No more DGX offload.** The web version pushed heavy jobs to a GPU cluster; desktop runs on the user's CPU. Set expectations in the UI: estimated time, model-count cost, and a warning for expensive combos (tabpfn + large N, neuralnetwork + high `n_iter`, spectral clustering on big data).
- **`n_jobs`** defaults to `os.cpu_count()` (already set in `api.py`); surface it in Settings.
- **Single job at a time** (CPU-bound). Queue additional submissions; show queue position. Optionally allow N small jobs if memory permits.
- **Progress reporting**: reuse the `update_training` "completed/total" callback → write to `Report.status` → UI polls (or use webview JS bridge / SSE for live updates). Also stream stdout to `output_log` live.
- **Cancellation**: add a Stop button → `threading.Event` passed into the trainer; for runaway subprocess-heavy steps (Optuna spawns parallel CV) consider running the job in a **subprocess** that can be hard-killed, with results collected from disk. (The web version has no cancellation; this is a net improvement.)
- **Memory**: 500 MB CSV (current web limit) is fine locally but watch RAM during one-hot expansion and `IterativeImputer`. Add a row/col sanity check + memory estimate before training.
- **Crash recovery**: replace `epilog.py`'s SLURM logic with a `try/except` around `trainer()` that still writes a partial `report.csv`/`results.json` + `output_log` and marks the job `Failed` with the traceback.

---

## 11. Security & data handling

- **Local-only data is a selling point** — the web UI literally says "(Not HIPAA-Compliant)" because data leaves for S3. Desktop keeps everything on disk; remove that disclaimer and market the privacy benefit.
- Optional: **at-rest encryption** of the data directory (SQLite SQLCipher, or an encrypted volume) for sensitive datasets — a differentiator vs. the web version.
- No remote auth, but still: bind the local server to **`127.0.0.1` only** (never `0.0.0.0`), use a random port, and optionally a ephemeral token to prevent other local apps from hitting the API. (The web `apiKey_required` middleware can become a per-launch random token check.)
- No telemetry by default; if added, make it opt-in.

---

## 12. Phased plan & rough effort

Assumes ~2 engineers, Python+frontend competent. Sizes are SWAGs.

**Phase 0 — Spike (1–2 wks)**
- Stand up pywebview + Flask hello-world, PyInstaller build on all 3 OSes.
- Prove torch/tabpfn bundle size + first-run HF download on each OS.
- Decide torch CPU-vs-GPU + Lite-build strategy (§9).
- *Gate: confirm acceptable installer size & startup time before committing.*

**Phase 1 — Storage + DB + runner (2–3 wks)**
- `LocalStorage` adapter; refactor `api.py`/`models.py`/`retest_file.py` off S3.
- SQLite schema (`reports`/`results`/`jobs`/`settings`).
- `LocalJobRunner` with queue, status, cancellation, partial-result save.
- CLI-driven end-to-end test: upload CSV → train → results on disk (no UI yet).

**Phase 2 — Frontend port (3–4 wks)**
- Port `home`, `prepare`, `results/index`, `results/details` to Jinja2.
- Port `ReportsController`/`ResultsController` to Flask routes (drop auth/tenant guards).
- Wire JS AJAX to local endpoints; keep Bootstrap/DataTables/selectpicker.
- Drag-drop mapping modal, testset upload, re-test, SHAP row graph, downloads.

**Phase 3 — Feature parity + polish (2–3 wks)**
- All 12+3 models, Optuna, CV/CI, synthesis, SHAP beeswarm/multiclass.
- Settings UI, dataset duplicate/delete, comments, rerun-with-params.
- Output log streaming, progress UI, cancellation.
- Theme port (`variables.css.php` → static CSS).

**Phase 4 — Packaging & release (2–3 wks)**
- PyInstaller specs per OS; CI build matrix; code signing (Win) + notarization (mac).
- Auto-update mechanism; first-run wizard (data dir, TabPFN weights, GPU check).
- Installers: Win `.exe` (NSIS/Inno), Linux AppImage+deb, mac `.dmg`.
- Smoke tests on clean VMs per OS.

**Phase 5 — Hardening (ongoing)**
- Crash reporting, telemetry (opt-in), accessibility audit (the web UI already has ARIA/tooltips — preserve).
- Disk-space/memory guards, large-dataset handling.
- Docs + in-app user guide (the `user_guide` plugin becomes a local page).

**Total to v1.0: ~10–15 weeks** for two engineers, gated by Phase 0 spike results (torch size) and signing/notarization lead times.

---

## 13. Risks & open questions

| Risk | Impact | Mitigation |
|---|---|---|
| torch/TabPFN bundle too large for casual distribution | High | Lite build; optional add-ons; CPU-only default |
| TabPFN needs HF download (offline fails) | Medium | Pre-bundle weights as optional add-on; clear first-run prompt |
| Local CPU too slow for tabpfn/neuralnetwork/large tuning | Medium | Time estimates; allow model subset; GPU build |
| PyInstaller hidden-import / hook issues (numba, torch, shap, xgboost native libs) | Medium | Thorough per-OS smoke tests; maintain hooks/`runtime-hooks` |
| macOS notarization + arm64/intel matrix complexity | Medium | CI on macos-14 + macos-13; universal2 fallback |
| Cancellation of CPU-bound parallel CV (Optuna) | Medium | Run job in subprocess; kill on cancel |
| SHAP `Explainer` memory on wide datasets | Low | Existing `shap_sample_size`/`shap_diagram_features` caps |
| `IterativeImputer` memory on big CSVs | Low | Row/size guard + warning |
| Loss of multi-tenant/user-audit features some users rely on | Low | Keep local `user_actions` history; note in docs |
| matplotlib backend (`Agg`) in bundle | Low | Already set to `Agg`; verify in packaged run |

**Open questions for you:**
1. **GPU support in v1?** Determines torch variant and a second build matrix. (Recommend: CPU-only v1, GPU as optional.)
2. **TabPFN + SDV in the base installer, or as downloadable add-ons?** Drives installer size (§9).
3. **Auto-update required at v1**, or manual download OK?
4. **Local encryption at rest** required (SQLCipher/encrypted volume), or nice-to-have?
5. **Keep the admin usage report** (`create_report.py`) as a local stats dashboard, or drop?
6. **Tray/resident app** (jobs continue when window closed) or quit-on-close?
7. Single combined codebase with the web project (shared `models.py` via git subtree/submodule), or fork? A shared ML core keeps both in sync.

---

## 14. Decisions needed to start

1. Confirm shell choice (recommend **pywebview + Flask**).
2. Confirm torch/GPU + Lite-build strategy (recommend **CPU-only base, GPU + TabPFN/SDV as add-ons**).
3. Confirm code-signing budget (Windows cert + Apple Developer ID).
4. Confirm repo strategy: fork `CLASSify-app` sharing the ML core with `CLASSify-2`.
5. Confirm target macOS arch (arm64-only vs universal2).

Once these are set, Phase 0 spike can begin immediately.
