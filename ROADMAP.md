# CLASSify Desktop — Roadmap

Items deliberately deferred from v1 (see `IMPLEMENTATION_CHECKLIST.md` Phase N5 and the locked decisions). Each entry: rationale for deferral, what it unlocks, rough size, dependencies, and a stub of the implementation approach so it can be picked up later without re-discovery.

v1 deliberately ships a **CPU-only, single-user, manually-updated, torch-free-base** desktop app with feature parity to CLASSify-2's core ML workflow. Everything below is post-v1, ordered by value-to-effort.

---

## Tier 1 — High value, ship soon after v1

### R1. Real in-app auto-update
- **Deferred because**: signing/notarization pipelines are enough for v1; a real updater adds packaging complexity (delta patches, atomic swap of a running app, per-OS restart behavior) that isn't worth blocking v1.
- **Unlocks**: seamless upgrades; users stay current without manual download.
- **Size**: M (2–3 weeks).
- **Depends on**: `latest.json` manifest (already shipped v1), signing infra.
- **Approach**:
  - **macOS**: Sparkle (via a thin PyObjC bridge) — the de-facto Mac updater; handles app-swap + restart + delta updates. Bundle `AutoUpdate.xml` + EdDSA public key.
  - **Windows**: `pyupdater` (PyInstaller-aware, supports delta patches, EdDSA signatures) or a custom "download installer → run with `/SILENT /NOCANCEL` → relaunch" flow using the Inno Setup installer.
  - **Linux**: AppImage has `AppImageUpdate` (zsync2 delta updates) built in; `.deb` users use apt if we host a repo, otherwise prompt to download.
  - Update channel support: `stable` / `beta` toggles in Settings; `latest.json` gains a `channel` field.
  - Backwards-compat migration: each release ships a `migrate_<from>_to_<to>.py` hook for DB schema or on-disk layout changes (Alembic handles DB; a small `migrations/fs/` runner handles data-dir layout).

### R2. GPU (CUDA) support
- **Deferred because**: doubles the build matrix and the installer size; most v1 users are fine on CPU for the 10 base models.
- **Unlocks**: TabPFN and SDV synthesis become practical on real datasets; large neural-network/spectral jobs go from hours to minutes.
- **Size**: L (3–5 weeks incl. CI matrix).
- **Depends on**: addon system (Phase J).
- **Approach**: ship as a **GPU addon** (`gpu_addon`) using the same `pip install --target` + `sys.path` mechanism as TabPFN/SDV, but with the CUDA torch wheel. At install time: detect NVIDIA GPU + CUDA runtime via `nvidia-smi`/`torch.cuda.is_available()`; if absent, refuse install with a clear message. Engine checks `backends.is_available('gpu')` before dispatching to a CUDA path; TabPFN/SDV auto-use CUDA when the addon is present. Keep CPU torch as a fallback. CI gains `windows-gpu`/`linux-gpu` legs (self-hosted runners with GPUs, or skip GPU tests in cloud CI and run them on a dedicated GPU host nightly).

### R3. Interactive Plotly charts (replace/augment matplotlib PNGs)
- **Deferred because**: matplotlib PNGs are good enough for v1 parity and require zero new frontend charting work.
- **Unlocks**: hover tooltips, zoom, pan, export, model comparison overlays — much better results exploration.
- **Size**: M (2–3 weeks).
- **Approach**: port `visualization.py` generators to emit Plotly JSON (server-side via `plotly`) served by the existing viz endpoints as `application/json`; frontend renders with `svelte-plotly` (or `@plotly/d3` directly). Keep matplotlib PNG as a "download image" export from the interactive chart. ROC, confusion-matrix heatmap, metric-heatmap, cluster scatter, SHAP beeswarm (via `shap.plots._waterfall` JSON) are the prime candidates. SHAP beeswarm interactivity is the biggest UX win.

---

## Tier 2 — Valuable, medium effort

### R4. Optional local LLM-assisted analysis
- **Deferred because**: the web LLM plugin (`LLMUtility.php`) is optional and not part of core ML parity.
- **Unlocks**: "explain these results in plain language", "which features matter most and why", "suggest next experiments" — a real differentiator for non-ML-expert users.
- **Size**: M (2–4 weeks).
- **Approach**: ship an **LLM addon** with a pluggable backend: (a) OpenAI-compatible local server (Ollama / llama.cpp server / LM Studio) via the user's base URL + key in Settings, or (b) a remote OpenAI/Anthropic API key. Strict opt-in; data only sent to the chosen endpoint, never to us. Feed the engine `results.json` + metric-defs + a compact feature-importance summary as context; render the response in a new "AI Insights" results tab. Include a system prompt that constrains it to the provided metrics and refuses to invent numbers. Never stream raw dataset rows to the model.

### R5. Encryption-at-rest hardening
- **Deferred because**: optional, off-by-default encryption exists in v1 (SQLCipher + AES-GCM data dir), but it's basic.
- **Unlocks**: genuinely private handling of sensitive (e.g. clinical) datasets on shared machines.
- **Size**: S–M (1–2 weeks).
- **Approach**:
  - Argon2id key derivation from the user passphrase (replace the v1 simple KDF).
  - Per-report file keys derived from a master key so individual datasets can be deleted securely.
  - Secure key caching in the OS keyring (keyring lib) with a re-prompt-after-idle policy.
  - Memory-zeroization of decrypted DataFrames after use where feasible.
  - A "wipe dataset" action that overwrites then deletes (best-effort on SSDs).
  - Document limitations honestly (no full-disk encryption substitute; swap/hibernation caveats).

### R6. Model export & external prediction (ONNX / standalone predictor)
- **Deferred because**: not in the web version; pure new value.
- **Unlocks**: take a trained model out of CLASSify and run it in production / another tool / hand to a collaborator.
- **Size**: M (2–3 weeks).
- **Approach**:
  - Export a trained pipeline (scaler + model) as a single `joblib` bundle + a small `predict.py` CLI already exists conceptually (retest loads joblib). Productize it: "Export model → zip with predict.py + requirements + the scaler + a sample input schema".
  - ONNX export for the tree-based models (`skl2onnx`) and logistic/SGD where supported; clearly mark unsupported models (TabPFN, spectral clustering, HDBSCAN) as joblib-only.
  - A "Predict on new CSV" desktop action that loads a saved model and writes a predictions CSV — essentially retest generalized to any external model dir.

### R7. Multi-profile / multi-project workspace
- **Deferred because**: v1 is single-user, single implicit workspace.
- **Unlocks**: organize datasets/projects; switch contexts; share a project folder with a collaborator (via a synced dir).
- **Size**: M (2–3 weeks).
- **Approach**: introduce a `Project` entity (name, data subdirectory, description, color). Datasets belong to a project. Sidebar project switcher. Each project is just a subdirectory under appdata, so a user can point a project at an arbitrary folder (e.g. a Dropbox/OneDrive folder) for cross-machine sync — document that concurrent edits across machines are not safe. This is the lightweight, local-first replacement for the web app's multi-tenant/projects plugins — no auth, no server.

---

## Tier 3 — Polish & reach

### R8. Localization / i18n
- **Deferred because**: v1 is English-only to match the web app.
- **Unlocks**: non-English users; institutional adoption.
- **Size**: S (1–2 weeks framework + ongoing translation).
- **Approach**: `svelte-i18n` for the SPA; backend error messages keyed + translated client-side via a locale bundle. Extract all strings (the web app has no i18n, so this is net-new). Crowdsource translations via a simple JSON catalog; ship en-US as the source of truth.

### R9. Accessibility enhancements
- **Deferred because**: v1 preserves the web app's existing ARIA/tooltips/keyboard work (already decent), but a full a11y audit is its own effort.
- **Unlocks**: WCAG 2.1 AA conformance; institutional procurement requirements.
- **Size**: S–M (1–3 weeks audit + fixes).
- **Approach**: axe + manual audit of every route; focus on the drag-drop class mapper (needs a keyboard-accessible reorder list fallback — the web version's drag-only mapper is an a11y gap we should fix), the TanStack Table (column sort/filter via keyboard), and live regions for job SSE status announcements. High-contrast theme variant.

### R10. Telemetry & crash reporting (opt-in)
- **Deferred because**: privacy-first default for v1; no telemetry out of the box.
- **Unlocks**: real-world bug fixes, model-failure insights.
- **Size**: S (1 week).
- **Approach**: strictly opt-in, off by default, with a clear first-run prompt. Send anonymized crash stack traces + a minimal env summary (OS, app version, model, dataset shape — never column names or row data) to a self-hosted Sentry (or a GitHub-issue-creator for crashes). Surface the toggle prominently in Settings + a "what we send" transparency list.

### R11. Batch / scheduled prediction
- **Deferred because**: retest covers ad-hoc prediction; batch is a workflow feature.
- **Unlocks**: productionize a model on recurring incoming data.
- **Size**: S (1–2 weeks).
- **Approach**: a "Watch folder" mode — point a saved model at a directory; new CSVs dropped in are auto-predicted and results written alongside. Uses the model-export predictor from R6. Tray-resident app makes this natural.

### R12. Plugin / extension SDK
- **Deferred because**: the web app's plugin system is server-side admin tooling, not user extensibility; a desktop SDK is a different beast.
- **Unlocks**: third-party custom models, custom visualizations, custom data connectors.
- **Size**: L (4+ weeks).
- **Approach**: define a Python entry-point protocol (`classify.plugins.ModelPlugin`, `VizPlugin`, `ConnectorPlugin`) discovered via the addon `sys.path` dir. Ship a `classify-plugin init` template repo. Custom models register into `train_group`; custom viz register into the Visualizations tab; connectors let users pull from SQL/REST instead of CSV. Sandboxing is best-effort (document that plugins run with full app privileges — same as any local tool).

---

## Tier 4 — Speculative / research

### R13. Distributed/remote compute backend
- **What**: optionally offload heavy jobs to a remote box (the original ClearML/DGX use case) without losing the local-first default.
- **Why later**: v1's whole point is removing that complexity; only revisit if users consistently hit CPU ceilings even with GPU.
- **Sketch**: a "Remote runner" addon that ships the job subprocess + args to a remote `classify-jobworker` over SSH or a small agent protocol; results streamed back into local storage. The local SQLite queue + SSE UI stay identical, so it's transparent to the user.

### R14. Dataset diffing & versioning
- **What**: track dataset edits over time; diff two datasets; restore a prior version.
- **Why later**: the web app stores `original_file` + `file`; full versioning is new.
- **Sketch**: content-hash each `file` write; store a lightweight commit log per report; a "History" tab with diffs (column/type/missing-strategy changes + row deltas summary). Avoid storing full copies — dedup by hash.

### R15. Collaborative sharing via signed bundles
- **What**: export a full project (dataset + model + results + report) as a single self-contained, optionally-encrypted bundle a colleague can open in their CLASSify.
- **Why later**: R7 project folders cover the simple case; bundles cover the "send to someone outside your sync" case.
- **Sketch**: `File ▸ Export Project Bundle` → zip of the project dir + a `manifest.json`; `File ▸ Import` reverses it. Encryption optional (reuse R5 keys).

---

## Maintenance track (ongoing, not version-gated)

- Keep the forked ML engine within reasonable drift of CLASSify-2's `models.py` — periodic diff review, cherry-pick modeling improvements, accept I/O-layer divergence. Tag a `ml-sync/<date>` branch each review.
- Dependency bumps: monthly Dependabot PRs; quarterly torch/numpy/sklearn major-bump spike on a branch.
- OS-compat upkeep: new macOS notarization rules, Windows EV-cert rotation, new Ubuntu LTS validation.
- Golden-dataset regression set grows as real-world bug reports come in (each user-reported modeling bug → a new `tests/fixtures/datasets/` case).
- `latest.json` + SBOM + provenance refreshed each release (CI_CD.md §10).
