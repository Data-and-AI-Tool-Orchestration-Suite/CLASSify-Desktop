<script lang="ts">
  import { onMount } from "svelte";
  import { push } from "svelte-spa-router";
  import {
    results as resultsApi,
    datasets as datasetsApi,
    system,
    type DatasetRow,
    type RunInfo,
  } from "$lib/api/client";
  import { toasts, currentJob, jobPolling, liveLog, cancelJob } from "$lib/stores/app";
  import JobProgress from "$lib/components/JobProgress.svelte";

  let { params } = $props<{ params: { reportId?: string } }>();
  let reportId = $derived(params?.reportId ?? "");

  let report = $state<DatasetRow | null>(null);
  let loading = $state(true);
  let activeTab = $state<"table" | "viz" | "download" | "retest" | "shap" | "log">("table");

  // Results data
  let resultsData = $state<{ rows: Record<string, any>[]; columns: string[] }>({
    rows: [],
    columns: [],
  });
  let vizList = $state<string[]>([]);
  let outputLog = $state("");
  let shapModel = $state("");
  let shapModels = $state<string[]>([]);
  let shapRows = $state<Record<string, any>[]>([]);
  let metricDefs = $state<Record<string, string>>({});
  let shapColumns = $state<string[]>([]);

  // Run history
  let runs = $state<RunInfo[]>([]);
  let selectedRunId = $state<string | null>(null);
  let selectedRunIsCurrent = $state(true);

  let selectedRun = $derived(runs.find((r) => r.job_id === selectedRunId) ?? null);

  async function loadAll() {
    loading = true;
    try {
      report = await datasetsApi.get(reportId);

      // If the job just finished but the report status hasn't updated yet,
      // wait briefly and retry
      if (report.status === "Processing" && !jobActive) {
        await new Promise((r) => setTimeout(r, 1500));
        report = await datasetsApi.get(reportId);
      }

      // Load run history
      try {
        const runsResp = await resultsApi.listRuns(reportId);
        runs = runsResp.runs;
        const currentRun = runs.find((r) => r.is_current);
        if (currentRun) {
          selectedRunId = currentRun.job_id;
          selectedRunIsCurrent = true;
        }
      } catch {
        // No run history
      }

      if (report.status === "Processed") {
        await Promise.all([loadRunData(), loadMetricDefs()]);
      }
    } catch {
      toasts.error("Failed to load results");
      push("/results");
    } finally {
      loading = false;
    }
  }

  async function loadRunData() {
    await Promise.all([loadResults(), loadViz(), loadLog()]);
  }

  async function loadResults() {
    try {
      const resp = selectedRunIsCurrent
        ? await resultsApi.get(reportId)
        : selectedRunId
          ? await resultsApi.runResults(reportId, selectedRunId)
          : await resultsApi.get(reportId);
      if (resp.success) {
        resultsData = { rows: resp.report_csv, columns: resp.columns };
        shapModels = resp.report_csv
          .map((r) => r.model as string)
          .filter((m) => m && m !== "kmeans" && m !== "spectralclustering" && m !== "hdbscan");
        if (shapModels.length > 0) shapModel = shapModels[0];
      } else {
        resultsData = { rows: [], columns: [] };
      }
    } catch {
      // Results not available yet
    }
  }

  async function loadViz() {
    try {
      const resp = selectedRunIsCurrent
        ? await resultsApi.vizList(reportId)
        : selectedRunId
          ? await resultsApi.runVizList(reportId, selectedRunId)
          : await resultsApi.vizList(reportId);
      vizList = resp.visualizations;
    } catch {
      vizList = [];
    }
  }

  async function loadLog() {
    try {
      const resp = selectedRunIsCurrent
        ? await resultsApi.outputLog(reportId)
        : selectedRunId
          ? await resultsApi.runOutputLog(reportId, selectedRunId)
          : await resultsApi.outputLog(reportId);
      outputLog = resp.log;
    } catch {
      outputLog = "";
    }
  }

  async function loadMetricDefs() {
    try {
      metricDefs = await system.metricDefs();
    } catch {
      // Non-fatal
    }
  }

  async function loadShapRows() {
    if (!shapModel) return;
    try {
      const resp = await resultsApi.shapRows(reportId, shapModel);
      if (resp.success) {
        shapRows = resp.rows;
        shapColumns = resp.columns;
      }
    } catch {
      // No SHAP data
    }
  }

  onMount(() => {
    loadAll();
  });

  let jobActive = $derived.by(() => {
    const job = $currentJob;
    return (
      $jobPolling &&
      job !== null &&
      job.report_uuid === reportId &&
      job.state !== "succeeded" &&
      job.state !== "failed"
    );
  });

  let wasJobActive = false;

  $effect(() => {
    const active = jobActive;
    if (wasJobActive && !active) {
      loadAll();
    }
    wasJobActive = active;
  });

  $effect(() => {
    if (activeTab === "shap" && shapModel) {
      loadShapRows();
    }
  });

  function handleRunChange() {
    selectedRunIsCurrent = selectedRun?.is_current ?? true;
    resultsData = { rows: [], columns: [] };
    vizList = [];
    outputLog = "";
    shapRows = [];
    shapColumns = [];
    if (report?.status === "Processed") {
      loadRunData();
    }
  }

  function download(suffix: string, _label: string) {
    window.open(resultsApi.downloadUrl(reportId, suffix), "_blank");
  }

  function vizImgUrl(name: string): string {
    if (selectedRunIsCurrent || !selectedRunId) {
      return resultsApi.vizUrl(reportId, name);
    }
    return resultsApi.runVizUrl(reportId, selectedRunId, name);
  }

  function formatDate(iso: string | null): string {
    if (!iso) return "unknown date";
    return new Date(iso).toLocaleString();
  }

  let logCopied = $state(false);

  async function copyLog() {
    if (!outputLog) return;
    try {
      await navigator.clipboard.writeText(outputLog);
      logCopied = true;
      setTimeout(() => (logCopied = false), 2000);
    } catch {
      toasts.error("Failed to copy log to clipboard");
    }
  }

  let liveLogEl = $state<HTMLPreElement | null>(null);

  let lightboxViz = $state<string | null>(null);

  function vizDownloadUrl(name: string): string {
    return resultsApi.downloadUrl(reportId, `viz/${name}`);
  }

  $effect(() => {
    $liveLog;
    if (liveLogEl) {
      liveLogEl.scrollTop = liveLogEl.scrollHeight;
    }
  });
</script>

{#if loading}
  <div class="text-center py-5">
    <div class="spinner-border" role="status"></div>
  </div>
{:else if report}
  <div
    class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pb-2 mb-3 border-bottom"
  >
    <div>
      <h1 class="h4">{report.filename}</h1>
      <span
        class="badge bg-{report.status === 'Processed'
          ? 'success'
          : report.status === 'Processing'
            ? 'warning'
            : 'danger'}"
      >
        {report.status}
      </span>
    </div>
    <div class="d-flex gap-2">
      {#if report.status === "Processed" || report.status === "Failed"}
        <a href={`#/prepare/${reportId}`} class="btn btn-outline-primary btn-sm">
          Edit Settings &amp; Rerun
        </a>
      {/if}
      <a href="#/results" class="btn btn-outline-secondary btn-sm">Back to Results</a>
    </div>
  </div>

  <!-- Run history selector -->
  {#if runs.length > 1 && report.status === "Processed"}
    <div class="d-flex align-items-center gap-2 mb-3">
      <label class="form-label mb-0 text-muted small fw-bold" for="run-select">Run:</label>
      <select
        id="run-select"
        class="form-select form-select-sm"
        style="max-width: 350px;"
        bind:value={selectedRunId}
        onchange={handleRunChange}
      >
        {#each runs as run}
          <option value={run.job_id}>
            {run.is_current ? "Latest" : "Archived"} — {formatDate(run.created_at)}
            {#if run.args?.train_group}
              ({Array.isArray(run.args.train_group) ? run.args.train_group.join(", ") : run.args.train_group})
            {/if}
          </option>
        {/each}
      </select>
      {#if selectedRun && !selectedRun.is_current}
        <span class="badge bg-secondary">viewing archived run</span>
      {/if}
    </div>
  {/if}

  <!-- Job progress (if processing) -->
  {#if $jobPolling && $currentJob?.report_uuid === reportId}
    <JobProgress job={$currentJob} oncancel={() => cancelJob($currentJob!.id)} />
  {/if}

  {#if jobActive}
    <div class="alert alert-warning mb-3">
      <div class="spinner-border spinner-border-sm me-2"></div>
      {#if $currentJob?.state === "queued"}
        Job queued — waiting to start...
      {:else}
        Training in progress... This page will update automatically when complete.
      {/if}
    </div>
    {#if $liveLog}
      <div class="mb-3">
        <div class="d-flex justify-content-between align-items-center mb-1">
          <h6 class="mb-0 text-muted">Live Output</h6>
          <span class="badge bg-warning text-dark">streaming</span>
        </div>
        <pre
          bind:this={liveLogEl}
          class="bg-dark text-light p-3 rounded"
          style="max-height: 400px; overflow-y: auto; font-size: 0.85rem; white-space: pre-wrap; word-break: break-word; user-select: text; cursor: text;">{$liveLog}</pre>
      </div>
    {:else}
      <div class="text-center py-3">
        <div class="spinner-border spinner-border-sm me-2"></div>
        <span class="text-muted">Waiting for output...</span>
      </div>
    {/if}
  {:else if report.status === "Processing"}
    <div class="alert alert-warning mb-3">
      <div class="spinner-border spinner-border-sm me-2"></div>
      Training in progress... This page will update automatically when complete.
    </div>
  {:else if report.status === "Failed"}
    <div class="alert alert-danger">
      Training failed. Check the Output Log tab for details.
      <a href={`#/prepare/${reportId}`} class="alert-link">Edit settings and rerun</a>.
    </div>
  {:else if report.status === "Preview" || report.status === "Uploaded"}
    <div class="alert alert-info">
      This dataset hasn't been trained yet.
      <a href={`#/prepare/${reportId}`} class="alert-link">Go to Prepare page</a>
    </div>
  {:else}
    <!-- Tabs -->
    <ul class="nav nav-tabs mb-3" role="tablist">
      <li class="nav-item" role="presentation">
        <button
          class="nav-link {activeTab === 'table' ? 'active' : ''}"
          onclick={() => (activeTab = "table")}
        >
          Results Table
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button
          class="nav-link {activeTab === 'viz' ? 'active' : ''}"
          onclick={() => (activeTab = "viz")}
        >
          Visualizations
          {#if vizList.length > 0}
            <span class="badge bg-secondary ms-1">{vizList.length}</span>
          {/if}
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button
          class="nav-link {activeTab === 'download' ? 'active' : ''}"
          onclick={() => (activeTab = "download")}
        >
          Download Data
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button
          class="nav-link {activeTab === 'shap' ? 'active' : ''}"
          onclick={() => (activeTab = "shap")}
        >
          Prediction Insights
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button
          class="nav-link {activeTab === 'log' ? 'active' : ''}"
          onclick={() => (activeTab = "log")}
        >
          Output Log
        </button>
      </li>
    </ul>

    <!-- Results Table -->
    {#if activeTab === "table"}
      {#if resultsData.rows.length > 0}
        <div class="table-responsive">
          <table class="table table-sm table-striped table-hover">
            <thead class="table-light">
              <tr>
                {#each resultsData.columns as col}
                  <th title={metricDefs[col] || ""}>{col}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each resultsData.rows as row}
                <tr>
                  {#each resultsData.columns as col}
                    <td>{row[col] ?? ""}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {:else}
        <p class="text-muted">No results data available.</p>
      {/if}
    {/if}

    <!-- Visualizations -->
    {#if activeTab === "viz"}
      {#if vizList.length > 0}
        <div class="row g-3">
          {#each vizList as viz}
            <div class="col-md-6 col-lg-4">
              <div class="card h-100">
                <button
                  type="button"
                  class="btn p-0 border-0"
                  onclick={() => (lightboxViz = viz)}
                  title="Click to enlarge"
                >
                  <img
                    src={vizImgUrl(viz)}
                    class="card-img-top"
                    alt={viz}
                    loading="lazy"
                    style="cursor: zoom-in;"
                  />
                </button>
                <div class="card-body p-2 d-flex justify-content-between align-items-center">
                  <p class="card-text small text-muted mb-0 text-truncate">{viz}</p>
                  <a
                    href={vizDownloadUrl(viz)}
                    class="btn btn-outline-secondary btn-sm ms-2 flex-shrink-0"
                    title="Download PNG"
                    download
                  >
                    Download
                  </a>
                </div>
              </div>
            </div>
          {/each}
        </div>
      {:else}
        <p class="text-muted">No visualizations available.</p>
      {/if}
    {/if}

    <!-- Download Data -->
    {#if activeTab === "download"}
      <div class="row g-3">
        <div class="col-md-4">
          <div class="card h-100">
            <div class="card-body text-center">
              <h6 class="card-title">Dataset (CSV)</h6>
              <button
                class="btn btn-outline-primary btn-sm"
                onclick={() => download("file", "Dataset")}
              >
                Download
              </button>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card h-100">
            <div class="card-body text-center">
              <h6 class="card-title">Results Report (CSV)</h6>
              <button
                class="btn btn-outline-primary btn-sm"
                onclick={() => download("results", "Report")}
              >
                Download
              </button>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card h-100">
            <div class="card-body text-center">
              <h6 class="card-title">Output Log</h6>
              <button
                class="btn btn-outline-primary btn-sm"
                onclick={() => download("output_log", "Log")}
              >
                Download
              </button>
            </div>
          </div>
        </div>
        {#each resultsData.rows as row}
          {#if row.model}
            <div class="col-md-4">
              <div class="card h-100">
                <div class="card-body text-center">
                  <h6 class="card-title">{row.model} Model</h6>
                  <button
                    class="btn btn-outline-primary btn-sm"
                    onclick={() => download(`${row.model}_model.joblib`, "Model")}
                  >
                    Download .joblib
                  </button>
                </div>
              </div>
            </div>
          {/if}
        {/each}
      </div>
    {/if}

    <!-- Prediction Insights (SHAP) -->
    {#if activeTab === "shap"}
      {#if shapModels.length > 0}
        <div class="mb-3">
          <label class="form-label" for="shap-model-select">Select Model</label>
          <select
            class="form-select"
            id="shap-model-select"
            style="max-width: 300px;"
            bind:value={shapModel}
          >
            {#each shapModels as m}
              <option value={m}>{m}</option>
            {/each}
          </select>
        </div>

        {#if shapColumns.length > 0}
          <div class="table-responsive" style="max-height: 500px; overflow-y: auto;">
            <table class="table table-sm table-striped">
              <thead class="table-light sticky-top">
                <tr>
                  {#each shapColumns as col}
                    <th>{col}</th>
                  {/each}
                </tr>
              </thead>
              <tbody>
                {#each shapRows as row}
                  <tr>
                    {#each shapColumns as col}
                      <td>{row[col] ?? ""}</td>
                    {/each}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {:else}
          <p class="text-muted">Loading SHAP data...</p>
        {/if}
      {:else}
        <p class="text-muted">
          No SHAP data available. Enable SHAP in training options to see prediction insights.
        </p>
      {/if}
    {/if}

    <!-- Output Log -->
    {#if activeTab === "log"}
      <div class="d-flex justify-content-end mb-2">
        <button
          class="btn btn-outline-secondary btn-sm"
          onclick={copyLog}
          disabled={!outputLog}
        >
          {#if logCopied}
            Copied!
          {:else}
            Copy to Clipboard
          {/if}
        </button>
      </div>
      <pre
        class="bg-dark text-light p-3 rounded"
        style="max-height: 600px; overflow-y: auto; font-size: 0.85rem; white-space: pre-wrap; word-break: break-word; user-select: text; cursor: text;">{outputLog ||
          "No output log available."}</pre>
    {/if}
  {/if}
{/if}

{#if lightboxViz}
  <div
    class="modal-backdrop fade show"
    onclick={() => (lightboxViz = null)}
    onkeydown={(e) => e.key === "Escape" && (lightboxViz = null)}
    role="presentation"
  ></div>
  <div class="modal fade show d-block" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-xl modal-dialog-centered" role="document">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title text-truncate">{lightboxViz}</h5>
          <div class="d-flex align-items-center gap-2">
            <a
              href={vizDownloadUrl(lightboxViz)}
              class="btn btn-outline-secondary btn-sm"
              download
            >
              Download
            </a>
            <button
              type="button"
              class="btn-close"
              onclick={() => (lightboxViz = null)}
              aria-label="Close"
            ></button>
          </div>
        </div>
        <div class="modal-body text-center p-2">
          <img
            src={vizImgUrl(lightboxViz)}
            class="img-fluid rounded"
            alt={lightboxViz}
            style="max-height: 80vh;"
          />
        </div>
      </div>
    </div>
  </div>
{/if}
