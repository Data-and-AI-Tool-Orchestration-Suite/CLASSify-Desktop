<script lang="ts">
  import { onMount } from "svelte";
  import { push } from "svelte-spa-router";
  import { results as resultsApi, datasets as datasetsApi, type DatasetRow } from "$lib/api/client";
  import { toasts, currentJob, jobPolling, cancelJob } from "$lib/stores/app";
  import JobProgress from "$lib/components/JobProgress.svelte";

  let { reportId } = $props<{ reportId: string }>();

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
  let shapColumns = $state<string[]>([]);

  async function loadAll() {
    loading = true;
    try {
      report = await datasetsApi.get(reportId);

      if (report.status === "Processed") {
        await Promise.all([loadResults(), loadViz(), loadLog()]);
      }
    } catch {
      toasts.error("Failed to load results");
      push("/results");
    } finally {
      loading = false;
    }
  }

  async function loadResults() {
    try {
      const resp = await resultsApi.get(reportId);
      if (resp.success) {
        resultsData = { rows: resp.report_csv, columns: resp.columns };
        // Detect models with SHAP data
        shapModels = resp.report_csv
          .map((r) => r.model as string)
          .filter((m) => m && m !== "kmeans" && m !== "spectralclustering" && m !== "hdbscan");
        if (shapModels.length > 0) shapModel = shapModels[0];
      }
    } catch {
      // Results not available yet
    }
  }

  async function loadViz() {
    try {
      const resp = await resultsApi.vizList(reportId);
      vizList = resp.visualizations;
    } catch {
      // No visualizations
    }
  }

  async function loadLog() {
    try {
      const resp = await resultsApi.outputLog(reportId);
      outputLog = resp.log;
    } catch {
      // No log
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

  $effect(() => {
    if (activeTab === "shap" && shapModel) {
      loadShapRows();
    }
  });

  function download(suffix: string, _label: string) {
    window.open(resultsApi.downloadUrl(reportId, suffix), "_blank");
  }

  function vizImgUrl(name: string): string {
    return resultsApi.vizUrl(reportId, name);
  }
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
    <a href="#/results" class="btn btn-outline-secondary btn-sm">Back to Results</a>
  </div>

  <!-- Job progress (if processing) -->
  {#if $jobPolling && $currentJob?.report_uuid === reportId}
    <JobProgress job={$currentJob} oncancel={() => cancelJob($currentJob!.id)} />
  {/if}

  {#if report.status === "Processing"}
    <div class="alert alert-warning">
      <div class="spinner-border spinner-border-sm me-2"></div>
      Training in progress... This page will update automatically when complete.
    </div>
  {:else if report.status === "Failed"}
    <div class="alert alert-danger">Training failed. Check the Output Log tab for details.</div>
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
                  <th>{col}</th>
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
              <div class="card">
                <img src={vizImgUrl(viz)} class="card-img-top" alt={viz} loading="lazy" />
                <div class="card-body p-2">
                  <p class="card-text text-center small text-muted mb-0">{viz}</p>
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
      <pre
        class="bg-dark text-light p-3 rounded"
        style="max-height: 600px; overflow-y: auto; font-size: 0.85rem;">{outputLog ||
          "No output log available."}</pre>
    {/if}
  {/if}
{/if}
