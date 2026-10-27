<script lang="ts">
  /**
   * Forecasting results view — metrics table, interactive forecast charts
   * (Plotly, ported from the Forecaster results page), generated future
   * forecasts, and the output log.
   */
  import { onMount } from "svelte";
  import {
    results as resultsApi,
    system as systemApi,
    type JobResponse,
    type RunInfo,
  } from "$lib/api/client";
  import { toasts } from "$lib/stores/app";
  import {
    COMPARE_PALETTE,
    FORECAST_TRACE_COLORS,
    modelLabel,
    sortDates,
    type ForecastChartData,
    type ForecastMetricRow,
    type ForecastsPayload,
  } from "$lib/forecast";
  import PlotlyChart from "./PlotlyChart.svelte";
  import ForecastMetricsTable from "./ForecastMetricsTable.svelte";
  import GenerateForecastModal from "./GenerateForecastModal.svelte";

  let { reportId, jobActive = false }: { reportId: string; jobActive?: boolean } = $props();

  let activeTab = $state<"metrics" | "charts" | "forecasts" | "log">("metrics");
  let loading = $state(true);
  let metricDefs = $state<Record<string, string>>({});

  let runs = $state<RunInfo[]>([]);
  let selectedRunId = $state<string>("latest");

  let rows = $state<ForecastMetricRow[]>([]);
  let resultsAvailable = $state(false);

  // Charts tab
  let chartModel = $state("");
  let chartGrouping = $state("");
  let chartComponent = $state("");
  let chartData = $state<ForecastChartData | null>(null);
  let chartDataMissing = $state(false);
  let chartLoadedKey = $state("");
  let compareMode = $state(false);
  let compareModels = $state<string[]>([]);
  let compareData = $state<Record<string, ForecastChartData>>({});

  // Forecasts tab
  let forecasts = $state<ForecastsPayload | null>(null);
  let forecastModel = $state("");
  let forecastGrouping = $state("");
  let forecastComponent = $state("");
  let showGenerate = $state(false);
  let generationLog = $state("");

  let outputLog = $state("");

  onMount(() => {
    loadAll();
  });

  let lastJobActive = false;
  $effect(() => {
    if (lastJobActive && !jobActive) {
      loadAll();
    }
    lastJobActive = jobActive;
  });

  async function loadAll() {
    loading = true;
    try {
      metricDefs = await systemApi.metricDefs();
      try {
        const runsResp = await resultsApi.listRuns(reportId);
        // Forecast-generation jobs are additive to the last training run —
        // they never appear as their own run in the selector.
        runs = runsResp.runs.filter((run) => run.args?.mode !== "forecast-generation");
      } catch {
        runs = [];
      }
      await loadResults();
    } catch (e) {
      toasts.error(`Failed to load results: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      loading = false;
    }
  }

  function isArchived(): boolean {
    return selectedRunId !== "latest";
  }

  async function loadResults() {
    resultsAvailable = false;
    rows = [];
    chartData = null;
    chartLoadedKey = "";
    chartDataMissing = false;
    forecasts = null;
    try {
      const resp = isArchived()
        ? await resultsApi.runResults(reportId, selectedRunId)
        : await resultsApi.get(reportId);
      if (resp.success && resp.columns.includes("model")) {
        resultsAvailable = true;
        rows = resp.report_csv.map((r) => ({
          model: String(r.model ?? ""),
          grouping: String(r.grouping ?? ""),
          component: String(r.component ?? ""),
          ...r,
        }));
        chartModel = rows[0]?.model ?? "";
        compareModels = [];
        compareData = {};
      }
    } catch {
      resultsAvailable = false;
    }
  }

  function onRunChange() {
    loadResults();
  }

  const trainedModelsWithData = $derived([
    ...new Set(rows.filter((r) => r.model !== "Baseline Average Prediction").map((r) => r.model)),
  ]);

  const chartGroupings = $derived(chartData ? Object.keys(chartData) : [""]);
  const chartComponents = $derived(
    chartData && chartData[chartGrouping] ? Object.keys(chartData[chartGrouping]) : [""],
  );

  async function loadChartData() {
    if (!chartModel) return;
    const key = `${selectedRunId}:${chartModel}`;
    chartLoadedKey = key;
    chartDataMissing = false;
    try {
      const data = isArchived()
        ? await resultsApi.runForecastData(reportId, selectedRunId, chartModel)
        : await resultsApi.forecastData(reportId, chartModel);
      chartData = data;
      if (!Object.hasOwn(data, chartGrouping)) {
        chartGrouping = Object.keys(data)[0] ?? "";
      }
      const comps = data[chartGrouping];
      if (comps && !Object.hasOwn(comps, chartComponent)) {
        chartComponent = Object.keys(comps)[0] ?? "";
      }
    } catch {
      chartData = null;
      chartDataMissing = true;
    }
  }

  $effect(() => {
    if (activeTab !== "charts" || !chartModel || compareMode) return;
    if (chartLoadedKey !== `${selectedRunId}:${chartModel}`) {
      loadChartData();
    }
  });

  function chartTraces(): Record<string, unknown>[] {
    if (!chartData) return [];
    const seriesBlock = chartData[chartGrouping]?.[chartComponent];
    if (!seriesBlock) return [];
    const traces: Record<string, unknown>[] = [];
    traces.push({
      x: seriesBlock.train_x,
      y: seriesBlock.train_y,
      name: `Training Set (${chartComponent})`,
      mode: "lines",
      line: { color: FORECAST_TRACE_COLORS.train, width: 3 },
    });
    if (seriesBlock.val_x && seriesBlock.val_x.length > 0) {
      traces.push({
        x: seriesBlock.val_x,
        y: seriesBlock.val_y,
        name: `Validation Set (${chartComponent})`,
        mode: "lines",
        line: { color: FORECAST_TRACE_COLORS.val, width: 3 },
      });
    }
    traces.push({
      x: seriesBlock.pred_x,
      y: seriesBlock.pred_y,
      name: `Predictions (${chartComponent})`,
      mode: "lines",
      line: { color: FORECAST_TRACE_COLORS.pred, width: 4, dash: "dot" },
    });
    return traces;
  }

  const compareTraces = $derived.by(() => {
    if (!compareMode || !chartData) return [];
    const traces: Record<string, unknown>[] = [];
    const base = chartData[chartGrouping]?.[chartComponent];
    if (base) {
      traces.push({
        x: base.train_x,
        y: base.train_y,
        name: `Training Set (${chartComponent})`,
        mode: "lines",
        line: { color: FORECAST_TRACE_COLORS.train, width: 3 },
      });
    }
    compareModels.forEach((model, i) => {
      const seriesBlock = compareData[model]?.[chartGrouping]?.[chartComponent];
      if (seriesBlock) {
        traces.push({
          x: seriesBlock.pred_x,
          y: seriesBlock.pred_y,
          name: `${modelLabel(model)} predictions (${chartComponent})`,
          mode: "lines",
          line: { color: COMPARE_PALETTE[(i + 1) % COMPARE_PALETTE.length], width: 3, dash: "dot" },
        });
      }
    });
    return traces;
  });

  async function toggleCompareMode() {
    compareMode = !compareMode;
    if (compareMode && compareModels.length === 0) {
      compareModels = trainedModelsWithData.slice(0, 4);
      const results: Record<string, ForecastChartData> = {};
      for (const model of compareModels) {
        try {
          results[model] = isArchived()
            ? await resultsApi.runForecastData(reportId, selectedRunId, model)
            : await resultsApi.forecastData(reportId, model);
        } catch {
          // model without viz data — skipped in the chart
        }
      }
      compareData = results;
    }
  }

  async function toggleCompareModel(model: string) {
    if (compareModels.includes(model)) {
      compareModels = compareModels.filter((m) => m !== model);
      return;
    }
    if (compareModels.length >= 4) {
      toasts.warning(
        "You cannot generate a plot with more than four distinct series. Use the filters to limit the components/groupings.",
      );
      return;
    }
    compareModels = [...compareModels, model];
    try {
      const data = isArchived()
        ? await resultsApi.runForecastData(reportId, selectedRunId, model)
        : await resultsApi.forecastData(reportId, model);
      compareData = { ...compareData, [model]: data };
    } catch {
      // no viz data for this model
    }
  }

  const chartLayout = $derived({
    xaxis: { title: { text: "Timestamp" }, type: "date", tickformat: "%Y-%m-%d" },
    yaxis: { title: { text: "Value" } },
    hovermode: "x unified",
    legend: { orientation: "h", x: 0, y: 1.12 },
    margin: { t: 40 },
  });

  const forecastModelOptions = $derived(forecasts ? Object.keys(forecasts) : []);

  const forecastGroupings = $derived(
    forecasts?.[forecastModel]?.groupings ? Object.keys(forecasts[forecastModel].groupings!) : [""],
  );

  const forecastComponents = $derived.by(() => {
    const block = forecasts?.[forecastModel];
    if (!block) return [""];
    if (block.groupings) {
      return Object.keys(block.groupings[forecastGrouping]?.components ?? {});
    }
    return Object.keys(block.components ?? {});
  });

  function forecastSeries(): Record<string, number> | null {
    const block = forecasts?.[forecastModel];
    if (!block) return null;
    if (block.groupings) {
      return block.groupings[forecastGrouping]?.components?.[forecastComponent] ?? null;
    }
    return block.components?.[forecastComponent] ?? null;
  }

  const forecastTraces = $derived.by(() => {
    const seriesBlock = forecastSeries();
    if (!seriesBlock) return [];
    const dates = sortDates(Object.keys(seriesBlock));
    return [
      {
        x: dates,
        y: dates.map((d) => seriesBlock[d]),
        name: `Future Forecasts (${forecastComponent})`,
        mode: "lines",
        line: { color: FORECAST_TRACE_COLORS.forecast, width: 4, dash: "dash" },
      },
    ];
  });

  const forecastTableRows = $derived.by(() => {
    const seriesBlock = forecastSeries();
    if (!seriesBlock) return [];
    return sortDates(Object.keys(seriesBlock)).map((d) => ({ date: d, value: seriesBlock[d] }));
  });

  async function loadForecasts() {
    try {
      const data = isArchived()
        ? await resultsApi.runForecasts(reportId, selectedRunId)
        : await resultsApi.forecasts(reportId);
      forecasts = data;
      if (!Object.hasOwn(data, forecastModel)) {
        forecastModel = Object.keys(data)[0] ?? "";
      }
      const block = data[forecastModel];
      if (block?.groupings && !Object.hasOwn(block.groupings, forecastGrouping)) {
        forecastGrouping = Object.keys(block.groupings)[0] ?? "";
      }
      const comps = block?.groupings
        ? block.groupings[forecastGrouping]?.components
        : block?.components;
      if (comps && !Object.hasOwn(comps, forecastComponent)) {
        forecastComponent = Object.keys(comps)[0] ?? "";
      }
    } catch {
      forecasts = null;
    }
    if (!isArchived()) {
      try {
        const resp = await resultsApi.generationLog(reportId);
        generationLog = resp.success ? resp.log : "";
      } catch {
        generationLog = "";
      }
    } else {
      generationLog = "";
    }
  }

  $effect(() => {
    if (activeTab === "forecasts" && !forecasts) {
      loadForecasts();
    }
  });

  async function loadOutputLog() {
    try {
      const resp = isArchived()
        ? await resultsApi.runOutputLog(reportId, selectedRunId)
        : await resultsApi.outputLog(reportId);
      outputLog = resp.log;
    } catch {
      outputLog = "";
    }
  }

  $effect(() => {
    if (activeTab === "log" && !outputLog) {
      loadOutputLog();
    }
  });

  function handleForecastStarted(_job: JobResponse) {
    showGenerate = false;
  }

  function runLabel(run: RunInfo): string {
    const models = run.args?.train_group;
    const list = Array.isArray(models) ? (models as string[]).join(", ") : "";
    const date = run.created_at ? new Date(run.created_at).toLocaleString() : "unknown date";
    return `${run.is_current ? "Latest" : date}${list ? ` — ${list}` : ""}`;
  }
</script>

{#if loading}
  <div class="text-center py-5">
    <div class="spinner-border" role="status">
      <span class="visually-hidden">Loading...</span>
    </div>
  </div>
{:else}
  {#if runs.length > 1}
    <div class="d-flex align-items-center gap-2 mb-3">
      <label class="form-label small mb-0" for="fc-run-select">Run</label>
      <select
        id="fc-run-select"
        class="form-select form-select-sm"
        style="max-width: 480px;"
        bind:value={selectedRunId}
        onchange={onRunChange}
      >
        {#each runs as run (run.job_id)}
          <option value={run.job_id}>{runLabel(run)}</option>
        {/each}
      </select>
    </div>
  {/if}

  <ul class="nav nav-tabs mb-3">
    <li class="nav-item">
      <button
        class="nav-link {activeTab === 'metrics' ? 'active' : ''}"
        onclick={() => (activeTab = "metrics")}
        type="button"
      >
        Metrics
      </button>
    </li>
    <li class="nav-item">
      <button
        class="nav-link {activeTab === 'charts' ? 'active' : ''}"
        onclick={() => (activeTab = "charts")}
        type="button"
      >
        Forecast Charts
      </button>
    </li>
    <li class="nav-item">
      <button
        class="nav-link {activeTab === 'forecasts' ? 'active' : ''}"
        onclick={() => {
          activeTab = "forecasts";
          loadForecasts();
        }}
        type="button"
      >
        Future Forecasts
      </button>
    </li>
    <li class="nav-item">
      <button
        class="nav-link {activeTab === 'log' ? 'active' : ''}"
        onclick={() => (activeTab = "log")}
        type="button"
      >
        Output Log
      </button>
    </li>
  </ul>

  {#if activeTab === "metrics"}
    {#if resultsAvailable}
      <div class="card shadow-sm mb-3">
        <div class="card-header d-flex justify-content-between align-items-center">
          <h5 class="mb-0">Forecasting Results</h5>
          <a
            class="btn btn-sm btn-outline-secondary"
            href={resultsApi.downloadUrl(reportId, "results")}
            download
          >
            Download Results CSV
          </a>
        </div>
        <div class="card-body">
          <ForecastMetricsTable {rows} {metricDefs} />
          <div class="d-flex flex-wrap gap-2 mt-2">
            {#each trainedModelsWithData as model (model)}
              <a
                class="btn btn-sm btn-outline-primary"
                href={resultsApi.downloadUrl(reportId, `${model}_model.joblib`)}
                download
                title="Download the saved model"
              >
                {modelLabel(model)}
              </a>
            {/each}
          </div>
          <p class="text-muted small mb-0 mt-2">
            The naive baseline model is not saved and cannot be downloaded.
          </p>
        </div>
      </div>
    {:else}
      <div class="alert alert-info">
        No forecasting results yet for this run. Start a training job from the Prepare page.
      </div>
    {/if}
  {:else if activeTab === "charts"}
    <div class="card shadow-sm mb-3">
      <div class="card-header">
        <h5 class="mb-0">Time Series Forecast</h5>
      </div>
      <div class="card-body">
        {#if !resultsAvailable}
          <div class="alert alert-info mb-0">Train a model first to see forecast charts.</div>
        {:else}
          <div class="d-flex flex-wrap gap-3 align-items-end mb-3">
            <div>
              <label class="form-label small mb-1" for="fc-chart-model">Model</label>
              <select
                id="fc-chart-model"
                class="form-select form-select-sm"
                bind:value={chartModel}
                disabled={compareMode}
              >
                {#each trainedModelsWithData as model (model)}
                  <option value={model}>{modelLabel(model)}</option>
                {/each}
              </select>
            </div>
            {#if !compareMode}
              {#if chartGroupings.length > 1}
                <div>
                  <label class="form-label small mb-1" for="fc-chart-grouping">Grouping</label>
                  <select
                    id="fc-chart-grouping"
                    class="form-select form-select-sm"
                    bind:value={chartGrouping}
                  >
                    {#each chartGroupings as g (g)}
                      <option value={g}>{g === "" ? "All Groupings" : g}</option>
                    {/each}
                  </select>
                </div>
              {/if}
              {#if chartComponents.length > 1}
                <div>
                  <label class="form-label small mb-1" for="fc-chart-component">Component</label>
                  <select
                    id="fc-chart-component"
                    class="form-select form-select-sm"
                    bind:value={chartComponent}
                  >
                    {#each chartComponents as c (c)}
                      <option value={c}>{c}</option>
                    {/each}
                  </select>
                </div>
              {/if}
            {:else}
              <div>
                <span class="form-label small mb-1 d-block">Compare models</span>
                <div class="d-flex flex-wrap gap-2">
                  {#each trainedModelsWithData as model (model)}
                    <div class="form-check mb-0">
                      <input
                        class="form-check-input"
                        type="checkbox"
                        id={`cmp-${model}`}
                        checked={compareModels.includes(model)}
                        onchange={() => toggleCompareModel(model)}
                      />
                      <label class="form-check-label small" for={`cmp-${model}`}>
                        {modelLabel(model)}
                      </label>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
            <button class="btn btn-sm btn-outline-secondary" onclick={toggleCompareMode}>
              {compareMode ? "Single model view" : "Compare models"}
            </button>
          </div>

          <p class="text-muted small">
            Plot Tips: Double-click to reset zoom / Click and drag to zoom in / Click legend items
            to toggle off / Double-click legend items to isolate
          </p>

          {#if chartDataMissing}
            <div class="alert alert-warning mb-0">
              No visualization data found for this model. Enable "Create Visualizations" when
              training to generate forecast charts.
            </div>
          {:else if compareMode}
            <PlotlyChart data={compareTraces} layout={chartLayout} />
          {:else}
            <PlotlyChart data={chartTraces()} layout={chartLayout} />
          {/if}
        {/if}
      </div>
    </div>
  {:else if activeTab === "forecasts"}
    <div class="card shadow-sm mb-3">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">Future Forecasts</h5>
        <div class="d-flex gap-2">
          {#if forecasts}
            <a
              class="btn btn-sm btn-outline-secondary"
              href={resultsApi.downloadUrl(reportId, "forecasts")}
              download
            >
              Download Future Forecasts
            </a>
          {/if}
          <button class="btn btn-sm btn-primary" onclick={() => (showGenerate = true)}>
            Generate Future Forecasts
          </button>
        </div>
      </div>
      <div class="card-body">
        {#if forecasts && forecastModelOptions.length > 0}
          <div class="d-flex flex-wrap gap-3 align-items-end mb-3">
            <div>
              <label class="form-label small mb-1" for="fc-fc-model">Model</label>
              <select
                id="fc-fc-model"
                class="form-select form-select-sm"
                bind:value={forecastModel}
              >
                {#each forecastModelOptions as m (m)}
                  <option value={m}>{modelLabel(m)}</option>
                {/each}
              </select>
            </div>
            {#if forecastGroupings.length > 1}
              <div>
                <label class="form-label small mb-1" for="fc-fc-grouping">Grouping</label>
                <select
                  id="fc-fc-grouping"
                  class="form-select form-select-sm"
                  bind:value={forecastGrouping}
                >
                  {#each forecastGroupings as g (g)}
                    <option value={g}>{g}</option>
                  {/each}
                </select>
              </div>
            {/if}
            {#if forecastComponents.length > 1}
              <div>
                <label class="form-label small mb-1" for="fc-fc-component">Component</label>
                <select
                  id="fc-fc-component"
                  class="form-select form-select-sm"
                  bind:value={forecastComponent}
                >
                  {#each forecastComponents as c (c)}
                    <option value={c}>{c}</option>
                  {/each}
                </select>
              </div>
            {/if}
          </div>
          <PlotlyChart data={forecastTraces} layout={chartLayout} />
          {#if forecastTableRows.length > 0}
            <h6 class="mt-3">Forecast Values — {forecastComponent}</h6>
            <div class="table-scroll-container">
              <table class="table table-sm table-striped align-middle w-auto">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Forecast</th>
                  </tr>
                </thead>
                <tbody>
                  {#each forecastTableRows as row (row.date)}
                    <tr>
                      <td>{row.date}</td>
                      <td>{row.value}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
          {#if generationLog}
            <details class="mt-3">
              <summary class="text-muted small" style="cursor: pointer;">
                Forecast generation log
              </summary>
              <pre class="log-output mt-2 mb-0">{generationLog}</pre>
            </details>
          {/if}
        {:else}
          <div class="alert alert-info mb-0">
            No future forecasts generated yet. Click "Generate Future Forecasts" to refit the saved
            models and predict ahead.
          </div>
        {/if}
      </div>
    </div>
  {:else}
    <div class="card shadow-sm mb-3">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="mb-0">Output Log</h5>
        <button
          class="btn btn-sm btn-outline-secondary"
          onclick={() => navigator.clipboard.writeText(outputLog)}
        >
          Copy
        </button>
      </div>
      <div class="card-body">
        <pre class="log-output mb-0">{outputLog || "No log output available."}</pre>
      </div>
    </div>
  {/if}

  {#if showGenerate}
    <GenerateForecastModal
      {reportId}
      trainedModels={trainedModelsWithData}
      onclose={() => (showGenerate = false)}
      onstarted={handleForecastStarted}
    />
  {/if}
{/if}
