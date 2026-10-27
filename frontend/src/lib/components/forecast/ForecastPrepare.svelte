<script lang="ts">
  /**
   * Forecasting-mode training configuration (embedded in Prepare.svelte).
   * Ported from the Forecaster web app's job configuration wizard.
   */
  import { push } from "svelte-spa-router";
  import {
    addons as addonsApi,
    datasets as datasetsApi,
    jobs as jobsApi,
    type DatasetRow,
    type TrainOption,
    type TsRoleAssignment,
  } from "$lib/api/client";
  import { toasts, startJobMonitoring, triggerDatasetRefresh } from "$lib/stores/app";
  import {
    FORECASTING_MODELS,
    FORECAST_MODEL_LABELS,
    FORECAST_PARAM_SECTIONS,
  } from "$lib/forecast";
  import ForecastColumnRoles from "./ForecastColumnRoles.svelte";

  let { report, onModeSwitched }: { report: DatasetRow; onModeSwitched?: () => void } = $props();

  let reportId = $derived(report.uuid);

  let loading = $state(true);
  let training = $state(false);

  let tsRoles = $state<TsRoleAssignment[]>([]);
  let columnTypes = $state<{ column: string; dataType: string; missing: boolean }[]>([]);
  let mlOptions = $state<Record<string, any>>({});
  let dartsAvailable = $state(false);
  let showRoles = $state(false);

  // Models
  let trainGroup = $state<string[]>([...FORECASTING_MODELS]);

  // Global options
  let testSize = $state(0.2);
  let expandingWindow = $state(false);
  let expandingWindowForecastHorizon = $state(6);
  let expandingWindowStride = $state(6);
  let visualize = $state(true);
  let generateRealPredictions = $state(false);
  let realPredictionLength = $state(12);
  let fillFutureCovariates = $state(true);
  let probabilistic = $state(false);
  let probabilisticLikelihood = $state("quantile");
  let probabilisticNumSamples = $state(100);

  // Model options
  let arimaP = $state(12);
  let arimaD = $state(1);
  let arimaQ = $state(6);
  let lags = $state(12);
  let lagsFuture = $state(0);
  let outputChunkLength = $state(6);
  let epochs = $state(10);

  const timeComponent = $derived(tsRoles.find((r) => r.role === "time-component")?.column ?? "");
  const targets = $derived(tsRoles.filter((r) => r.role === "target").map((r) => r.column));
  const groupingCol = $derived(tsRoles.find((r) => r.role === "grouping")?.column ?? "");
  const pastCovariates = $derived(
    tsRoles.filter((r) => r.role === "past-covariate").map((r) => r.column),
  );
  const futureCovariates = $derived(
    tsRoles.filter((r) => r.role === "future-covariate").map((r) => r.column),
  );
  const staticCovariates = $derived(
    tsRoles.filter((r) => r.role === "static-covariate").map((r) => r.column),
  );

  const showSharedParams = $derived(
    trainGroup.some((m) => FORECAST_PARAM_SECTIONS[m]?.includes("shared")),
  );
  const showArimaParams = $derived(
    trainGroup.some((m) => FORECAST_PARAM_SECTIONS[m]?.includes("arima")),
  );
  const showNnParams = $derived(trainGroup.some((m) => FORECAST_PARAM_SECTIONS[m]?.includes("nn")));

  const hasFutureCovariates = $derived(futureCovariates.length > 0);

  const startDisabledReason = $derived.by(() => {
    if (tsRoles.length === 0) return "Configure time-series columns first";
    if (trainGroup.length === 0) return "Select at least one model";
    if (!dartsAvailable) return "Install the Forecasting add-on first";
    if (trainGroup.includes("tft") && !hasFutureCovariates)
      return "TFT requires a future covariate column";
    return "";
  });

  const slowRun = $derived(
    trainGroup.length > 3 || trainGroup.some((m) => m === "tft" || m === "nlinear"),
  );

  async function loadData() {
    loading = true;
    try {
      const [typesResp, optionsResp] = await Promise.all([
        datasetsApi.columnTypes(reportId),
        jobsApi.mlOptionsForecasting(),
      ]);
      mlOptions = optionsResp;
      columnTypes = Object.entries(typesResp.data_types).map(([column, dataType]) => ({
        column,
        dataType,
        missing: typesResp.missing_values[column] ?? false,
      }));

      const savedRoles = report.column_changes?.ts_roles as TsRoleAssignment[] | undefined;
      if (savedRoles && Array.isArray(savedRoles)) {
        tsRoles = savedRoles.filter((r) => r.role !== "not-included");
      }

      // Defaults from the options endpoint
      testSize = (mlOptions.test_size?.default as number) ?? 0.2;
      expandingWindow = (mlOptions.expanding_window?.default as boolean) ?? false;
      expandingWindowForecastHorizon =
        (mlOptions.expanding_window_forecast_horizon?.default as number) ?? 6;
      expandingWindowStride = (mlOptions.expanding_window_stride?.default as number) ?? 6;
      visualize = (mlOptions.visualize?.default as boolean) ?? true;
      generateRealPredictions = (mlOptions.generate_real_predictions?.default as boolean) ?? false;
      realPredictionLength = (mlOptions.real_prediction_length?.default as number) ?? 12;
      fillFutureCovariates = (mlOptions.fill_future_covariates?.default as boolean) ?? true;
      probabilistic = (mlOptions.probabilistic?.default as boolean) ?? false;
      probabilisticLikelihood =
        (mlOptions.probabilistic_likelihood?.default as any)?.[0] ?? "quantile";
      probabilisticNumSamples = (mlOptions.probabilistic_num_samples?.default as number) ?? 100;
      arimaP = (mlOptions.arima_p?.default as number) ?? 12;
      arimaD = (mlOptions.arima_d?.default as number) ?? 1;
      arimaQ = (mlOptions.arima_q?.default as number) ?? 6;
      lags = (mlOptions.lags?.default as number) ?? 12;
      lagsFuture = (mlOptions.lags_future?.default as number) ?? 0;
      outputChunkLength = (mlOptions.output_chunk_length?.default as number) ?? 6;
      epochs = (mlOptions.epochs?.default as number) ?? 10;

      // Rerun prefill from the previous forecasting job
      try {
        const paramsResp = await datasetsApi.parameters(reportId);
        if (paramsResp.success && paramsResp.args?.mode === "forecasting") {
          const prev = paramsResp.args;
          if (Array.isArray(prev.train_group)) trainGroup = [...prev.train_group];
          if (typeof prev.test_size === "number") testSize = prev.test_size;
          if (typeof prev.expanding_window === "boolean") expandingWindow = prev.expanding_window;
          if (typeof prev.expanding_window_forecast_horizon === "number")
            expandingWindowForecastHorizon = prev.expanding_window_forecast_horizon;
          if (typeof prev.expanding_window_stride === "number")
            expandingWindowStride = prev.expanding_window_stride;
          if (typeof prev.visualize === "boolean") visualize = prev.visualize;
          if (typeof prev.generate_real_predictions === "boolean")
            generateRealPredictions = prev.generate_real_predictions;
          if (typeof prev.real_prediction_length === "number")
            realPredictionLength = prev.real_prediction_length;
          if (typeof prev.fill_future_covariates === "boolean")
            fillFutureCovariates = prev.fill_future_covariates;
          if (typeof prev.probabilistic === "boolean") probabilistic = prev.probabilistic;
          if (typeof prev.probabilistic_likelihood === "string")
            probabilisticLikelihood = prev.probabilistic_likelihood;
          if (typeof prev.probabilistic_num_samples === "number")
            probabilisticNumSamples = prev.probabilistic_num_samples;
          if (typeof prev.arima_p === "number") arimaP = prev.arima_p;
          if (typeof prev.arima_d === "number") arimaD = prev.arima_d;
          if (typeof prev.arima_q === "number") arimaQ = prev.arima_q;
          if (typeof prev.lags === "number") lags = prev.lags;
          if (typeof prev.lags_future === "number") lagsFuture = prev.lags_future;
          if (typeof prev.output_chunk_length === "number")
            outputChunkLength = prev.output_chunk_length;
          if (typeof prev.epochs === "number") epochs = prev.epochs;
        }
      } catch {
        // No previous params — use defaults
      }

      try {
        const resp = await addonsApi.list();
        for (const a of resp.addons) {
          if (a.name === "forecast" && a.installed) dartsAvailable = true;
        }
      } catch {
        // Add-on status is best-effort
      }
      const modulesResp = await addonsApi.checkModules();
      for (const m of modulesResp.modules) {
        if (m.name === "darts" && m.installed) dartsAvailable = true;
      }
    } catch (e) {
      toasts.error(
        `Failed to load forecasting options: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    loadData();
  });

  function toggleModel(model: string) {
    if (trainGroup.includes(model)) {
      trainGroup = trainGroup.filter((m) => m !== model);
    } else {
      trainGroup = [...trainGroup, model];
    }
  }

  function selectAllModels() {
    trainGroup = [...FORECASTING_MODELS];
  }

  function clearModels() {
    trainGroup = [];
  }

  function setExpandingWindow(value: boolean) {
    expandingWindow = value;
    if (value) generateRealPredictions = false;
  }

  function setGeneratePredictions(value: boolean) {
    generateRealPredictions = value;
    if (value) expandingWindow = false;
  }

  function handleRolesComplete(roles: TsRoleAssignment[]) {
    tsRoles = roles;
    showRoles = false;
    onModeSwitched?.();
    if (trainGroup.includes("tft") && !roles.some((r) => r.role === "future-covariate")) {
      trainGroup = trainGroup.filter((m) => m !== "tft");
      toasts.info("TFT model deselected — it requires a future covariate column");
    }
  }

  async function startTraining() {
    if (startDisabledReason) {
      toasts.warning(startDisabledReason);
      return;
    }

    training = true;
    try {
      const options: TrainOption[] = [
        ...trainGroup.map((m) => ({ name: "train_group", value: m })),
        { name: "test_size", value: String(testSize) },
        { name: "expanding_window", value: String(expandingWindow) },
        {
          name: "expanding_window_forecast_horizon",
          value: String(expandingWindowForecastHorizon),
        },
        { name: "expanding_window_stride", value: String(expandingWindowStride) },
        { name: "visualize", value: String(visualize) },
        { name: "generate_real_predictions", value: String(generateRealPredictions) },
        { name: "real_prediction_length", value: String(realPredictionLength) },
        { name: "fill_future_covariates", value: String(fillFutureCovariates) },
        { name: "probabilistic", value: String(probabilistic) },
        { name: "probabilistic_likelihood", value: probabilisticLikelihood },
        { name: "probabilistic_num_samples", value: String(probabilisticNumSamples) },
        { name: "arima_p", value: String(arimaP) },
        { name: "arima_d", value: String(arimaD) },
        { name: "arima_q", value: String(arimaQ) },
        { name: "lags", value: String(lags) },
        { name: "lags_future", value: String(lagsFuture) },
        { name: "output_chunk_length", value: String(outputChunkLength) },
        { name: "epochs", value: String(epochs) },
      ];

      const job = await jobsApi.start(reportId, options, "forecasting");
      toasts.info("Forecasting job submitted...");
      startJobMonitoring(job.id, job);
      triggerDatasetRefresh();
      push(`/results/${reportId}`);
    } catch (e) {
      toasts.error(`Failed to start training: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      training = false;
    }
  }
</script>

{#if loading}
  <div class="text-center py-5">
    <div class="spinner-border" role="status">
      <span class="visually-hidden">Loading...</span>
    </div>
  </div>
{:else}
  {#if !dartsAvailable}
    <div class="alert alert-warning d-flex justify-content-between align-items-center">
      <span>
        The <strong>Forecasting add-on</strong> is not installed — it provides the Darts engine used by
        all forecasting models.
      </span>
      <a href="#/addons" class="btn btn-warning btn-sm">Open Add-ons</a>
    </div>
  {/if}

  <!-- Step 1: Data & time-series configuration -->
  <div class="card mb-3">
    <div class="card-header d-flex justify-content-between align-items-center">
      <h5 class="mb-0"><span class="step-num">1</span> Data &amp; Time-Series Columns</h5>
      <button class="btn btn-outline-primary btn-sm" onclick={() => (showRoles = true)}>
        Configure Columns
      </button>
    </div>
    <div class="card-body">
      {#if tsRoles.length > 0}
        <div class="row g-3">
          <div class="col-md-4">
            <span class="small text-muted d-block mb-1">Time component</span>
            <span class="badge text-bg-light border">{timeComponent}</span>
          </div>
          <div class="col-md-4">
            <span class="small text-muted d-block mb-1">Target series</span>
            {#each targets as t (t)}
              <span class="badge text-bg-primary me-1">{t}</span>
            {/each}
          </div>
          <div class="col-md-4">
            <span class="small text-muted d-block mb-1">Grouping</span>
            {#if groupingCol}
              <span class="badge text-bg-light border">{groupingCol}</span>
            {:else}
              <span class="text-muted small">none — single series</span>
            {/if}
          </div>
          <div class="col-12">
            <details>
              <summary class="small text-muted" style="cursor: pointer;">
                Covariates ({pastCovariates.length +
                  futureCovariates.length +
                  staticCovariates.length})
              </summary>
              <div class="d-flex flex-wrap gap-1 mt-2">
                {#each futureCovariates as c (c)}
                  <span class="badge text-bg-light border">future: {c}</span>
                {/each}
                {#each pastCovariates as c (c)}
                  <span class="badge text-bg-light border">past: {c}</span>
                {/each}
                {#each staticCovariates as c (c)}
                  <span class="badge text-bg-light border">static: {c}</span>
                {/each}
                {#if pastCovariates.length + futureCovariates.length + staticCovariates.length === 0}
                  <span class="text-muted small">No covariates selected.</span>
                {/if}
              </div>
            </details>
          </div>
        </div>
      {:else}
        <p class="text-muted mb-0">
          Assign the time component, target series and any covariates. Click "Configure Columns" to
          begin.
        </p>
      {/if}
    </div>
  </div>

  <!-- Step 2: Model selection -->
  <div class="card mb-3">
    <div class="card-header d-flex justify-content-between align-items-center">
      <h5 class="mb-0"><span class="step-num">2</span> Models</h5>
      <div class="btn-group btn-group-sm" role="group" aria-label="Model selection helpers">
        <button type="button" class="btn btn-outline-secondary" onclick={selectAllModels}>
          Select all
        </button>
        <button type="button" class="btn btn-outline-secondary" onclick={clearModels}>Clear</button>
      </div>
    </div>
    <div class="card-body">
      <p class="text-muted small">
        Select one or more models. Model-specific parameters appear below. A naive baseline model is
        always trained for comparison.
      </p>
      <div class="row row-cols-1 row-cols-md-2 row-cols-xl-3 g-2">
        {#each FORECASTING_MODELS as model (model)}
          <div class="col">
            <label
              class="model-card h-100 {trainGroup.includes(model) ? 'model-card-active' : ''}"
              for={`fc-model-${model}`}
            >
              <input
                type="checkbox"
                class="form-check-input"
                id={`fc-model-${model}`}
                checked={trainGroup.includes(model)}
                onchange={() => toggleModel(model)}
              />
              <span class="form-check-label d-inline">{FORECAST_MODEL_LABELS[model]}</span>
            </label>
          </div>
        {/each}
      </div>
    </div>
  </div>

  <!-- Step 3: Options -->
  <div class="card mb-3">
    <div class="card-header">
      <h5 class="mb-0"><span class="step-num">3</span> Options</h5>
    </div>
    <div class="card-body">
      <h6 class="text-primary">Global Parameters</h6>
      <div class="row g-3">
        <div class="col-md-4">
          <label class="form-label" for="fc-test-size">
            Test Size
            <span
              class="d-block small text-muted"
              title="Input a decimal to indicate a proportion to leave out as a test set, or a whole number to indicate the number of timestamps."
            >
              proportion or number of timestamps
            </span>
          </label>
          <input
            type="number"
            class="form-control form-control-sm"
            id="fc-test-size"
            step="0.01"
            min="0.01"
            bind:value={testSize}
            disabled={expandingWindow || generateRealPredictions}
          />
        </div>
        <div class="col-md-8">
          <div class="form-check form-switch">
            <input
              type="checkbox"
              class="form-check-input"
              role="switch"
              id="fc-expanding"
              checked={expandingWindow}
              onchange={(e) => setExpandingWindow((e.target as HTMLInputElement).checked)}
            />
            <label class="form-check-label" for="fc-expanding">
              Expanding Window Evaluation
              <span
                class="d-block small text-muted"
                title="Rather than evaluating on a single test set, models will be continually evaluated and re-trained over a growing subsection of data, producing predictions for almost all of the time series."
              >
                continually re-train over a growing subsection of data (slower)
              </span>
            </label>
          </div>
        </div>
        {#if expandingWindow}
          <div class="col-md-3">
            <label class="form-label" for="fc-ew-horizon">Forecast Horizon</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-ew-horizon"
              min="1"
              bind:value={expandingWindowForecastHorizon}
            />
            <span class="d-block small text-muted">timesteps forecast at each step</span>
          </div>
          <div class="col-md-3">
            <label class="form-label" for="fc-ew-stride">Stride</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-ew-stride"
              min="1"
              bind:value={expandingWindowStride}
            />
            <span class="d-block small text-muted">timesteps to expand by at each step</span>
          </div>
        {/if}
        <div class="col-md-6">
          <div class="form-check form-switch">
            <input
              type="checkbox"
              class="form-check-input"
              role="switch"
              id="fc-viz"
              bind:checked={visualize}
            />
            <label class="form-check-label" for="fc-viz">
              Create Visualizations
              <span class="d-block small text-muted">forecast charts on the results page</span>
            </label>
          </div>
        </div>
        <div class="col-md-6">
          <div class="form-check form-switch">
            <input
              type="checkbox"
              class="form-check-input"
              role="switch"
              id="fc-real-preds"
              checked={generateRealPredictions}
              onchange={(e) => setGeneratePredictions((e.target as HTMLInputElement).checked)}
            />
            <label class="form-check-label" for="fc-real-preds">
              Generate Future Predictions
              <span
                class="d-block small text-muted"
                title="Train models on the entire dataset and create future predictions rather than holding out a test set for evaluation."
              >
                train on the entire dataset and predict the final timesteps
              </span>
            </label>
          </div>
        </div>
        {#if generateRealPredictions}
          <div class="col-md-4">
            <label class="form-label" for="fc-pred-length">Length of Prediction</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-pred-length"
              min="1"
              bind:value={realPredictionLength}
            />
          </div>
          <div class="col-md-8">
            <div class="form-check form-switch">
              <input
                type="checkbox"
                class="form-check-input"
                role="switch"
                id="fc-fill-cov"
                bind:checked={fillFutureCovariates}
              />
              <label class="form-check-label" for="fc-fill-cov">
                Fill In Future Covariates
                <span
                  class="d-block small text-muted"
                  title="Impute your future covariates into the future to be used with prediction generation. Otherwise, future covariates will be used as past covariates at prediction time."
                >
                  impute future covariates into the future (off = use them as past covariates)
                </span>
              </label>
            </div>
          </div>
        {/if}
        <div class="col-md-6">
          <div class="form-check form-switch">
            <input
              type="checkbox"
              class="form-check-input"
              role="switch"
              id="fc-prob"
              bind:checked={probabilistic}
            />
            <label class="form-check-label" for="fc-prob">
              Use Probabilistic Forecasting
              <span
                class="d-block small text-muted"
                title="Models will produce many predictions using a distribution function for each timestamp and average together, rather than producing a single prediction."
              >
                average many sampled predictions per timestamp
              </span>
            </label>
          </div>
        </div>
        {#if probabilistic}
          <div class="col-md-3">
            <label class="form-label" for="fc-likelihood">Distribution Function</label>
            <select
              id="fc-likelihood"
              class="form-select form-select-sm"
              bind:value={probabilisticLikelihood}
            >
              <option value="quantile">Quantile</option>
              <option value="gaussian">Gaussian</option>
              <option value="laplace">Laplace</option>
            </select>
          </div>
          <div class="col-md-3">
            <label class="form-label" for="fc-samples">Number of Samples</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-samples"
              min="1"
              bind:value={probabilisticNumSamples}
            />
          </div>
        {/if}
      </div>

      <hr />
      <h6 class="text-primary">Model Parameters</h6>
      <p class="text-muted small mb-2">
        Default values are provided, but consider aligning parameter choices with the natural cycles
        in your data — e.g. 12 or 24 for hourly series, 7 or 30 for daily series.
      </p>
      {#if showArimaParams}
        <h6 class="small text-muted mt-2">ARIMA</h6>
        <div class="row g-3">
          <div class="col-md-3">
            <label class="form-label" for="fc-arima-p">Order (p)</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-arima-p"
              min="0"
              bind:value={arimaP}
            />
            <span class="d-block small text-muted"
              >previous timesteps in the autoregressive model</span
            >
          </div>
          <div class="col-md-3">
            <label class="form-label" for="fc-arima-d">Differentiation (d)</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-arima-d"
              min="0"
              bind:value={arimaD}
            />
            <span class="d-block small text-muted">number of times past values are subtracted</span>
          </div>
          <div class="col-md-3">
            <label class="form-label" for="fc-arima-q">Moving Average (q)</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-arima-q"
              min="0"
              bind:value={arimaQ}
            />
            <span class="d-block small text-muted">moving average window size</span>
          </div>
        </div>
      {/if}
      {#if showSharedParams}
        <h6 class="small text-muted mt-2">Shared Parameters</h6>
        <div class="row g-3">
          <div class="col-md-3">
            <label class="form-label" for="fc-lags">Number of Lags</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-lags"
              min="1"
              bind:value={lags}
            />
            <span class="d-block small text-muted">previous timesteps used as model input</span>
          </div>
          <div class="col-md-3">
            <label class="form-label" for="fc-lags-future">Timesteps for Future Covariates</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-lags-future"
              min="0"
              bind:value={lagsFuture}
              disabled={!hasFutureCovariates}
            />
            <span class="d-block small text-muted">
              {hasFutureCovariates
                ? "0 = use covariate at prediction time"
                : "requires a future covariate column"}
            </span>
          </div>
          <div class="col-md-3">
            <label class="form-label" for="fc-ocl">Output Length</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-ocl"
              min="1"
              bind:value={outputChunkLength}
            />
            <span class="d-block small text-muted">timesteps predicted at once by the model</span>
          </div>
        </div>
      {/if}
      {#if showNnParams}
        <h6 class="small text-muted mt-2">Neural Network Parameters</h6>
        <div class="row g-3">
          <div class="col-md-3">
            <label class="form-label" for="fc-epochs">Number of Epochs</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="fc-epochs"
              min="1"
              bind:value={epochs}
            />
            <span class="d-block small text-muted">higher numbers will take longer</span>
          </div>
        </div>
      {/if}
      {#if !showArimaParams && !showSharedParams && !showNnParams}
        <p class="text-muted small mb-0">
          No additional parameters for the selected model{trainGroup.length > 1 ? "s" : ""}.
        </p>
      {/if}
    </div>
  </div>

  <!-- Sticky action bar -->
  <div class="prepare-actionbar">
    <div class="small">
      {#if startDisabledReason}
        <span class="text-warning-emphasis">{startDisabledReason}</span>
      {:else if slowRun}
        <span class="text-warning-emphasis">
          {trainGroup.length} model{trainGroup.length > 1 ? "s" : ""} — this may take a while. You can
          cancel mid-training.
        </span>
      {:else}
        <span class="text-muted">
          Ready: {trainGroup.length} model{trainGroup.length > 1 ? "s" : ""} selected
        </span>
      {/if}
    </div>
    <div class="d-flex gap-2">
      <a href="#/" class="btn btn-secondary">Cancel</a>
      <button
        class="btn btn-primary"
        disabled={training || startDisabledReason !== ""}
        onclick={startTraining}
      >
        {#if training}
          <span class="spinner-border spinner-border-sm me-1"></span>
          Starting...
        {:else}
          Start Training
        {/if}
      </button>
    </div>
  </div>

  {#if showRoles}
    <ForecastColumnRoles
      {reportId}
      columns={columnTypes}
      initialRoles={tsRoles}
      onclose={() => (showRoles = false)}
      oncomplete={handleRolesComplete}
    />
  {/if}
{/if}
