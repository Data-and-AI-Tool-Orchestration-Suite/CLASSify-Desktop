<script lang="ts">
  import { onMount } from "svelte";
  import { push } from "svelte-spa-router";
  import {
    addons as addonsApi,
    datasets as datasetsApi,
    jobs as jobsApi,
    type AddonInfo,
    type DatasetRow,
    type ColumnChange,
    type TrainOption,
  } from "$lib/api/client";
  import { toasts, startJobMonitoring, triggerDatasetRefresh } from "$lib/stores/app";
  import ColumnPreviewModal from "$lib/components/ColumnPreviewModal.svelte";
  import ClassMappingModal from "$lib/components/ClassMappingModal.svelte";

  let { params } = $props<{ params: { reportId?: string } }>();
  let reportId = $derived(params?.reportId ?? "");

  let report = $state<DatasetRow | null>(null);
  let mlOptions = $state<Record<string, any>>({});
  let loading = $state(true);
  let supervised = $state(true);
  let training = $state(false);
  let prefilled = $state(false);

  // Column preview
  let showColumnPreview = $state(false);
  let columnChanges = $state<ColumnChange[]>([]);

  // Class mapping
  let showClassMapping = $state(false);
  let classColumn = $state<string>("");
  let classValues = $state<string[]>([]);

  // ML options state
  let trainGroup = $state<string[]>([]);
  let parameterTune = $state(true);
  let shapFeatureExplainability = $state(true);
  let visualize = $state(true);
  let standardScaler = $state(false);
  let testSize = $state(0.2);
  let nIter = $state(100);
  let folds = $state(5);
  let repeats = $state(1);
  let randomState = $state(42);

  // Unsupervised
  let numClusters = $state(2);

  // Add-on availability (e.g. TabPFN needs the torch-based add-on)
  let addonProvides = $state<Set<string>>(new Set());

  const ALL_SUPERVISED_MODELS = [
    "randomforest",
    "neuralnetwork",
    "tabpfn",
    "xgboost",
    "gradientboosting",
    "histgradientboosting",
    "bagging",
    "sgdclassifier",
    "logisticregression",
    "kneighbors",
  ];
  const UNSUPERVISED_MODELS = ["spectralclustering", "kmeans", "hdbscan"];

  const MODEL_LABELS: Record<string, string> = {
    randomforest: "Random Forest",
    neuralnetwork: "Neural Network",
    tabpfn: "TabPFN",
    xgboost: "XGBoost",
    gradientboosting: "Gradient Boosting",
    histgradientboosting: "Hist Gradient Boosting",
    bagging: "Bagging",
    sgdclassifier: "SGD Classifier",
    logisticregression: "Logistic Regression",
    kneighbors: "K-Neighbors",
    spectralclustering: "Spectral Clustering",
    kmeans: "K-Means",
    hdbscan: "HDBSCAN",
  };

  const MODEL_ADDONS: Record<string, string> = { tabpfn: "tabpfn" };

  function modelAvailable(model: string): boolean {
    const addon = MODEL_ADDONS[model];
    return addon === undefined || addonProvides.has(addon);
  }

  const activeModels = $derived(supervised ? ALL_SUPERVISED_MODELS : UNSUPERVISED_MODELS);

  function setMode(mode: boolean) {
    supervised = mode;
    // Keep the selection consistent when switching learning mode
    const allowed = mode ? ALL_SUPERVISED_MODELS : UNSUPERVISED_MODELS;
    trainGroup = trainGroup.filter((m) => allowed.includes(m) && modelAvailable(m));
  }

  const startDisabledReason = $derived.by(() => {
    if (columnChanges.length === 0) return "Configure columns first";
    if (trainGroup.length === 0) return "Select at least one model";
    return "";
  });

  const selectedCols = $derived(columnChanges.filter((c) => c.checked));

  const featureSummary = $derived.by(() => {
    const counts = { numeric: 0, categorical: 0, bool: 0 };
    for (const c of selectedCols) {
      if (c.is_class) continue;
      if (c.data_type === "float" || c.data_type === "integer") counts.numeric++;
      else if (c.data_type === "categorical" || c.data_type === "string") counts.categorical++;
      else if (c.data_type === "bool") counts.bool++;
    }
    const parts: string[] = [];
    if (counts.numeric) parts.push(`${counts.numeric} numeric`);
    if (counts.categorical) parts.push(`${counts.categorical} categorical`);
    if (counts.bool) parts.push(`${counts.bool} yes/no`);
    return parts.join(" · ");
  });

  let savingTarget = $state(false);

  async function handleTargetChange(event: Event) {
    const newTarget = (event.target as HTMLSelectElement).value;
    if (!newTarget || newTarget === classColumn) return;
    savingTarget = true;
    try {
      const updated = columnChanges.map((c) => ({ ...c, is_class: c.column === newTarget }));
      const resp = await datasetsApi.columnChanges(reportId, updated);
      if (resp.success) {
        columnChanges = updated;
        classColumn = newTarget;
        toasts.success(`Target column set to ${newTarget}`);
        const classChange = updated.find((c) => c.is_class);
        if (classChange && classChange.data_type === "categorical") {
          await openClassMapping();
        }
      } else {
        toasts.error(resp.message || "Failed to update target column");
      }
    } catch (e) {
      toasts.error(`Failed to update target: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      savingTarget = false;
    }
  }

  const slowRun = $derived(
    trainGroup.length > 3 ||
      (parameterTune && trainGroup.length > 1) ||
      trainGroup.includes("neuralnetwork"),
  );

  async function loadData() {
    loading = true;
    try {
      report = await datasetsApi.get(reportId);

      // Load previous training parameters (for rerun)
      let prevArgs: Record<string, any> | null = null;
      try {
        const paramsResp = await datasetsApi.parameters(reportId);
        if (paramsResp.success && paramsResp.args) {
          prevArgs = paramsResp.args;
          if (typeof prevArgs.supervised === "boolean") {
            supervised = prevArgs.supervised;
          }
        }
      } catch {
        // No previous params — use defaults
      }

      const options = supervised
        ? await jobsApi.mlOptionsSupervised()
        : await jobsApi.mlOptionsUnsupervised();
      mlOptions = options;

      // Initialize from defaults
      const models = supervised ? ALL_SUPERVISED_MODELS : UNSUPERVISED_MODELS;
      trainGroup = [...models];
      parameterTune = (mlOptions.parameter_tune?.default as boolean) ?? true;
      shapFeatureExplainability =
        (mlOptions.shap_feature_explainability?.default as boolean) ?? true;
      visualize = (mlOptions.visualize?.default as boolean) ?? true;
      standardScaler = (mlOptions.standard_scaler?.default as boolean) ?? false;
      testSize = (mlOptions.test_size?.default as number) ?? 0.2;
      nIter = (mlOptions.n_iter?.default as number) ?? 100;
      folds = (mlOptions.folds?.default as number) ?? 5;
      repeats = (mlOptions.repeats?.default as number) ?? 1;
      randomState = (mlOptions.random_state?.default as number) ?? 42;
      numClusters = (mlOptions.num_clusters?.default as number) ?? 2;

      // Override with previous training args (rerun)
      if (prevArgs) {
        prefilled = true;
        if (Array.isArray(prevArgs.train_group)) {
          trainGroup = [...prevArgs.train_group];
        }
        if (typeof prevArgs.parameter_tune === "boolean") parameterTune = prevArgs.parameter_tune;
        if (typeof prevArgs.shap_feature_explainability === "boolean")
          shapFeatureExplainability = prevArgs.shap_feature_explainability;
        if (typeof prevArgs.visualize === "boolean") visualize = prevArgs.visualize;
        if (typeof prevArgs.standard_scaler === "boolean")
          standardScaler = prevArgs.standard_scaler;
        if (typeof prevArgs.test_size === "number") testSize = prevArgs.test_size;
        if (typeof prevArgs.n_iter === "number") nIter = prevArgs.n_iter;
        if (typeof prevArgs.folds === "number") folds = prevArgs.folds;
        if (typeof prevArgs.repeats === "number") repeats = prevArgs.repeats;
        if (typeof prevArgs.random_state === "number") randomState = prevArgs.random_state;
        if (typeof prevArgs.num_clusters === "number") numClusters = prevArgs.num_clusters;
        if (typeof prevArgs.class_column === "string") classColumn = prevArgs.class_column;
      }

      // Load add-on availability and prune unavailable models (e.g. TabPFN
      // without its add-on) from the selection
      await loadAddons();
      trainGroup = trainGroup.filter((m) => modelAvailable(m));

      // Load existing column configuration (if previously saved)
      const savedChanges = report.column_changes?.changes as ColumnChange[] | undefined;
      if (savedChanges && Array.isArray(savedChanges) && savedChanges.length > 0) {
        columnChanges = savedChanges;
        const classCol = columnChanges.find((c) => c.is_class);
        if (classCol) classColumn = classCol.column;
      }
    } catch (e) {
      toasts.error("Failed to load dataset");
      push("/");
    } finally {
      loading = false;
    }
  }

  async function loadAddons() {
    try {
      const resp: { addons: AddonInfo[] } = await addonsApi.list();
      const set = new Set<string>();
      for (const a of resp.addons) {
        if (a.installed) {
          for (const p of a.provides) set.add(p);
        }
      }
      addonProvides = set;
    } catch {
      // Add-on status is best-effort; treat everything as unavailable
    }
  }

  onMount(() => {
    loadData();
    loadAddons();
  });

  function toggleModel(model: string) {
    if (!modelAvailable(model)) return;
    if (trainGroup.includes(model)) {
      trainGroup = trainGroup.filter((m) => m !== model);
    } else {
      trainGroup = [...trainGroup, model];
    }
  }

  function selectAllModels() {
    trainGroup = activeModels.filter(modelAvailable);
  }

  function clearModels() {
    trainGroup = [];
  }

  function handleColumnChangesComplete(changes: ColumnChange[], classCol: string) {
    columnChanges = changes;
    classColumn = classCol;
    showColumnPreview = false;

    // Check if class column is categorical — needs mapping
    const classChange = changes.find((c) => c.is_class);
    if (classChange && classChange.data_type === "categorical") {
      openClassMapping();
    } else {
      toasts.success("Column configuration saved");
    }
  }

  async function openClassMapping() {
    if (!classColumn) return;
    try {
      const resp = await datasetsApi.classValues(reportId, classColumn);
      classValues = resp.class_values;
      showClassMapping = true;
    } catch (e) {
      toasts.error("Failed to get class values");
    }
  }

  function handleClassMappingComplete() {
    showClassMapping = false;
    toasts.success("Class mapping applied");
  }

  async function startTraining() {
    if (trainGroup.length === 0) {
      toasts.warning("Select at least one model to train");
      return;
    }

    training = true;
    try {
      const options: TrainOption[] = [
        { name: "supervised", value: String(supervised) },
        { name: "train_group", value: trainGroup[0] },
        ...trainGroup.slice(1).map((m) => ({ name: "train_group", value: m })),
        { name: "parameter_tune", value: String(parameterTune) },
        { name: "shap_feature_explainability", value: String(shapFeatureExplainability) },
        { name: "visualize", value: String(visualize) },
        { name: "standard_scaler", value: String(standardScaler) },
        { name: "test_size", value: String(testSize) },
        { name: "n_iter", value: String(nIter) },
        { name: "folds", value: String(folds) },
        { name: "repeats", value: String(repeats) },
        { name: "random_state", value: String(randomState) },
        { name: "class_column", value: classColumn || "class" },
      ];

      if (!supervised) {
        options.push({ name: "num_clusters", value: String(numClusters) });
      }

      const job = await jobsApi.start(reportId, options);
      toasts.info("Training job submitted...");
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
{:else if report}
  <div
    class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pb-2 mb-3 border-bottom"
  >
    <h1 class="h4 mb-0">
      Prepare: {report.filename}
      {#if prefilled}
        <span
          class="badge bg-info-subtle text-info-emphasis ms-2"
          title="Settings were loaded from the previous run"
        >
          settings from last run
        </span>
      {/if}
    </h1>
    <a href="#/" class="btn btn-outline-secondary btn-sm">Back</a>
  </div>

  <!-- Step 1: Data & target -->
  <div class="card mb-3">
    <div class="card-header d-flex justify-content-between align-items-center">
      <h5 class="mb-0">
        <span class="step-num">1</span> Data &amp; Target
      </h5>
      <button class="btn btn-outline-primary btn-sm" onclick={() => (showColumnPreview = true)}>
        Configure Columns
      </button>
    </div>
    <div class="card-body">
      {#if selectedCols.length > 0}
        <div class="row g-3">
          <div class="col-md-7">
            <div class="small text-muted mb-1">Feature columns</div>
            <p class="mb-2">
              <strong>{selectedCols.length - 1}</strong> features selected
              {#if featureSummary}
                <span class="text-muted">({featureSummary})</span>
              {/if}
            </p>
            <details>
              <summary class="small text-muted" style="cursor: pointer;">
                Show column list
              </summary>
              <div class="d-flex flex-wrap gap-1 mt-2">
                {#each selectedCols.filter((c) => !c.is_class) as col (col.column)}
                  <span class="badge text-bg-light border">
                    {col.column}
                    <span class="opacity-75 fw-normal">{col.data_type}</span>
                  </span>
                {/each}
              </div>
            </details>
          </div>
          <div class="col-md-5 border-md-start ps-md-4">
            <label class="small text-muted mb-1" for="target-select">Prediction target</label>
            <select
              id="target-select"
              class="form-select"
              value={classColumn}
              onchange={handleTargetChange}
              disabled={savingTarget}
            >
              {#each selectedCols as col (col.column)}
                <option value={col.column} selected={col.column === classColumn}>
                  {col.column} ({col.data_type})
                </option>
              {/each}
            </select>
            <div class="form-text mb-0">
              The column the models will predict. Change it here or via Configure Columns.
            </div>
          </div>
        </div>
      {:else}
        <p class="text-muted mb-0">
          Choose which columns to include and which one is the prediction target. Click "Configure
          Columns" to begin.
        </p>
      {/if}
    </div>
  </div>

  <!-- Step 2: Mode & models -->
  <div class="card mb-3">
    <div class="card-header">
      <h5 class="mb-0"><span class="step-num">2</span> Learning Mode &amp; Models</h5>
    </div>
    <div class="card-body">
      <div class="d-flex flex-wrap justify-content-between align-items-center mb-3 gap-2">
        <div class="btn-group" role="group" aria-label="Learning mode">
          <button
            type="button"
            class="btn {supervised ? 'btn-primary' : 'btn-outline-primary'}"
            onclick={() => setMode(true)}
          >
            Supervised
            <span class="d-block small opacity-75">predict a target column</span>
          </button>
          <button
            type="button"
            class="btn {supervised ? 'btn-outline-primary' : 'btn-primary'}"
            onclick={() => setMode(false)}
          >
            Unsupervised
            <span class="d-block small opacity-75">find clusters</span>
          </button>
        </div>
        <div class="btn-group btn-group-sm" role="group" aria-label="Model selection helpers">
          <button type="button" class="btn btn-outline-secondary" onclick={selectAllModels}>
            Select all
          </button>
          <button type="button" class="btn btn-outline-secondary" onclick={clearModels}>
            Clear
          </button>
        </div>
      </div>

      <div class="row row-cols-1 row-cols-md-2 row-cols-xl-3 g-2">
        {#each activeModels as model (model)}
          {@const available = modelAvailable(model)}
          <div class="col">
            <label
              class="model-card h-100 {trainGroup.includes(model)
                ? 'model-card-active'
                : ''} {available ? '' : 'model-card-disabled'}"
              for={`model-${model}`}
            >
              <input
                type="checkbox"
                class="form-check-input"
                id={`model-${model}`}
                checked={trainGroup.includes(model)}
                disabled={!available}
                onchange={() => toggleModel(model)}
              />
              <span class="form-check-label d-inline">
                {MODEL_LABELS[model]}
                {#if !available}
                  <span class="d-block small text-muted">add-on not installed</span>
                {/if}
              </span>
            </label>
          </div>
        {/each}
      </div>

      {#if !supervised}
        <div class="mt-3 d-flex align-items-center gap-2">
          <label class="form-label mb-0" for="num-clusters">Number of clusters</label>
          <input
            type="number"
            class="form-control"
            id="num-clusters"
            style="max-width: 120px;"
            bind:value={numClusters}
            min="2"
          />
          <span class="text-muted small">used when parameter tuning is off</span>
        </div>
      {/if}
    </div>
  </div>

  <!-- Step 3: Options -->
  <div class="card mb-3">
    <div class="card-header">
      <h5 class="mb-0"><span class="step-num">3</span> Options</h5>
    </div>
    <div class="card-body">
      <div class="row g-3">
        {#if supervised}
          <div class="col-md-6">
            <div class="form-check form-switch">
              <input
                type="checkbox"
                class="form-check-input"
                role="switch"
                id="parameter-tune"
                bind:checked={parameterTune}
              />
              <label class="form-check-label" for="parameter-tune">
                Parameter tuning
                <span class="d-block small text-muted">Optuna hyperparameter search — slower</span>
              </label>
            </div>
          </div>
          {#if parameterTune}
            <div class="col-md-6 d-flex align-items-center">
              <label class="form-label mb-0 me-2" for="n-iter">Tuning iterations</label>
              <input
                type="number"
                class="form-control"
                id="n-iter"
                style="max-width: 120px;"
                min="1"
                max="1000"
                bind:value={nIter}
              />
            </div>
          {/if}
          <div class="col-md-6">
            <div class="form-check form-switch">
              <input
                type="checkbox"
                class="form-check-input"
                role="switch"
                id="shap"
                bind:checked={shapFeatureExplainability}
              />
              <label class="form-check-label" for="shap">
                SHAP feature explainability
                <span class="d-block small text-muted">per-prediction insights</span>
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
              id="visualize"
              bind:checked={visualize}
            />
            <label class="form-check-label" for="visualize">
              Generate visualizations
              <span class="d-block small text-muted">charts on the results page</span>
            </label>
          </div>
        </div>
      </div>

      <details class="mt-3">
        <summary class="text-muted" style="cursor: pointer;">Advanced settings</summary>
        <div class="row g-3 mt-1">
          <div class="col-md-3">
            <div class="form-check form-switch">
              <input
                type="checkbox"
                class="form-check-input"
                role="switch"
                id="standard-scaler"
                bind:checked={standardScaler}
              />
              <label class="form-check-label" for="standard-scaler">Standard scaler</label>
              <span class="d-block small text-muted">instead of MinMax</span>
            </div>
          </div>
          {#if supervised}
            <div class="col-md-3">
              <label class="form-label" for="test-size">Test size</label>
              <input
                type="number"
                class="form-control form-control-sm"
                id="test-size"
                step="0.05"
                min="0.1"
                max="0.5"
                bind:value={testSize}
              />
            </div>
            <div class="col-md-3">
              <label class="form-label" for="folds">CV folds</label>
              <input
                type="number"
                class="form-control form-control-sm"
                id="folds"
                min="2"
                max="20"
                bind:value={folds}
              />
            </div>
            <div class="col-md-3">
              <label class="form-label" for="repeats">CV repeats</label>
              <input
                type="number"
                class="form-control form-control-sm"
                id="repeats"
                min="1"
                max="10"
                bind:value={repeats}
              />
            </div>
          {/if}
          <div class="col-md-3">
            <label class="form-label" for="random-state">Random state</label>
            <input
              type="number"
              class="form-control form-control-sm"
              id="random-state"
              bind:value={randomState}
            />
          </div>
        </div>
      </details>
    </div>
  </div>

  <!-- Sticky action bar -->
  <div class="prepare-actionbar">
    <div class="small">
      {#if startDisabledReason}
        <span class="text-warning-emphasis">{startDisabledReason}</span>
      {:else if slowRun}
        <span class="text-warning-emphasis">
          {trainGroup.length} model{trainGroup.length > 1 ? "s" : ""}{parameterTune
            ? " with tuning"
            : ""} — this may take a while. You can cancel mid-training.
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

  {#if showColumnPreview}
    <ColumnPreviewModal
      {reportId}
      onclose={() => (showColumnPreview = false)}
      oncomplete={handleColumnChangesComplete}
    />
  {/if}

  {#if showClassMapping}
    <ClassMappingModal
      {reportId}
      {classColumn}
      {classValues}
      onclose={() => (showClassMapping = false)}
      oncomplete={handleClassMappingComplete}
    />
  {/if}
{/if}
