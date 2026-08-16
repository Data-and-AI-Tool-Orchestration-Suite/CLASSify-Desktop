<script lang="ts">
  import { onMount } from "svelte";
  import { push } from "svelte-spa-router";
  import {
    datasets as datasetsApi,
    jobs as jobsApi,
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
        if (Array.isArray(prevArgs.train_group)) {
          trainGroup = [...prevArgs.train_group];
        }
        if (typeof prevArgs.parameter_tune === "boolean") parameterTune = prevArgs.parameter_tune;
        if (typeof prevArgs.shap_feature_explainability === "boolean")
          shapFeatureExplainability = prevArgs.shap_feature_explainability;
        if (typeof prevArgs.visualize === "boolean") visualize = prevArgs.visualize;
        if (typeof prevArgs.standard_scaler === "boolean") standardScaler = prevArgs.standard_scaler;
        if (typeof prevArgs.test_size === "number") testSize = prevArgs.test_size;
        if (typeof prevArgs.n_iter === "number") nIter = prevArgs.n_iter;
        if (typeof prevArgs.folds === "number") folds = prevArgs.folds;
        if (typeof prevArgs.repeats === "number") repeats = prevArgs.repeats;
        if (typeof prevArgs.random_state === "number") randomState = prevArgs.random_state;
        if (typeof prevArgs.num_clusters === "number") numClusters = prevArgs.num_clusters;
        if (typeof prevArgs.class_column === "string") classColumn = prevArgs.class_column;
      }

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

  onMount(() => {
    loadData();
  });

  function toggleModel(model: string) {
    if (trainGroup.includes(model)) {
      trainGroup = trainGroup.filter((m) => m !== model);
    } else {
      trainGroup = [...trainGroup, model];
    }
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
    <h1 class="h4">Prepare: {report.filename}</h1>
    <a href="#/" class="btn btn-outline-secondary btn-sm">Back</a>
  </div>

  <!-- Column Configuration -->
  <div class="card mb-4">
    <div class="card-header d-flex justify-content-between align-items-center">
      <h5 class="mb-0">Column Configuration</h5>
      <button class="btn btn-outline-primary btn-sm" onclick={() => (showColumnPreview = true)}>
        Configure Columns
      </button>
    </div>
    <div class="card-body">
      {#if columnChanges.length > 0}
        <p class="text-muted mb-2">
          {columnChanges.filter((c) => c.checked).length} columns selected, class column:
          <strong>{classColumn || "not set"}</strong>
        </p>
        <div class="d-flex flex-wrap gap-1">
          {#each columnChanges.filter((c) => c.checked) as col}
            <span class="badge bg-{col.is_class ? 'primary' : 'secondary'}">
              {col.column} ({col.data_type})
            </span>
          {/each}
        </div>
      {:else}
        <p class="text-muted mb-0">
          No column configuration yet. Click "Configure Columns" to begin.
        </p>
      {/if}
    </div>
  </div>

  <!-- ML Options -->
  <div class="card mb-4">
    <div class="card-header d-flex justify-content-between align-items-center">
      <h5 class="mb-0">Training Options</h5>
      {#if trainGroup.length > 0}
        <span class="text-muted small">
          {trainGroup.length} model{trainGroup.length > 1 ? "s" : ""} selected
          {#if parameterTune}
            · tuning enabled (slower)
          {/if}
        </span>
      {/if}
    </div>
    <div class="card-body">
      <!-- Supervised/Unsupervised toggle -->
      <div class="mb-3">
        <div class="btn-group" role="group">
          <input
            type="radio"
            class="btn-check"
            name="mode"
            id="supervised"
            bind:group={supervised}
            value={true}
          />
          <label class="btn btn-outline-primary" for="supervised">Supervised</label>
          <input
            type="radio"
            class="btn-check"
            name="mode"
            id="unsupervised"
            bind:group={supervised}
            value={false}
          />
          <label class="btn btn-outline-primary" for="unsupervised">Unsupervised</label>
        </div>
      </div>

      <!-- Model selection -->
      <div class="mb-3">
        <label class="form-label fw-bold" for="model-select">Models to Train</label>
        <div class="d-flex flex-wrap gap-2">
          {#each supervised ? ALL_SUPERVISED_MODELS : UNSUPERVISED_MODELS as model}
            <div class="form-check">
              <input
                type="checkbox"
                class="form-check-input"
                id={`model-${model}`}
                checked={trainGroup.includes(model)}
                onchange={() => toggleModel(model)}
              />
              <label class="form-check-label" for={`model-${model}`}>{model}</label>
            </div>
          {/each}
        </div>
      </div>

      {#if !supervised}
        <div class="mb-3">
          <label class="form-label" for="num-clusters">Number of Clusters</label>
          <input
            type="number"
            class="form-control"
            id="num-clusters"
            style="max-width: 120px;"
            bind:value={numClusters}
            min="2"
          />
        </div>
      {/if}

      <!-- General options -->
      <div class="row g-3">
        {#if supervised}
          <div class="col-md-6">
            <div class="form-check">
              <input
                type="checkbox"
                class="form-check-input"
                id="parameter-tune"
                bind:checked={parameterTune}
              />
              <label class="form-check-label" for="parameter-tune">Parameter Tuning (Optuna)</label>
            </div>
          </div>
          <div class="col-md-6">
            <div class="form-check">
              <input
                type="checkbox"
                class="form-check-input"
                id="shap"
                bind:checked={shapFeatureExplainability}
              />
              <label class="form-check-label" for="shap">SHAP Feature Explainability</label>
            </div>
          </div>
        {/if}
        <div class="col-md-6">
          <div class="form-check">
            <input
              type="checkbox"
              class="form-check-input"
              id="visualize"
              bind:checked={visualize}
            />
            <label class="form-check-label" for="visualize">Generate Visualizations</label>
          </div>
        </div>
        <div class="col-md-6">
          <div class="form-check">
            <input
              type="checkbox"
              class="form-check-input"
              id="standard-scaler"
              bind:checked={standardScaler}
            />
            <label class="form-check-label" for="standard-scaler">Standard Scaler (vs MinMax)</label
            >
          </div>
        </div>

        {#if supervised}
          <div class="col-md-3">
            <label class="form-label" for="test-size">Test Size</label>
            <input
              type="number"
              class="form-control"
              id="test-size"
              step="0.05"
              min="0.1"
              max="0.5"
              bind:value={testSize}
            />
          </div>
          <div class="col-md-3">
            <label class="form-label" for="folds">CV Folds</label>
            <input
              type="number"
              class="form-control"
              id="folds"
              min="2"
              max="20"
              bind:value={folds}
            />
          </div>
          <div class="col-md-3">
            <label class="form-label" for="repeats">CV Repeats</label>
            <input
              type="number"
              class="form-control"
              id="repeats"
              min="1"
              max="10"
              bind:value={repeats}
            />
          </div>
        {/if}
        <div class="col-md-3">
          <label class="form-label" for="n-iter">Tuning Iterations</label>
          <input
            type="number"
            class="form-control"
            id="n-iter"
            min="1"
            max="1000"
            bind:value={nIter}
          />
        </div>
        <div class="col-md-3">
          <label class="form-label" for="random-state">Random State</label>
          <input type="number" class="form-control" id="random-state" bind:value={randomState} />
        </div>
      </div>
    </div>
  </div>

  <!-- Training estimate warning -->
  {#if trainGroup.length > 3 || (parameterTune && trainGroup.length > 1)}
    <div class="alert alert-warning d-flex align-items-center gap-2">
      <span>⏱️</span>
      <div class="small">
        Training {trainGroup.length} models{parameterTune ? " with parameter tuning" : ""} may take several
        minutes. You can cancel mid-training from the results page.
        {#if trainGroup.includes("neuralnetwork") || trainGroup.includes("tabpfn")}
          Neural networks and TabPFN are especially compute-intensive on large datasets.
        {/if}
      </div>
    </div>
  {/if}

  <!-- Train button -->
  <div class="d-flex justify-content-end gap-2">
    <a href="#/" class="btn btn-secondary">Cancel</a>
    <button
      class="btn btn-primary btn-lg"
      disabled={training || columnChanges.length === 0}
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
