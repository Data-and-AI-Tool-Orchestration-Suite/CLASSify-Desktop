<script lang="ts">
  /** Generate-future-forecasts modal (horizon, models, future covariates). */
  import { toasts, startJobMonitoring, triggerDatasetRefresh } from "$lib/stores/app";
  import { jobs as jobsApi, type JobResponse } from "$lib/api/client";
  import { modelLabel } from "$lib/forecast";

  let {
    reportId,
    trainedModels,
    onclose,
    onstarted,
  }: {
    reportId: string;
    trainedModels: string[];
    onclose: () => void;
    onstarted: (job: JobResponse) => void;
  } = $props();

  let horizon = $state(12);
  let selected = $state<string[]>([]);
  let covariatesFile = $state<File | null>(null);
  let submitting = $state(false);
  let seeded = false;

  $effect(() => {
    if (!seeded && trainedModels.length > 0) {
      seeded = true;
      selected = [...trainedModels];
    }
  });

  function toggle(model: string) {
    if (selected.includes(model)) {
      selected = selected.filter((m) => m !== model);
    } else {
      selected = [...selected, model];
    }
  }

  async function submit() {
    if (!horizon || horizon < 1) {
      toasts.error("You must provide a valid number for the forecast horizon value.");
      return;
    }
    if (selected.length === 0) {
      toasts.error("You must select at least one model to perform forecasting.");
      return;
    }
    submitting = true;
    try {
      const job = await jobsApi.generateForecasts(reportId, {
        horizon,
        models: selected,
        covariatesFile,
      });
      toasts.info("Forecasts are being generated. Check back in a couple of minutes.");
      startJobMonitoring(job.id, job);
      triggerDatasetRefresh();
      onstarted(job);
    } catch (e) {
      toasts.error(
        `Failed to start forecast generation: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      submitting = false;
    }
  }
</script>

<div
  class="modal-backdrop fade show"
  onclick={onclose}
  onkeydown={(e) => e.key === "Escape" && onclose()}
  role="presentation"
></div>
<div class="modal fade show d-block" tabindex="-1" role="dialog">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Generate New Forecasts</h5>
        <button type="button" class="btn-close" onclick={onclose} aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <div class="mb-3">
          <label class="form-label" for="gf-horizon">Forecast Horizon</label>
          <input type="number" class="form-control" id="gf-horizon" min="1" bind:value={horizon} />
          <div class="form-text">The number of future timesteps you want the model to predict.</div>
        </div>
        <div class="mb-3">
          <span class="form-label d-block">Forecasting Model(s)</span>
          <div class="d-flex flex-column gap-1">
            {#each trainedModels as model (model)}
              <div class="form-check mb-0">
                <input
                  class="form-check-input"
                  type="checkbox"
                  id={`gf-${model}`}
                  checked={selected.includes(model)}
                  onchange={() => toggle(model)}
                />
                <label class="form-check-label" for={`gf-${model}`}>{modelLabel(model)}</label>
              </div>
            {/each}
          </div>
          <div class="form-text">Select one or more models to use for generating the forecast.</div>
        </div>
        <div class="mb-1">
          <label class="form-label" for="gf-cov">Future Covariate Data (Optional)</label>
          <input
            class="form-control"
            type="file"
            id="gf-cov"
            accept=".csv"
            onchange={(e) => (covariatesFile = (e.target as HTMLInputElement).files?.[0] ?? null)}
          />
          <div class="form-text">
            Upload a .csv file containing future values for any covariate features used during
            training. The file must contain a date column and columns for each required covariate.
          </div>
        </div>
        <p class="text-muted small mb-0">
          Models without a saved forecast model (e.g. the baseline, or models trained on multiple
          groupings) will be skipped.
        </p>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" onclick={onclose}>Cancel</button>
        <button type="button" class="btn btn-primary" disabled={submitting} onclick={submit}>
          {#if submitting}
            <span class="spinner-border spinner-border-sm me-1"></span>
            Starting...
          {:else}
            Generate Forecasts
          {/if}
        </button>
      </div>
    </div>
  </div>
</div>
