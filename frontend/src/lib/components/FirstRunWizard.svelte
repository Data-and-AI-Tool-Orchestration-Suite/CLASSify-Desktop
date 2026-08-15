<script lang="ts">
  import { onMount } from "svelte";
  import { system } from "$lib/api/client";
  import { toasts } from "$lib/stores/app";

  let { oncomplete } = $props<{ oncomplete: () => void }>();

  let step = $state(0);
  let loading = $state(true);
  let wizardState = $state<{
    first_run: boolean;
    data_dir: string;
    disk_free_bytes: number;
    cpu_count: number;
    has_addons: { tabpfn: boolean; sdv: boolean };
  } | null>(null);

  const steps = ["Welcome", "System Check", "Add-ons", "Ready"];

  async function loadState() {
    loading = true;
    try {
      wizardState = await system.firstRun();
      if (!wizardState.first_run) {
        oncomplete();
      }
    } catch {
      oncomplete();
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    loadState();
  });

  async function finish() {
    try {
      await system.completeFirstRun();
      toasts.success("CLASSify Desktop is ready!");
    } catch {
      // Non-fatal
    }
    oncomplete();
  }

  function formatBytes(bytes: number): string {
    if (bytes === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
  }

  function next() {
    if (step < steps.length - 1) {
      step++;
    } else {
      finish();
    }
  }

  function skip() {
    finish();
  }
</script>

{#if loading}
  <div class="text-center py-5">
    <div class="spinner-border" role="status"></div>
  </div>
{:else if wizardState}
  <div
    class="d-flex align-items-center justify-content-center"
    style="min-height: calc(100vh - 56px);"
  >
    <div class="card shadow" style="max-width: 600px; width: 100%;">
      <div class="card-body p-4">
        <!-- Progress dots -->
        <div class="d-flex justify-content-center mb-4 gap-2">
          {#each steps as s, i}
            <div
              class="rounded-circle"
              style="width: 10px; height: 10px; background: {i <= step ? '#0d6efd' : '#dee2e6'};"
              title={s}
            ></div>
          {/each}
        </div>

        {#if step === 0}
          <div class="text-center">
            <h2 class="mb-3">Welcome to CLASSify Desktop</h2>
            <p class="text-muted">
              A local-first machine learning tool for training classification and clustering models
              on tabular data. All data stays on your machine.
            </p>
            <div class="alert alert-info text-start small">
              <strong>What's different from CLASSify web:</strong>
              <ul class="mb-0 mt-1">
                <li>No server required — everything runs locally</li>
                <li>No S3, ClearML, or database server — local files + SQLite</li>
                <li>No login — single-user, your data only</li>
                <li>Optional add-ons for TabPFN and SDV (synthetic data)</li>
              </ul>
            </div>
          </div>
        {:else if step === 1}
          <div>
            <h4 class="mb-3">System Check</h4>
            <table class="table table-sm">
              <tbody>
                <tr>
                  <td>Data Directory</td>
                  <td class="text-end"><code>{wizardState.data_dir}</code></td>
                </tr>
                <tr>
                  <td>Free Disk Space</td>
                  <td class="text-end">
                    <span
                      class={wizardState.disk_free_bytes > 500_000_000
                        ? "text-success"
                        : "text-danger"}
                    >
                      {formatBytes(wizardState.disk_free_bytes)}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td>CPU Cores</td>
                  <td class="text-end">{wizardState.cpu_count}</td>
                </tr>
                <tr>
                  <td>TabPFN Add-on</td>
                  <td class="text-end">
                    {#if wizardState.has_addons.tabpfn}
                      <span class="badge bg-success">Installed</span>
                    {:else}
                      <span class="badge bg-secondary">Not installed</span>
                    {/if}
                  </td>
                </tr>
                <tr>
                  <td>SDV Add-on</td>
                  <td class="text-end">
                    {#if wizardState.has_addons.sdv}
                      <span class="badge bg-success">Installed</span>
                    {:else}
                      <span class="badge bg-secondary">Not installed</span>
                    {/if}
                  </td>
                </tr>
              </tbody>
            </table>
            {#if wizardState.disk_free_bytes < 500_000_000}
              <div class="alert alert-warning small">
                Low disk space. Training models and storing datasets requires at least 500 MB free.
              </div>
            {/if}
          </div>
        {:else if step === 2}
          <div>
            <h4 class="mb-3">Optional Add-ons</h4>
            <p class="text-muted small">
              Add-ons provide TabPFN and SDV (synthetic data) models. They require torch (~2GB
              download) and can be installed later from the Add-ons page.
            </p>
            <div class="d-flex gap-3 mb-3">
              <div class="card flex-grow-1">
                <div class="card-body">
                  <h6 class="card-title">TabPFN</h6>
                  <p class="card-text small text-muted mb-0">
                    Prior-Data Fitted Networks — powerful for small tabular datasets.
                  </p>
                  {#if wizardState.has_addons.tabpfn}
                    <span class="badge bg-success mt-2">Installed</span>
                  {:else}
                    <span class="badge bg-secondary mt-2">Install later</span>
                  {/if}
                </div>
              </div>
              <div class="card flex-grow-1">
                <div class="card-body">
                  <h6 class="card-title">SDV</h6>
                  <p class="card-text small text-muted mb-0">
                    Synthetic Data Vault — generate synthetic training data.
                  </p>
                  {#if wizardState.has_addons.sdv}
                    <span class="badge bg-success mt-2">Installed</span>
                  {:else}
                    <span class="badge bg-secondary mt-2">Install later</span>
                  {/if}
                </div>
              </div>
            </div>
            <p class="text-muted small">You can install these anytime from Settings → Add-ons.</p>
          </div>
        {:else if step === 3}
          <div class="text-center">
            <h2 class="mb-3 text-success">You're all set!</h2>
            <p class="text-muted">
              Upload a CSV file to get started. CLASSify will auto-detect column types and guide you
              through model selection.
            </p>
            <div class="alert alert-light text-start small">
              <strong>Quick start:</strong>
              <ol class="mb-0 mt-1">
                <li>Click "Upload CSV" on the home page</li>
                <li>Configure columns and select a class column</li>
                <li>Choose models and training options</li>
                <li>Click "Start Training"</li>
                <li>Explore results in the Results tab</li>
              </ol>
            </div>
          </div>
        {/if}

        <!-- Navigation -->
        <div class="d-flex justify-content-between mt-4">
          <button class="btn btn-link text-muted" onclick={skip}> Skip wizard </button>
          <button class="btn btn-primary" onclick={next}>
            {step === steps.length - 1 ? "Get Started" : "Next"}
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}
