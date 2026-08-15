<script lang="ts">
  import { onMount } from "svelte";
  import { addons as addonsApi, type AddonInfo } from "$lib/api/client";
  import { toasts } from "$lib/stores/app";

  let addons = $state<AddonInfo[]>([]);
  let loading = $state(true);
  let installing = $state<string | null>(null);

  async function loadAddons() {
    loading = true;
    try {
      const resp = await addonsApi.list();
      addons = resp.addons;
    } catch {
      toasts.error("Failed to load add-ons");
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    loadAddons();
  });

  async function handleInstall(name: string) {
    if (!confirm(`Install ${name}? This will download ~2GB of dependencies (torch).`)) return;
    installing = name;
    toasts.info(`Installing ${name}... This may take several minutes.`);
    try {
      const result = await addonsApi.install(name);
      if (result.success) {
        toasts.success(result.message);
      } else {
        toasts.error(result.message);
      }
      await loadAddons();
    } catch (e) {
      toasts.error(`Installation failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      installing = null;
    }
  }

  async function handleUninstall(name: string) {
    if (!confirm(`Uninstall ${name}?`)) return;
    try {
      const result = await addonsApi.uninstall(name);
      if (result.success) {
        toasts.success(result.message);
      } else {
        toasts.error(result.message);
      }
      await loadAddons();
    } catch (e) {
      toasts.error(`Uninstall failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
</script>

<div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pb-2 mb-3 border-bottom">
  <h1 class="h4">Add-ons</h1>
</div>

<div class="alert alert-info">
  <strong>About Add-ons</strong><br />
  Add-ons provide optional ML capabilities that require large dependencies (torch ~2GB).
  They are not included in the base installer. Install them on demand — all data stays local.
</div>

{#if loading}
  <div class="text-center py-5">
    <div class="spinner-border" role="status"></div>
  </div>
{:else if addons.length === 0}
  <p class="text-muted">No add-ons available.</p>
{:else}
  <div class="row g-4">
    {#each addons as addon (addon.name)}
      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-body">
            <div class="d-flex justify-content-between align-items-start mb-2">
              <div>
                <h5 class="card-title mb-0">{addon.name}</h5>
                <span class="text-muted small">v{addon.version}</span>
              </div>
              {#if addon.installed}
                <span class="badge bg-success">Installed</span>
              {:else}
                <span class="badge bg-secondary">Not Installed</span>
              {/if}
            </div>

            <p class="card-text">{addon.description}</p>

            <dl class="row mb-3 small">
              <dt class="col-sm-5">Download Size</dt>
              <dd class="col-sm-7">~{addon.size_estimate_mb} MB</dd>
              <dt class="col-sm-5">Provides</dt>
              <dd class="col-sm-7">{addon.provides.join(", ")}</dd>
            </dl>

            <div class="d-flex gap-2">
              {#if addon.installed}
                <button
                  class="btn btn-outline-danger btn-sm"
                  onclick={() => handleUninstall(addon.name)}
                >
                  Uninstall
                </button>
              {:else}
                <button
                  class="btn btn-primary btn-sm"
                  disabled={installing === addon.name}
                  onclick={() => handleInstall(addon.name)}
                >
                  {#if installing === addon.name}
                    <span class="spinner-border spinner-border-sm me-1"></span>
                    Installing...
                  {:else}
                    Install
                  {/if}
                </button>
              {/if}
            </div>
          </div>
        </div>
      </div>
    {/each}
  </div>
{/if}
