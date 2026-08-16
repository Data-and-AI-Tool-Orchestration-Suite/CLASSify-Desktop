<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { addons as addonsApi, type AddonInfo } from "$lib/api/client";
  import { toasts } from "$lib/stores/app";

  let addons = $state<AddonInfo[]>([]);
  let loading = $state(true);
  let installing = $state<string | null>(null);
  let installProgress = $state<string[]>([]);
  let installError = $state<string | null>(null);
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  async function loadAddons() {
    loading = true;
    try {
      const resp = await addonsApi.list();
      addons = resp.addons;

      // Check if any addon is currently installing/queued (resume after navigation)
      if (!installing) {
        for (const addon of resp.addons) {
          try {
            const status = await addonsApi.installStatus(addon.name);
            if (status.state === "installing" || status.state === "queued") {
              resumePolling(addon.name, status.progress);
              break;
            }
          } catch {
            // Ignore
          }
        }
      }
    } catch {
      toasts.error("Failed to load add-ons");
    } finally {
      loading = false;
    }
  }

  function resumePolling(name: string, initialProgress: string[]) {
    installing = name;
    installProgress = [...initialProgress];
    installError = null;
    startPolling(name);
  }

  function startPolling(name: string) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const status = await addonsApi.installStatus(name);
        installProgress = [...status.progress];
        if (status.state === "succeeded") {
          if (pollTimer) clearInterval(pollTimer);
          pollTimer = null;
          installing = null;
          toasts.success(`${name} installed successfully!`);
          await loadAddons();
        } else if (status.state === "failed") {
          if (pollTimer) clearInterval(pollTimer);
          pollTimer = null;
          installing = null;
          installError = status.error;
          toasts.error(`${name} installation failed: ${status.error ?? "Unknown error"}`);
          await loadAddons();
        }
      } catch {
        // Ignore polling errors
      }
    }, 2000);
  }

  onMount(() => {
    loadAddons();
  });

  onDestroy(() => {
    if (pollTimer) clearInterval(pollTimer);
  });

  async function handleInstall(name: string) {
    if (!confirm(`Install ${name}? This will download ~2GB of dependencies (torch).`)) return;
    installing = name;
    installProgress = [];
    installError = null;
    toasts.info(`Starting ${name} installation...`);

    try {
      const result = await addonsApi.install(name);
      if (!result.success) {
        toasts.error(result.message);
        installing = null;
        return;
      }
      startPolling(name);
    } catch (e) {
      toasts.error(`Installation failed: ${e instanceof Error ? e.message : String(e)}`);
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

<div
  class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pb-2 mb-3 border-bottom"
>
  <h1 class="h4">Add-ons</h1>
</div>

<div class="alert alert-info">
  <strong>About Add-ons</strong><br />
  Add-ons provide optional ML capabilities that require large dependencies (torch ~2GB). They are not
  included in the base installer. Install them on demand — all data stays local.
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
              {:else if installing === addon.name}
                <span class="badge bg-warning text-dark">Installing</span>
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

            {#if installing === addon.name}
              <div class="mb-3">
                <div class="d-flex align-items-center gap-2 mb-2">
                  <div class="spinner-border spinner-border-sm"></div>
                  <span class="small text-muted">Installing in background...</span>
                </div>
                {#if installProgress.length > 0}
                  <div
                    class="bg-dark text-light p-2 rounded small"
                    style="max-height: 200px; overflow-y: auto; font-family: monospace;"
                  >
                    {#each installProgress as line}
                      <div>{line}</div>
                    {/each}
                  </div>
                {/if}
                {#if installError}
                  <div class="alert alert-danger mt-2 mb-0 small">{installError}</div>
                {/if}
              </div>
            {/if}

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
                  disabled={installing !== null}
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
