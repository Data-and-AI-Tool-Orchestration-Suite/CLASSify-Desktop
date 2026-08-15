<script lang="ts">
  import { onMount } from "svelte";
  import { system } from "$lib/api/client";
  import { toasts } from "$lib/stores/app";

  let info = $state<Awaited<ReturnType<typeof system.info>> | null>(null);
  let usage = $state<Awaited<ReturnType<typeof system.usage>> | null>(null);
  let cpuCount = $state(0);
  let updateInfo = $state<{
    current_version: string;
    update_available: boolean;
    latest_version: string | null;
    download_url: string | null;
    release_notes: string;
    error: string | null;
  } | null>(null);
  let checkingUpdates = $state(false);

  function formatBytes(bytes: number): string {
    if (bytes === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
  }

  async function checkForUpdates() {
    checkingUpdates = true;
    try {
      updateInfo = await system.checkUpdates();
      if (updateInfo.update_available) {
        toasts.info(`Update available: v${updateInfo.latest_version}`);
      } else if (!updateInfo.error) {
        toasts.success("You're up to date!");
      }
    } catch {
      toasts.error("Failed to check for updates");
    } finally {
      checkingUpdates = false;
    }
  }

  onMount(async () => {
    try {
      [info, usage] = await Promise.all([system.info(), system.usage()]);
      const cpu = await system.cpuCount();
      cpuCount = cpu.count;
    } catch {
      toasts.error("Failed to load system info");
    }
  });
</script>

<div
  class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pb-2 mb-3 border-bottom"
>
  <h1 class="h4">Settings</h1>
</div>

{#if info}
  <div class="row g-4">
    <!-- Application info -->
    <div class="col-md-6">
      <div class="card">
        <div class="card-header"><h5 class="mb-0">Application</h5></div>
        <div class="card-body">
          <dl class="row mb-0">
            <dt class="col-sm-4">Version</dt>
            <dd class="col-sm-8">{info.version}</dd>
            <dt class="col-sm-4">OS</dt>
            <dd class="col-sm-8">{info.os} {info.os_version}</dd>
            <dt class="col-sm-4">Architecture</dt>
            <dd class="col-sm-8">{info.arch}</dd>
            <dt class="col-sm-4">Python</dt>
            <dd class="col-sm-8">{info.python}</dd>
            <dt class="col-sm-4">CPU Cores</dt>
            <dd class="col-sm-8">{cpuCount}</dd>
          </dl>
        </div>
      </div>
    </div>

    <!-- Storage -->
    <div class="col-md-6">
      <div class="card">
        <div class="card-header"><h5 class="mb-0">Storage</h5></div>
        <div class="card-body">
          <dl class="row mb-0">
            <dt class="col-sm-4">Data Directory</dt>
            <dd class="col-sm-8"><code>{info.data_dir}</code></dd>
            {#if usage}
              <dt class="col-sm-4">Disk Usage</dt>
              <dd class="col-sm-8">
                {formatBytes(usage.used)} used / {formatBytes(usage.total)} total
                <div class="progress mt-1" style="height: 8px;">
                  <div
                    class="progress-bar"
                    role="progressbar"
                    style="width: {((usage.used / usage.total) * 100).toFixed(1)}%"
                  ></div>
                </div>
                <span class="text-muted small">{formatBytes(usage.free)} free</span>
              </dd>
            {/if}
          </dl>
        </div>
      </div>
    </div>

    <!-- About -->
    <div class="col-12">
      <div class="card">
        <div class="card-header"><h5 class="mb-0">About</h5></div>
        <div class="card-body">
          <p class="mb-2">
            CLASSify Desktop is a local-first machine learning tool for training classification and
            clustering models on tabular data. All data stays on your machine — nothing is uploaded
            to any server.
          </p>
          <p class="text-muted small mb-3">
            Based on the CLASSify-2 web application. Replaces S3/ClearML/Postgres with local
            filesystem, SQLite, and in-process job execution.
          </p>

          <hr />

          <div class="d-flex justify-content-between align-items-center">
            <div>
              <strong>Check for Updates</strong>
              {#if updateInfo}
                <div class="mt-1">
                  {#if updateInfo.error}
                    <span class="text-danger small">{updateInfo.error}</span>
                  {:else if updateInfo.update_available}
                    <span class="text-success small">
                      Version {updateInfo.latest_version} is available! (current: {updateInfo.current_version})
                    </span>
                  {:else}
                    <span class="text-muted small">
                      You're up to date (v{updateInfo.current_version})
                    </span>
                  {/if}
                </div>
                {#if updateInfo.update_available && updateInfo.download_url}
                  <div class="mt-2">
                    <a
                      href={updateInfo.download_url}
                      target="_blank"
                      rel="noopener"
                      class="btn btn-primary btn-sm"
                    >
                      Download v{updateInfo.latest_version}
                    </a>
                  </div>
                  {#if updateInfo.release_notes}
                    <div class="mt-2">
                      <details>
                        <summary class="small text-muted">Release Notes</summary>
                        <pre class="small mt-1">{updateInfo.release_notes}</pre>
                      </details>
                    </div>
                  {/if}
                {/if}
              {/if}
            </div>
            <button
              class="btn btn-outline-primary btn-sm"
              disabled={checkingUpdates}
              onclick={checkForUpdates}
            >
              {#if checkingUpdates}
                <span class="spinner-border spinner-border-sm me-1"></span>
                Checking...
              {:else}
                Check Now
              {/if}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
{:else}
  <div class="text-center py-5">
    <div class="spinner-border" role="status"></div>
  </div>
{/if}
