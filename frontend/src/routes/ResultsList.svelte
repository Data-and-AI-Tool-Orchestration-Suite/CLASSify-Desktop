<script lang="ts">
  import { onMount } from "svelte";
  import { datasets as datasetsApi, type DatasetRow } from "$lib/api/client";
  import { toasts, datasetRefresh } from "$lib/stores/app";

  let reports = $state<DatasetRow[]>([]);
  let loading = $state(true);

  async function loadReports() {
    loading = true;
    try {
      const resp = await datasetsApi.list({ length: 100 });
      reports = resp.data.filter((r) => r.status === "Processed" || r.status === "Failed");
    } catch {
      toasts.error("Failed to load results");
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    loadReports();
  });

  $effect(() => {
    $datasetRefresh;
    loadReports();
  });

  function formatDate(iso: string | null): string {
    if (!iso) return "";
    return new Date(iso).toLocaleString();
  }
</script>

<div
  class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pb-2 mb-3 border-bottom"
>
  <h1 class="h4">Results</h1>
</div>

{#if loading}
  <div class="text-center py-5">
    <div class="spinner-border" role="status"></div>
  </div>
{:else if reports.length === 0}
  <div class="text-center py-5 text-muted">
    <p>No trained datasets yet. Upload a CSV and train models to see results here.</p>
    <a href="#/" class="btn btn-primary">Upload Data</a>
  </div>
{:else}
  <div class="table-responsive">
    <table class="table table-hover align-middle">
      <thead>
        <tr>
          <th>Filename</th>
          <th>Status</th>
          <th>Created</th>
          <th class="text-end">Actions</th>
        </tr>
      </thead>
      <tbody>
        {#each reports as report (report.uuid)}
          <tr>
            <td>
              <a href={`#/results/${report.uuid}`} class="text-decoration-none fw-medium">
                {report.filename}
              </a>
            </td>
            <td>
              <span class="badge bg-{report.status === 'Processed' ? 'success' : 'danger'}">
                {report.status}
              </span>
            </td>
            <td class="text-muted small">{formatDate(report.created_at)}</td>
            <td class="text-end">
              <a href={`#/results/${report.uuid}`} class="btn btn-outline-primary btn-sm">
                View Results
              </a>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}
