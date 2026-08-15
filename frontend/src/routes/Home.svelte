<script lang="ts">
  import { onMount } from "svelte";
  import { push } from "svelte-spa-router";
  import {
    datasets as datasetsApi,
    type DatasetRow,
    type DatasetUploadResponse,
  } from "$lib/api/client";
  import { toasts, triggerDatasetRefresh, datasetRefresh } from "$lib/stores/app";
  import UploadModal from "$lib/components/UploadModal.svelte";

  let reports = $state<DatasetRow[]>([]);
  let loading = $state(true);
  let showUpload = $state(false);
  let searchQuery = $state("");

  async function loadReports() {
    loading = true;
    try {
      const resp = await datasetsApi.list({ length: 100, search: searchQuery });
      reports = resp.data;
    } catch (e) {
      toasts.error("Failed to load datasets");
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    loadReports();
  });

  // Reload when datasetRefresh changes (e.g. after upload, delete, job completion)
  $effect(() => {
    $datasetRefresh;
    loadReports();
  });

  function handleUploadComplete(result: DatasetUploadResponse) {
    showUpload = false;
    if (result.success && result.report_id) {
      toasts.success(`Uploaded ${result.filename}`);
      triggerDatasetRefresh();
      push(`/prepare/${result.report_id}`);
    } else {
      toasts.error(result.message || "Upload failed");
    }
  }

  async function handleDelete(id: string, filename: string) {
    if (!confirm(`Delete "${filename}"? This cannot be undone.`)) return;
    try {
      await datasetsApi.delete(id);
      toasts.success("Dataset deleted");
      triggerDatasetRefresh();
    } catch (e) {
      toasts.error("Failed to delete dataset");
    }
  }

  async function handleDuplicate(id: string) {
    try {
      const resp = await datasetsApi.duplicate(id);
      toasts.success(resp.message);
      triggerDatasetRefresh();
    } catch (e) {
      toasts.error("Failed to duplicate dataset");
    }
  }

  function formatDate(iso: string | null): string {
    if (!iso) return "";
    return new Date(iso).toLocaleString();
  }

  function statusBadge(status: string): string {
    const map: Record<string, string> = {
      Preview: "bg-secondary",
      Uploaded: "bg-info",
      Processing: "bg-warning",
      Processed: "bg-success",
      Failed: "bg-danger",
    };
    return map[status] ?? "bg-secondary";
  }
</script>

<div
  class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pb-2 mb-3 border-bottom"
>
  <h1 class="h4">Datasets</h1>
  <button class="btn btn-primary" onclick={() => (showUpload = true)}> Upload CSV </button>
</div>

<div class="mb-3">
  <input
    type="text"
    class="form-control"
    placeholder="Search datasets..."
    bind:value={searchQuery}
    oninput={() => loadReports()}
  />
</div>

{#if loading}
  <div class="text-center py-5">
    <div class="spinner-border" role="status">
      <span class="visually-hidden">Loading...</span>
    </div>
  </div>
{:else if reports.length === 0}
  <div class="text-center py-5 text-muted">
    <p class="mb-3">No datasets yet. Upload a CSV to get started.</p>
    <button class="btn btn-primary btn-lg" onclick={() => (showUpload = true)}> Upload CSV </button>
  </div>
{:else}
  <div class="table-responsive">
    <table class="table table-hover align-middle">
      <thead>
        <tr>
          <th>Filename</th>
          <th>Status</th>
          <th>Created</th>
          <th>Comments</th>
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
              <span class="badge {statusBadge(report.status)}">{report.status}</span>
            </td>
            <td class="text-muted small">{formatDate(report.created_at)}</td>
            <td class="text-muted small text-truncate" style="max-width: 200px;">
              {report.comments ?? ""}
            </td>
            <td class="text-end">
              <div class="btn-group btn-group-sm">
                {#if report.status === "Preview" || report.status === "Uploaded"}
                  <a
                    href={`#/prepare/${report.uuid}`}
                    class="btn btn-outline-primary"
                    title="Prepare"
                  >
                    Prepare
                  </a>
                {/if}
                <a
                  href={`#/results/${report.uuid}`}
                  class="btn btn-outline-secondary"
                  title="View Results"
                >
                  Results
                </a>
                <button
                  class="btn btn-outline-secondary"
                  title="Duplicate"
                  onclick={() => handleDuplicate(report.uuid)}
                >
                  Copy
                </button>
                <button
                  class="btn btn-outline-danger"
                  title="Delete"
                  onclick={() => handleDelete(report.uuid, report.filename)}
                >
                  Delete
                </button>
              </div>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
{/if}

{#if showUpload}
  <UploadModal onclose={() => (showUpload = false)} oncomplete={handleUploadComplete} />
{/if}
