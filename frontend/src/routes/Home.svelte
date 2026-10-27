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
  let filter = $state<"all" | "trained" | "drafts" | "failed">("all");
  let modeFilter = $state<"all" | "classification" | "forecasting">("all");

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
      Queued: "bg-warning",
      Processing: "bg-warning",
      Processed: "bg-success",
      Failed: "bg-danger",
    };
    return map[status] ?? "bg-secondary";
  }

  function datasetMode(report: DatasetRow): "classification" | "forecasting" | null {
    if (report.mode === "forecasting") return "forecasting";
    if (report.mode === "classification") return "classification";
    return null;
  }

  function modeLabel(mode: string | null): string {
    if (mode === "forecasting") return "Forecasting";
    if (mode === "classification") return "Classification";
    return "Unconfigured";
  }

  function isDraft(status: string): boolean {
    return ["Preview", "Uploaded"].includes(status);
  }

  // Drafts open straight into the prepare flow; everything else opens results
  function openHref(report: DatasetRow): string {
    return isDraft(report.status) ? `#/prepare/${report.uuid}` : `#/results/${report.uuid}`;
  }

  const filtered = $derived(
    reports.filter((r) => {
      if (modeFilter !== "all" && datasetMode(r) !== modeFilter) return false;
      if (filter === "all") return true;
      if (filter === "trained") return r.status === "Processed";
      if (filter === "drafts") return isDraft(r.status);
      return r.status === "Failed";
    }),
  );

  const counts = $derived({
    all: reports.length,
    trained: reports.filter((r) => r.status === "Processed").length,
    drafts: reports.filter((r) => isDraft(r.status)).length,
    failed: reports.filter((r) => r.status === "Failed").length,
  });

  const modeCounts = $derived({
    all: reports.length,
    classification: reports.filter((r) => datasetMode(r) === "classification").length,
    forecasting: reports.filter((r) => datasetMode(r) === "forecasting").length,
  });

  const modeFilters: { key: typeof modeFilter; label: string }[] = [
    { key: "all", label: "All modes" },
    { key: "classification", label: "Classification" },
    { key: "forecasting", label: "Forecasting" },
  ];

  const filters: { key: typeof filter; label: string }[] = [
    { key: "all", label: "All" },
    { key: "trained", label: "Trained" },
    { key: "drafts", label: "Not trained" },
    { key: "failed", label: "Failed" },
  ];
</script>

<div
  class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pb-2 mb-3 border-bottom"
>
  <h1 class="h4">Datasets</h1>
  <button class="btn btn-primary" onclick={() => (showUpload = true)}> Upload CSV </button>
</div>

<div class="d-flex flex-wrap gap-2 align-items-center mb-3">
  <div class="btn-group" role="group" aria-label="Filter by modeling mode">
    {#each modeFilters as mf (mf.key)}
      <button
        type="button"
        class="btn btn-sm {modeFilter === mf.key ? 'btn-primary' : 'btn-outline-secondary'}"
        onclick={() => (modeFilter = mf.key)}
      >
        {mf.label}
        <span class="badge ms-1 {modeFilter === mf.key ? 'bg-light text-dark' : 'bg-secondary'}">
          {modeCounts[mf.key]}
        </span>
      </button>
    {/each}
  </div>
  <div class="btn-group" role="group" aria-label="Filter datasets">
    {#each filters as f (f.key)}
      <button
        type="button"
        class="btn btn-sm {filter === f.key ? 'btn-primary' : 'btn-outline-secondary'}"
        onclick={() => (filter = f.key)}
      >
        {f.label}
        <span class="badge ms-1 {filter === f.key ? 'bg-light text-dark' : 'bg-secondary'}">
          {counts[f.key]}
        </span>
      </button>
    {/each}
  </div>
  <input
    type="text"
    class="form-control form-control-sm ms-auto"
    style="max-width: 280px;"
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
{:else if filtered.length === 0}
  <div class="text-center py-5 text-muted">
    {#if reports.length === 0}
      <p class="mb-3">No datasets yet. Upload a CSV to get started.</p>
      <button class="btn btn-primary btn-lg" onclick={() => (showUpload = true)}>
        Upload CSV
      </button>
    {:else}
      <p class="mb-0">No datasets match this filter.</p>
    {/if}
  </div>
{:else}
  <div class="table-responsive">
    <table class="table table-hover align-middle">
      <thead>
        <tr>
          <th>Filename</th>
          <th>Mode</th>
          <th>Status</th>
          <th>Created</th>
          <th>Comments</th>
          <th class="text-end">Actions</th>
        </tr>
      </thead>
      <tbody>
        {#each filtered as report (report.uuid)}
          <tr>
            <td>
              <a href={openHref(report)} class="text-decoration-none fw-medium">
                {report.filename}
              </a>
            </td>
            <td>
              <span
                class="badge {datasetMode(report) === 'forecasting'
                  ? 'text-bg-dark'
                  : datasetMode(report) === 'classification'
                    ? 'text-bg-primary'
                    : 'text-bg-light'}"
              >
                {modeLabel(datasetMode(report))}
              </span>
            </td>
            <td>
              <span class="badge {statusBadge(report.status)}">
                {#if report.status === "Processing" || report.status === "Queued"}
                  <span class="spinner-border spinner-border-sm me-1"></span>
                {/if}
                {report.status}
              </span>
            </td>
            <td class="text-muted small">{formatDate(report.created_at)}</td>
            <td class="text-muted small text-truncate" style="max-width: 200px;">
              {report.comments ?? ""}
            </td>
            <td class="text-end">
              <div class="btn-group btn-group-sm">
                {#if isDraft(report.status)}
                  <a
                    href={`#/prepare/${report.uuid}`}
                    class="btn btn-outline-primary"
                    title="Configure and train"
                  >
                    Prepare
                  </a>
                {/if}
                <a href={openHref(report)} class="btn btn-outline-secondary" title="Open dataset">
                  Open
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
