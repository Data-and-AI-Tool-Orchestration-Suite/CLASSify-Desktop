<script lang="ts">
  import type { JobResponse } from "$lib/api/client";

  let { job, oncancel } = $props<{
    job: JobResponse | null;
    oncancel: () => void;
  }>();

  let pct = $derived(
    job && job.progress_total > 0 ? Math.round((job.progress / job.progress_total) * 100) : 0,
  );
</script>

{#if job}
  <div class="card mb-3 border-warning">
    <div class="card-body">
      <div class="d-flex justify-content-between align-items-center mb-2">
        <div>
          <span class="badge bg-warning text-dark">Training</span>
          <span class="ms-2 text-muted small">
            {job.progress_message ?? "Starting..."}
          </span>
        </div>
        <button class="btn btn-outline-danger btn-sm" onclick={oncancel}> Cancel </button>
      </div>
      <div class="progress" style="height: 24px;">
        <div
          class="progress-bar progress-bar-striped progress-bar-animated"
          role="progressbar"
          style="width: {pct}%"
          aria-valuenow={pct}
          aria-valuemin="0"
          aria-valuemax="100"
        >
          {pct}%
        </div>
      </div>
      <p class="text-muted small mt-1 mb-0">
        {job.progress} / {job.progress_total} models processed
      </p>
    </div>
  </div>
{/if}
