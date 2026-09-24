<script lang="ts">
  import type { JobResponse } from "$lib/api/client";

  let { job, log, oncancel } = $props<{
    job: JobResponse | null;
    log: string;
    oncancel: () => void;
  }>();

  type ModelState = "pending" | "running" | "done" | "skipped" | "failed";

  interface ModelInfo {
    name: string;
    state: ModelState;
    score: number | null;
    note: string;
  }

  let logEl = $state<HTMLPreElement | null>(null);

  const models = $derived.by(() => {
    const requested: string[] = Array.isArray(job?.args?.train_group) ? job!.args.train_group : [];
    const map = new Map<string, ModelInfo>();
    for (const name of requested) {
      map.set(name, { name, state: "pending", score: null, note: "" });
    }

    const ensure = (name: string): ModelInfo => {
      let info = map.get(name);
      if (!info) {
        info = { name, state: "pending", score: null, note: "" };
        map.set(name, info);
      }
      return info;
    };

    let current: ModelInfo | null = null;
    for (const raw of log.split("\n")) {
      const line = raw.trim();
      let m = line.match(/^Evaluating model:\s*(\S+)/);
      if (m) {
        current = ensure(m[1]);
        current.state = "running";
        continue;
      }
      m = line.match(/^Best params:.*\bscore:\s*(-?[\d.]+)/);
      if (m && current) {
        current.state = "done";
        current.score = Number.parseFloat(m[1]);
        continue;
      }
      m = line.match(/^Skipping\s+(\S+?)(?:\s+[—–-]\s+)(.+)$/);
      if (m) {
        const info = ensure(m[1]);
        info.state = "skipped";
        info.note = m[2];
        continue;
      }
      m = line.match(/^Error in (?:training|clustering|train\/evaluation of)\s+(\S+)/);
      if (m) {
        const info = ensure(m[1]);
        info.state = "failed";
        info.note = "error — see log";
        continue;
      }
      if (line.startsWith("All points classified as noise.") && current) {
        current.state = "done";
        current.note = "all points classified as noise";
      }
    }
    return [...map.values()];
  });

  const pct = $derived(
    job && job.progress_total > 0 ? Math.round((job.progress / job.progress_total) * 100) : 0,
  );

  const doneCount = $derived(models.filter((m) => m.state === "done").length);

  const currentState = $derived(models.find((m) => m.state === "running") ?? null);

  $effect(() => {
    if (logEl) {
      logEl.scrollTop = logEl.scrollHeight;
    }
  });

  function scoreLabel(score: number | null): string {
    if (score === null || Number.isNaN(score)) return "";
    return score.toFixed(3);
  }
</script>

<div class="card mb-3 border-warning">
  <div class="card-header d-flex justify-content-between align-items-center">
    <span>
      <span class="badge bg-warning text-dark">Training</span>
      {#if currentState}
        <span class="ms-2">
          Training <strong>{currentState.name}</strong>
          {#if log.includes("Tuning enabled")}
            <span class="text-muted small">(parameter tuning)</span>
          {/if}
        </span>
      {:else if job}
        <span class="ms-2 text-muted">{job.progress_message ?? "Starting..."}</span>
      {/if}
    </span>
    <button class="btn btn-outline-danger btn-sm" onclick={oncancel}> Cancel </button>
  </div>
  <div class="card-body">
    <div class="progress mb-1" style="height: 20px;">
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
    <p class="text-muted small mb-3">
      {doneCount} of {models.length || (job?.progress_total ?? 0)} models processed
    </p>

    {#if models.length > 0}
      <div class="list-group list-group-flush mb-3">
        {#each models as m (m.name)}
          <div class="list-group-item d-flex justify-content-between align-items-center px-0 py-2">
            <span>
              {#if m.state === "done"}
                <span class="text-success me-2">✓</span>
              {:else if m.state === "running"}
                <span class="spinner-border spinner-border-sm text-primary me-2" role="status"
                ></span>
              {:else if m.state === "failed"}
                <span class="text-danger me-2">✗</span>
              {:else if m.state === "skipped"}
                <span class="text-muted me-2">⊘</span>
              {:else}
                <span class="text-muted me-2">○</span>
              {/if}
              <span class={m.state === "pending" ? "text-muted" : ""}>{m.name}</span>
              {#if m.note}
                <span class="text-muted small ms-2">{m.note}</span>
              {/if}
            </span>
            <span>
              {#if m.state === "done" && m.score !== null}
                <span class="badge bg-success-subtle text-success-emphasis">
                  score {scoreLabel(m.score)}
                </span>
              {:else if m.state === "running"}
                <span class="badge bg-primary-subtle text-primary-emphasis">running</span>
              {:else if m.state === "failed"}
                <span class="badge bg-danger-subtle text-danger-emphasis">failed</span>
              {:else if m.state === "skipped"}
                <span class="badge bg-secondary-subtle text-secondary-emphasis">skipped</span>
              {/if}
            </span>
          </div>
        {/each}
      </div>
    {:else if !log}
      <div class="text-center py-3">
        <div class="spinner-border spinner-border-sm me-2"></div>
        <span class="text-muted">Waiting for output...</span>
      </div>
    {/if}

    <details>
      <summary class="text-muted small" style="cursor: pointer;">Technical log</summary>
      <pre
        bind:this={logEl}
        class="bg-dark text-light p-3 rounded mt-2 mb-0"
        style="max-height: 300px; overflow-y: auto; font-size: 0.8rem; white-space: pre-wrap; word-break: break-word; user-select: text; cursor: text;">{log ||
          "No output yet."}</pre>
    </details>
  </div>
</div>
