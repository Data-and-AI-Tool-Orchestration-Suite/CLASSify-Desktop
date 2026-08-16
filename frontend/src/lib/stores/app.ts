/**
 * Global stores for job status, toasts, and datasets.
 */

import { writable, type Writable } from "svelte/store";
import {
  jobs as jobsApi,
  results as resultsApi,
  type JobResponse,
} from "$lib/api/client";

// ── Toast store ──

export interface Toast {
  id: number;
  message: string;
  type: "success" | "error" | "info" | "warning";
  timeout?: number;
}

function createToastStore() {
  const { subscribe, update } = writable<Toast[]>([]);
  let nextId = 0;

  function show(message: string, type: Toast["type"] = "info", timeout = 5000) {
    const id = nextId++;
    update((toasts) => [...toasts, { id, message, type, timeout }]);
    if (timeout > 0) {
      setTimeout(() => dismiss(id), timeout);
    }
    return id;
  }

  function dismiss(id: number) {
    update((toasts) => toasts.filter((t) => t.id !== id));
  }

  return {
    subscribe,
    show,
    dismiss,
    success: (msg: string) => show(msg, "success"),
    error: (msg: string) => show(msg, "error", 8000),
    warning: (msg: string) => show(msg, "warning"),
    info: (msg: string) => show(msg, "info"),
  };
}

export const toasts = createToastStore();

// ── Job store ──

export const currentJob: Writable<JobResponse | null> = writable(null);
export const jobPolling: Writable<boolean> = writable(false);
export const liveLog: Writable<string> = writable("");

let pollInterval: ReturnType<typeof setInterval> | null = null;
let sseSource: EventSource | null = null;

export function startJobMonitoring(jobId: string, initialJob?: JobResponse) {
  stopJobMonitoring();
  jobPolling.set(true);
  if (initialJob) {
    currentJob.set(initialJob);
  }

  // SSE for live progress
  sseSource = new EventSource(jobsApi.sseUrl(jobId));
  sseSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      currentJob.set(data);
      if (data.log !== undefined) {
        liveLog.set(data.log);
      }
      if (data.state === "succeeded") {
        toasts.success("Training completed successfully!");
        stopJobMonitoring();
      } else if (data.state === "failed") {
        toasts.error(`Training failed: ${data.error || "Unknown error"}`);
        stopJobMonitoring();
      }
    } catch {
      // Ignore parse errors
    }
  };
  sseSource.onerror = () => {
    // Fallback to polling if SSE fails
    if (sseSource) {
      sseSource.close();
      sseSource = null;
    }
    if (!pollInterval) {
      pollInterval = setInterval(async () => {
        try {
          const job = await jobsApi.get(jobId);
          currentJob.set(job);
          if (job.report_uuid) {
            try {
              const logResp = await resultsApi.outputLog(job.report_uuid);
              liveLog.set(logResp.log);
            } catch {
              // Log not available yet
            }
          }
          if (job.state === "succeeded" || job.state === "failed") {
            stopJobMonitoring();
          }
        } catch {
          // Ignore polling errors
        }
      }, 2000);
    }
  };
}

export function stopJobMonitoring() {
  jobPolling.set(false);
  liveLog.set("");
  if (sseSource) {
    sseSource.close();
    sseSource = null;
  }
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
}

export async function cancelJob(jobId: string) {
  try {
    await jobsApi.cancel(jobId);
    toasts.info("Job cancellation requested...");
  } catch (e) {
    toasts.error(`Failed to cancel job: ${e instanceof Error ? e.message : String(e)}`);
  }
}

// ── Dataset refresh trigger ──

export const datasetRefresh: Writable<number> = writable(0);

export function triggerDatasetRefresh() {
  datasetRefresh.update((n) => n + 1);
}
