/**
 * Typed API client for the CLASSify Desktop frontend.
 *
 * All backend endpoints are exposed as typed async functions.
 * The base URL is `/api` (proxied by Vite in dev, served by FastAPI in production).
 */

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`API ${resp.status}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

async function upload<T>(
  path: string,
  file: File,
  extraParams?: Record<string, string>,
): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  if (extraParams) {
    for (const [k, v] of Object.entries(extraParams)) {
      form.append(k, v);
    }
  }
  const resp = await fetch(`${BASE}${path}`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => resp.statusText);
    throw new Error(`API ${resp.status}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

// ── System ──

export interface HealthResponse {
  status: string;
}
export interface InfoResponse {
  app: string;
  version: string;
  os: string;
  os_version: string;
  arch: string;
  python: string;
  data_dir: string;
  dev_mode: boolean;
}
export interface UsageResponse {
  total: number;
  used: number;
  free: number;
}

export const system = {
  health: () => request<HealthResponse>("/system/health"),
  info: () => request<InfoResponse>("/system/info"),
  usage: () => request<UsageResponse>("/system/usage"),
  cpuCount: () => request<{ count: number }>("/system/cpu-count"),
  checkUpdates: () =>
    request<{
      current_version: string;
      update_available: boolean;
      latest_version: string | null;
      download_url: string | null;
      release_notes: string;
      error: string | null;
    }>("/system/check-updates"),
  firstRun: () =>
    request<{
      first_run: boolean;
      data_dir: string;
      disk_free_bytes: number;
      cpu_count: number;
      has_addons: { tabpfn: boolean; sdv: boolean };
    }>("/system/first-run"),
  completeFirstRun: () =>
    request<{ status: string }>("/system/first-run/complete", { method: "POST" }),
  metricDefs: () => request<Record<string, string>>("/system/metric-defs"),
};

// ── Datasets ──

export interface DatasetUploadResponse {
  success: boolean;
  report_id: string | null;
  filename: string | null;
  data_types: Record<string, string>;
  missing_values: Record<string, boolean>;
  message: string | null;
}

export interface DatasetRow {
  uuid: string;
  filename: string;
  original_filename: string | null;
  status: string;
  job_id: string | null;
  column_changes: Record<string, unknown> | null;
  comments: string | null;
  created_at: string | null;
}

export interface DatasetListResponse {
  draw: number;
  recordsTotal: number;
  recordsFiltered: number;
  data: DatasetRow[];
}

export interface ColumnChange {
  column: string;
  data_type: string;
  checked: boolean;
  missing: string;
  fill_value: string;
  is_class: boolean;
}

export const datasets = {
  upload: (file: File) => upload<DatasetUploadResponse>("/datasets/upload", file),
  list: (params?: {
    start?: number;
    length?: number;
    search?: string;
    order_by?: string;
    order_dir?: string;
  }) => {
    const q = new URLSearchParams();
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined) q.set(k, String(v));
      }
    }
    return request<DatasetListResponse>(`/datasets?${q}`);
  },
  get: (id: string) => request<DatasetRow>(`/datasets/${id}`),
  delete: (id: string) =>
    request<{ success: boolean; message: string }>(`/datasets/${id}`, { method: "DELETE" }),
  duplicate: (id: string) =>
    request<{ success: boolean; message: string }>(`/datasets/${id}/duplicate`, { method: "POST" }),
  columnChanges: (id: string, dataTypes: ColumnChange[]) =>
    request<{ success: boolean; message: string; data_types: ColumnChange[] }>(
      `/datasets/${id}/column-changes`,
      {
        method: "POST",
        body: JSON.stringify({ data_types: dataTypes }),
      },
    ),
  classValues: (id: string, classColumn: string) =>
    request<{ success: boolean; class_values: string[] }>(
      `/datasets/${id}/class-values?class_column=${encodeURIComponent(classColumn)}`,
    ),
  classMapping: (id: string, classColumn: string, mapping: Record<string, number>) =>
    request<{ success: boolean; message: string }>(`/datasets/${id}/class-mapping`, {
      method: "POST",
      body: JSON.stringify({ class_column: classColumn, mapping }),
    }),
  testset: (id: string, file: File, classColumn?: string) =>
    upload<{ success: boolean; message: string }>(
      `/datasets/${id}/testset${classColumn ? `?class_column=${encodeURIComponent(classColumn)}` : ""}`,
      file,
    ),
  comment: (id: string, comments: string) =>
    request<{ success: boolean; message: string }>(`/datasets/${id}/comment`, {
      method: "PATCH",
      body: JSON.stringify({ comments }),
    }),
};

// ── Jobs ──

export interface JobResponse {
  id: string;
  report_uuid: string;
  state: string;
  args: Record<string, unknown> | null;
  progress: number;
  progress_total: number;
  progress_message: string | null;
  error: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface TrainOption {
  name: string;
  value: string;
}

export const jobs = {
  start: (reportId: string, options: TrainOption[]) =>
    request<JobResponse>("/jobs", {
      method: "POST",
      body: JSON.stringify({ report_id: reportId, options }),
    }),
  list: () => request<{ jobs: JobResponse[] }>("/jobs"),
  get: (jobId: string) => request<JobResponse>(`/jobs/${jobId}`),
  cancel: (jobId: string) =>
    request<{ status: string; message: string }>(`/jobs/${jobId}/cancel`, { method: "POST" }),
  recover: () => request<{ recovered: number }>("/jobs/recover", { method: "POST" }),
  mlOptionsSupervised: () => request<Record<string, unknown>>("/jobs/ml-options/supervised"),
  mlOptionsUnsupervised: () => request<Record<string, unknown>>("/jobs/ml-options/unsupervised"),
  sseUrl: (jobId: string) => `${BASE}/jobs/${jobId}/events`,
};

// ── Addons ──

export interface AddonInfo {
  name: string;
  version: string;
  description: string;
  pip_deps: string[];
  size_estimate_mb: number;
  min_app_version: string;
  provides: string[];
  installed: boolean;
}

export const addons = {
  list: () => request<{ addons: AddonInfo[] }>("/addons"),
  status: (name: string) => request<Record<string, unknown>>(`/addons/${name}/status`),
  install: (name: string) =>
    request<{ success: boolean; message: string }>(`/addons/${name}/install`, { method: "POST" }),
  uninstall: (name: string) =>
    request<{ success: boolean; message: string }>(`/addons/${name}/uninstall`, { method: "POST" }),
  checkModules: () =>
    request<{ modules: { name: string; module: string; installed: boolean }[] }>(
      "/addons/modules/check",
    ),
};

export interface ResultsResponse {
  success: boolean;
  report_csv: Record<string, unknown>[];
  columns: string[];
  results_json: Record<string, unknown> | null;
}

export interface VizListResponse {
  success: boolean;
  visualizations: string[];
}

export interface ShapRowsResponse {
  success: boolean;
  rows: Record<string, unknown>[];
  columns: string[];
}

export interface PrepareParamsResponse {
  success: boolean;
  parameters: Record<string, unknown> | null;
  class_column: string | null;
}

export const results = {
  get: (reportId: string) => request<ResultsResponse>(`/results/${reportId}`),
  vizList: (reportId: string) => request<VizListResponse>(`/results/${reportId}/viz`),
  vizUrl: (reportId: string, vizName: string) => `${BASE}/results/${reportId}/viz/${vizName}`,
  shapRows: (reportId: string, model: string) =>
    request<ShapRowsResponse>(`/results/${reportId}/shap-rows/${model}`),
  shapRowGraphUrl: (
    reportId: string,
    params: { model: string; row_num: number; train_test?: string; class_column?: string },
  ) => {
    const q = new URLSearchParams({
      model: params.model,
      row_num: String(params.row_num),
      train_test: params.train_test ?? "test",
      class_column: params.class_column ?? "class",
    });
    return `${BASE}/results/${reportId}/shap-row-graph?${q}`;
  },
  retest: (reportId: string, file: File, _modelNames: string[], _classColumn: string) =>
    upload<{ success: boolean; message: string }>(`/results/${reportId}/retest`, file),
  outputLog: (reportId: string) =>
    request<{ success: boolean; log: string }>(`/results/${reportId}/output-log`),
  downloadUrl: (reportId: string, suffix: string) =>
    `${BASE}/results/${reportId}/download?suffix=${encodeURIComponent(suffix)}`,
  prepareParams: (reportId: string) =>
    request<PrepareParamsResponse>(`/results/${reportId}/prepare-params`),
};
