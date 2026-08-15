import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock fetch for all API calls
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

beforeEach(() => {
  mockFetch.mockReset();
});

describe("system API", () => {
  it("system.health() returns status", async () => {
    const { system } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "ok" }),
      text: async () => "",
    });
    const result = await system.health();
    expect(result.status).toBe("ok");
  });

  it("system.firstRun() returns wizard state", async () => {
    const { system } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        first_run: true,
        data_dir: "/tmp/data",
        disk_free_bytes: 1000000,
        cpu_count: 4,
        has_addons: { tabpfn: false, sdv: false },
      }),
      text: async () => "",
    });
    const result = await system.firstRun();
    expect(result.first_run).toBe(true);
    expect(result.cpu_count).toBe(4);
  });

  it("system.completeFirstRun() sends POST", async () => {
    const { system } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "ok" }),
      text: async () => "",
    });
    await system.completeFirstRun();
    expect(mockFetch).toHaveBeenCalledWith("/api/system/first-run/complete", expect.objectContaining({ method: "POST" }));
  });

  it("system.metricDefs() returns definitions dict", async () => {
    const { system } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ test_auc: "AUC description", test_acc: "Accuracy description" }),
      text: async () => "",
    });
    const result = await system.metricDefs();
    expect(result.test_auc).toContain("AUC");
    expect(result.test_acc).toContain("Accuracy");
  });
});

describe("datasets API", () => {
  it("datasets.list() with params builds query string", async () => {
    const { datasets } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ draw: 0, recordsTotal: 0, recordsFiltered: 0, data: [] }),
      text: async () => "",
    });
    await datasets.list({ start: 10, length: 50, search: "test" });
    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain("start=10");
    expect(url).toContain("length=50");
    expect(url).toContain("search=test");
  });

  it("datasets.delete() sends DELETE", async () => {
    const { datasets } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, message: "Deleted" }),
      text: async () => "",
    });
    await datasets.delete("abc-123");
    expect(mockFetch).toHaveBeenCalledWith("/api/datasets/abc-123", expect.objectContaining({ method: "DELETE" }));
  });

  it("datasets.comment() sends PATCH with body", async () => {
    const { datasets } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, message: "Updated" }),
      text: async () => "",
    });
    await datasets.comment("abc-123", "my comment");
    const call = mockFetch.mock.calls[0];
    expect(call[0]).toBe("/api/datasets/abc-123/comment");
    expect(call[1].method).toBe("PATCH");
    expect(JSON.parse(call[1].body).comments).toBe("my comment");
  });
});

describe("jobs API", () => {
  it("jobs.start() sends POST with options array", async () => {
    const { jobs } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: "job-1",
        report_uuid: "r-1",
        state: "queued",
        args: null,
        progress: 0,
        progress_total: 0,
        progress_message: null,
        error: null,
        created_at: null,
        started_at: null,
        finished_at: null,
      }),
      text: async () => "",
    });
    const result = await jobs.start("r-1", [{ name: "supervised", value: "True" }]);
    expect(result.state).toBe("queued");
    const call = mockFetch.mock.calls[0];
    expect(call[0]).toBe("/api/jobs");
    expect(call[1].method).toBe("POST");
    expect(JSON.parse(call[1].body).report_id).toBe("r-1");
  });

  it("jobs.cancel() sends POST to cancel endpoint", async () => {
    const { jobs } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "cancelled", message: "done" }),
      text: async () => "",
    });
    await jobs.cancel("job-1");
    expect(mockFetch).toHaveBeenCalledWith("/api/jobs/job-1/cancel", expect.objectContaining({ method: "POST" }));
  });
});

describe("results API", () => {
  it("results.get() returns results table", async () => {
    const { results } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        report_csv: [{ model: "rf", test_acc: 0.95 }],
        columns: ["model", "test_acc"],
        results_json: null,
      }),
      text: async () => "",
    });
    const result = await results.get("r-1");
    expect(result.success).toBe(true);
    expect(result.report_csv[0].model).toBe("rf");
  });

  it("results.downloadUrl() encodes suffix", async () => {
    const { results } = await import("./client");
    const url = results.downloadUrl("r-1", "randomforest_model.joblib");
    expect(url).toContain("suffix=randomforest_model.joblib");
  });

  it("results.shapRowGraphUrl() builds query params", async () => {
    const { results } = await import("./client");
    const url = results.shapRowGraphUrl("r-1", { model: "rf", row_num: 5 });
    expect(url).toContain("model=rf");
    expect(url).toContain("row_num=5");
  });
});

describe("addons API", () => {
  it("addons.list() returns addon array", async () => {
    const { addons } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        addons: [
          {
            name: "tabpfn",
            version: "2.0.0",
            description: "TabPFN",
            pip_deps: [],
            size_estimate_mb: 2500,
            min_app_version: "1.0.0",
            provides: ["tabpfn"],
            installed: false,
          },
        ],
      }),
      text: async () => "",
    });
    const result = await addons.list();
    expect(result.addons).toHaveLength(1);
    expect(result.addons[0].name).toBe("tabpfn");
  });

  it("addons.install() sends POST", async () => {
    const { addons } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, message: "Installed" }),
      text: async () => "",
    });
    await addons.install("tabpfn");
    expect(mockFetch).toHaveBeenCalledWith("/api/addons/tabpfn/install", expect.objectContaining({ method: "POST" }));
  });
});
