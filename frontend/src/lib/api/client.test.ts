import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock fetch
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

beforeEach(() => {
  mockFetch.mockReset();
});

describe("API client", () => {
  it("system.health() calls /api/system/health", async () => {
    const { system } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: "ok" }),
      text: async () => "",
    });
    const result = await system.health();
    expect(result.status).toBe("ok");
    expect(mockFetch).toHaveBeenCalledWith("/api/system/health", expect.any(Object));
  });

  it("system.info() returns app metadata", async () => {
    const { system } = await import("./client");
    const mockInfo = {
      app: "CLASSify Desktop",
      version: "1.0.0.dev0",
      os: "Linux",
      os_version: "6.6",
      arch: "x86_64",
      python: "3.12.0",
      data_dir: "/tmp/data",
      dev_mode: true,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockInfo,
      text: async () => "",
    });
    const result = await system.info();
    expect(result.app).toBe("CLASSify Desktop");
  });

  it("throws on non-OK response", async () => {
    const { system } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      text: async () => "Server error",
    });
    await expect(system.health()).rejects.toThrow("API 500");
  });

  it("datasets.upload uses FormData", async () => {
    const { datasets } = await import("./client");
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        report_id: "abc",
        filename: "test",
        data_types: {},
        missing_values: {},
        message: null,
      }),
      text: async () => "",
    });
    const file = new File(["a,b,c\n1,2,3"], "test.csv", { type: "text/csv" });
    await datasets.upload(file);
    const call = mockFetch.mock.calls[0];
    expect(call[0]).toBe("/api/datasets/upload");
    expect(call[1].method).toBe("POST");
    expect(call[1].body).toBeInstanceOf(FormData);
  });

  it("jobs.sseUrl returns correct URL", async () => {
    const { jobs } = await import("./client");
    const url = jobs.sseUrl("job-123");
    expect(url).toBe("/api/jobs/job-123/events");
  });

  it("results.vizUrl returns correct URL", async () => {
    const { results } = await import("./client");
    const url = results.vizUrl("report-1", "ROC_curve");
    expect(url).toBe("/api/results/report-1/viz/ROC_curve");
  });

  it("results.downloadUrl encodes suffix", async () => {
    const { results } = await import("./client");
    const url = results.downloadUrl("report-1", "randomforest_model.joblib");
    expect(url).toBe("/api/results/report-1/download?suffix=randomforest_model.joblib");
  });
});
