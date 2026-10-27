<script lang="ts">
  /**
   * Forecast metrics table — grouped Normalized / De-Normalized headers
   * with grouping/component filters. Ported from the Forecaster results view.
   */
  import { FORECAST_METRIC_ORDER, modelLabel, type ForecastMetricRow } from "$lib/forecast";

  let {
    rows,
    metricDefs = {},
  }: {
    rows: ForecastMetricRow[];
    metricDefs?: Record<string, string>;
  } = $props();

  let groupingFilter = $state("all");
  let componentFilter = $state("all");

  const groupings = $derived([
    "all",
    ...new Set(
      rows.filter((r) => r.grouping !== "" && r.grouping != null).map((r) => String(r.grouping)),
    ),
  ]);
  const components = $derived([
    "all",
    ...new Set(
      rows.filter((r) => r.component !== "" && r.component != null).map((r) => String(r.component)),
    ),
  ]);

  const models = $derived([...new Set(rows.map((r) => r.model))]);

  function findRow(model: string): ForecastMetricRow | undefined {
    const g = groupingFilter === "all" ? "" : groupingFilter;
    const c = componentFilter === "all" ? "" : componentFilter;
    return rows.find(
      (r) => r.model === model && String(r.grouping ?? "") === g && String(r.component ?? "") === c,
    );
  }

  const visibleMetrics = $derived(
    FORECAST_METRIC_ORDER.filter((m) => rows.some((r) => m in r && r[m] !== null && r[m] !== "")),
  );

  function metricGroup(metric: string): "norm" | "denorm" | "solo" {
    if (metric.startsWith("Normalized ")) return "norm";
    if (metric.startsWith("De-Normalized ")) return "denorm";
    return "solo";
  }

  function metricSub(metric: string): string {
    return metric.replace(/^Normalized /, "").replace(/^De-Normalized /, "");
  }

  const hasNorm = $derived(visibleMetrics.some((m) => metricGroup(m) === "norm"));
  const hasDenorm = $derived(visibleMetrics.some((m) => metricGroup(m) === "denorm"));
  const soloMetrics = $derived(visibleMetrics.filter((m) => metricGroup(m) === "solo"));

  function formatValue(v: string | number | null | undefined): string {
    if (v === null || v === undefined || v === "") return "N/A";
    const num = Number(v);
    if (Number.isNaN(num)) return String(v);
    return num.toFixed(3);
  }

  const scaledMapeNote = $derived.by(() => {
    const normMape = rows.find((r) => "Normalized MAPE" in r)?.["Normalized MAPE"];
    const normSmape = rows.find((r) => "Normalized SMAPE" in r)?.["Normalized SMAPE"];
    const denormMape = rows.find((r) => "De-Normalized MAPE" in r)?.["De-Normalized MAPE"];
    const hasZeroNorm = Number(normMape) === 0 || Number(normSmape) === 0;
    if (!hasZeroNorm || !denormMape || Number(denormMape) === 0) return "";
    return (
      "Normalized MAPE/SMAPE could not be calculated because the scaled test series " +
      "contains zero values — use the De-Normalized MAPE/SMAPE columns instead."
    );
  });
</script>

<div class="d-flex flex-wrap gap-3 align-items-end mb-3">
  <div>
    <label class="form-label small mb-1" for="fc-grouping-filter">Select Grouping</label>
    <select id="fc-grouping-filter" class="form-select form-select-sm" bind:value={groupingFilter}>
      {#each groupings as g (g)}
        <option value={g}>{g === "all" ? "All Groupings" : g}</option>
      {/each}
    </select>
  </div>
  <div>
    <label class="form-label small mb-1" for="fc-component-filter">Select Component</label>
    <select
      id="fc-component-filter"
      class="form-select form-select-sm"
      bind:value={componentFilter}
    >
      {#each components as c (c)}
        <option value={c}>{c === "all" ? "All Components" : c}</option>
      {/each}
    </select>
  </div>
</div>

{#if scaledMapeNote}
  <div class="alert alert-info py-2 small">{scaledMapeNote}</div>
{/if}

<div class="table-responsive">
  <table class="table table-hover align-middle forecast-metrics-table">
    <thead class="table-light">
      <tr>
        <th rowspan="2">Model</th>
        {#if hasNorm}
          <th
            colspan={visibleMetrics.filter((m) => metricGroup(m) === "norm").length}
            class="text-center"
            title="Metrics calculated on data scaled between 0 and 1"
          >
            Normalized
          </th>
        {/if}
        {#if hasDenorm}
          <th
            colspan={visibleMetrics.filter((m) => metricGroup(m) === "denorm").length}
            class="text-center"
            title="Metrics calculated on the original, unscaled data"
          >
            De-Normalized
          </th>
        {/if}
        {#each soloMetrics as m (m)}
          <th rowspan="2" class="text-center" title={metricDefs[m] ?? ""}>{m}</th>
        {/each}
      </tr>
      <tr>
        {#if hasNorm}
          {#each visibleMetrics.filter((m) => metricGroup(m) === "norm") as m (m)}
            <th class="text-center" title={metricDefs[m] ?? ""}>{metricSub(m)}</th>
          {/each}
        {/if}
        {#if hasDenorm}
          {#each visibleMetrics.filter((m) => metricGroup(m) === "denorm") as m (m)}
            <th class="text-center" title={metricDefs[m] ?? ""}>{metricSub(m)}</th>
          {/each}
        {/if}
      </tr>
    </thead>
    <tbody>
      {#each models as model (model)}
        {@const row = findRow(model)}
        <tr>
          <td class="fw-medium">{modelLabel(model)}</td>
          {#if row}
            {#each visibleMetrics as m (m)}
              <td class="text-center">{formatValue(row[m])}</td>
            {/each}
          {:else}
            <td colspan={visibleMetrics.length} class="text-center text-muted">
              No data available for this selection.
            </td>
          {/if}
        </tr>
      {/each}
    </tbody>
  </table>
</div>
