<script lang="ts">
  /**
   * Thin Plotly wrapper — lazily loads the plotly bundle on first render
   * and re-renders reactively when data/layout change.
   */
  import { onDestroy } from "svelte";

  let {
    data = [],
    layout = {},
  }: { data?: Record<string, unknown>[]; layout?: Record<string, unknown> } = $props();

  let el: HTMLDivElement | undefined = $state();
  let plotly: any = null;
  let disposed = $state(false);

  $effect(() => {
    const d = data;
    const l = layout;
    (async () => {
      if (!plotly) {
        const mod = await import("plotly.js-dist-min");
        plotly = mod.default ?? mod;
      }
      if (el && !disposed) {
        plotly.react(el, d, { responsive: true, ...l });
      }
    })();
  });

  onDestroy(() => {
    disposed = true;
    if (plotly && el) {
      try {
        plotly.purge(el);
      } catch {
        // element already gone
      }
    }
  });
</script>

<div bind:this={el} class="forecast-plot"></div>

<style>
  .forecast-plot {
    width: 100%;
    min-height: 420px;
  }
</style>
