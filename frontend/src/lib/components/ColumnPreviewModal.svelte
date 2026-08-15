<script lang="ts">
  import { onMount } from "svelte";
  import { datasets as datasetsApi, type ColumnChange } from "$lib/api/client";
  import { toasts } from "$lib/stores/app";

  let { reportId, onclose, oncomplete } = $props<{
    reportId: string;
    onclose: () => void;
    oncomplete: (changes: ColumnChange[], classColumn: string) => void;
  }>();

  let loading = $state(true);
  let saving = $state(false);
  let changes = $state<ColumnChange[]>([]);
  let originalMissing = $state<Record<string, boolean>>({});

  const TYPE_OPTIONS = ["float", "integer", "bool", "categorical", "string"];
  const MISSING_OPTIONS = ["", "drop", "constant", "synthetic"];

  async function loadColumnTypes() {
    loading = true;
    try {
      // Re-upload is not needed — we get types from the upload response
      // But if we're revisiting, we need to re-read the file. For now,
      // we'll fetch by re-uploading (the web app does the same).
      // Actually, the API stores the file on upload. We need a way to
      // get column types without re-uploading. Let's add an endpoint later.
      // For now, we'll initialize from defaults.
      changes = [];
    } catch (e) {
      toasts.error("Failed to load column types");
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    loadColumnTypes();
  });

  function setClassColumn(colName: string) {
    changes = changes.map((c) => ({ ...c, is_class: c.column === colName }));
  }

  async function handleSave() {
    const classCol = changes.find((c) => c.is_class);
    if (!classCol) {
      toasts.warning("Please select a class column");
      return;
    }

    saving = true;
    try {
      const resp = await datasetsApi.columnChanges(reportId, changes);
      if (resp.success) {
        oncomplete(changes, classCol.column);
      } else {
        toasts.error(resp.message || "Failed to apply column changes");
      }
    } catch (e) {
      toasts.error(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      saving = false;
    }
  }

  export function setColumns(types: Record<string, string>, missing: Record<string, boolean>) {
    originalMissing = missing;
    changes = Object.entries(types).map(([col, type]) => ({
      column: col,
      data_type: type,
      checked: true,
      missing: missing[col] ? "drop" : "",
      fill_value: "",
      is_class: false,
    }));
    loading = false;
  }
</script>

<div
  class="modal-backdrop fade show"
  onclick={onclose}
  onkeydown={(e) => e.key === "Escape" && onclose()}
  role="presentation"
></div>
<div class="modal fade show d-block" tabindex="-1" role="dialog">
  <div class="modal-dialog modal-xl" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Column Preview &amp; Configuration</h5>
        <button type="button" class="btn-close" onclick={onclose} aria-label="Close"></button>
      </div>
      <div class="modal-body" style="max-height: 60vh; overflow-y: auto;">
        {#if loading}
          <div class="text-center py-4">
            <div class="spinner-border" role="status"></div>
          </div>
        {:else}
          <table class="table table-sm align-middle">
            <thead>
              <tr>
                <th style="width: 40px;">Include</th>
                <th>Column Name</th>
                <th style="width: 140px;">Data Type</th>
                <th style="width: 140px;">Missing Values</th>
                <th style="width: 120px;">Fill Value</th>
                <th style="width: 80px;">Class Col</th>
              </tr>
            </thead>
            <tbody>
              {#each changes as change, i (change.column)}
                <tr class={!change.checked ? "table-secondary" : ""}>
                  <td>
                    <input
                      type="checkbox"
                      class="form-check-input"
                      bind:checked={changes[i].checked}
                    />
                  </td>
                  <td class="fw-medium">{change.column}</td>
                  <td>
                    <select
                      class="form-select form-select-sm"
                      bind:value={changes[i].data_type}
                      disabled={!change.checked}
                    >
                      {#each TYPE_OPTIONS as t}
                        <option value={t}>{t}</option>
                      {/each}
                    </select>
                  </td>
                  <td>
                    <select
                      class="form-select form-select-sm"
                      bind:value={changes[i].missing}
                      disabled={!change.checked || !originalMissing[change.column]}
                    >
                      {#each MISSING_OPTIONS as m}
                        <option value={m}>{m || "none"}</option>
                      {/each}
                    </select>
                  </td>
                  <td>
                    <input
                      type="text"
                      class="form-control form-control-sm"
                      placeholder="N/A"
                      bind:value={changes[i].fill_value}
                      disabled={!change.checked || change.missing !== "constant"}
                    />
                  </td>
                  <td class="text-center">
                    <input
                      type="radio"
                      class="form-check-input"
                      name="class-col"
                      checked={change.is_class}
                      onchange={() => setClassColumn(change.column)}
                      disabled={!change.checked}
                    />
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        {/if}
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" onclick={onclose}>Cancel</button>
        <button
          type="button"
          class="btn btn-primary"
          disabled={saving || loading}
          onclick={handleSave}
        >
          {#if saving}
            <span class="spinner-border spinner-border-sm me-1"></span>
            Saving...
          {:else}
            Apply Changes
          {/if}
        </button>
      </div>
    </div>
  </div>
</div>
