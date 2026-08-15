<script lang="ts">
  import { datasets as datasetsApi } from "$lib/api/client";
  import { toasts } from "$lib/stores/app";

  let { reportId, classColumn, classValues, onclose, oncomplete } = $props<{
    reportId: string;
    classColumn: string;
    classValues: string[];
    onclose: () => void;
    oncomplete: () => void;
  }>();

  // Default mapping: first value → 1, rest → 0 (binary), or sequential (multiclass)
  let mapping = $state<Record<string, number>>({});

  // Initialize mapping — if binary with yes/no, map yes=1, no=0
  $effect(() => {
    const m: Record<string, number> = {};
    if (classValues.length === 2) {
      // Try to find yes/no pattern
      const yesIdx = classValues.findIndex(
        (v: string) => v.toLowerCase() === "yes" || v.toLowerCase() === "true",
      );
      if (yesIdx >= 0) {
        m[classValues[yesIdx]] = 1;
        m[classValues[1 - yesIdx]] = 0;
      } else {
        m[classValues[0]] = 0;
        m[classValues[1]] = 1;
      }
    } else {
      classValues.forEach((v: string) => {
      m[v] = Object.keys(m).length;
    });
    }
    mapping = m;
  });

  let dragItem = $state<string | null>(null);
  let orderedValues = $state<string[]>([]);

  $effect(() => {
    orderedValues = [...classValues];
  });

  function handleDragStart(value: string) {
    dragItem = value;
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
  }

  function handleDrop(target: string) {
    if (!dragItem || dragItem === target) return;
    const fromIdx = orderedValues.indexOf(dragItem);
    const toIdx = orderedValues.indexOf(target);
    orderedValues.splice(fromIdx, 1);
    orderedValues.splice(toIdx, 0, dragItem);
    // Reassign integers based on new order
    const m: Record<string, number> = {};
    orderedValues.forEach((v, i) => (m[v] = i));
    mapping = m;
    dragItem = null;
  }

  async function handleSave() {
    try {
      await datasetsApi.classMapping(reportId, classColumn, mapping);
      oncomplete();
    } catch (e) {
      toasts.error(`Failed to apply mapping: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
</script>

<div
  class="modal-backdrop fade show"
  onclick={onclose}
  onkeydown={(e) => e.key === "Escape" && onclose()}
  role="presentation"
></div>
<div class="modal fade show d-block" tabindex="-1" role="dialog">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Class Column Mapping</h5>
        <button type="button" class="btn-close" onclick={onclose} aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <p class="text-muted small">
          Drag to reorder. The integer value assigned to each class label determines the mapping
          used for training.
        </p>
        <div class="list-group">
          {#each orderedValues as value (value)}
            <div
              class="list-group-item d-flex align-items-center justify-content-between"
              draggable="true"
              ondragstart={() => handleDragStart(value)}
              ondragover={handleDragOver}
              ondrop={() => handleDrop(value)}
              style="cursor: grab;"
              role="option"
              aria-selected="false"
              tabindex="0"
            >
              <div>
                <span class="badge bg-primary me-2">{mapping[value]}</span>
                <strong>{value}</strong>
              </div>
              <span class="text-muted small">↕ drag to reorder</span>
            </div>
          {/each}
        </div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" onclick={onclose}>Cancel</button>
        <button type="button" class="btn btn-primary" onclick={handleSave}>Apply Mapping</button>
      </div>
    </div>
  </div>
</div>
