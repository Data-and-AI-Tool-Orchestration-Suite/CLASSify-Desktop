<script lang="ts">
  /**
   * Time-series column role assignment modal (forecasting mode).
   * Ported from the Forecaster web app's column categorization table.
   */
  import { toasts } from "$lib/stores/app";
  import { datasets as datasetsApi, type TsRole, type TsRoleAssignment } from "$lib/api/client";

  let {
    reportId,
    columns,
    initialRoles = [],
    onclose,
    oncomplete,
  }: {
    reportId: string;
    columns: { column: string; dataType: string; missing: boolean }[];
    initialRoles?: TsRoleAssignment[];
    onclose: () => void;
    oncomplete: (roles: TsRoleAssignment[]) => void;
  } = $props();

  const ROLE_ORDER: { role: TsRole; label: string; tooltip: string }[] = [
    {
      role: "time-component",
      label: "Time Component",
      tooltip: "The column containing the date/timestamp for your time series.",
    },
    {
      role: "grouping",
      label: "Grouping",
      tooltip:
        "Column containing grouping IDs if multiple groups are being forecasted (e.g. different counties).",
    },
    {
      role: "target",
      label: "Target",
      tooltip: "The main column(s) you want to forecast.",
    },
    {
      role: "future-covariate",
      label: "Future Covariate",
      tooltip:
        "Features whose values are already present and known for the dates you want to predict.",
    },
    {
      role: "past-covariate",
      label: "Past Covariate",
      tooltip: "Features whose values are missing/unknown for the dates you want to predict.",
    },
    {
      role: "static-covariate",
      label: "Static Covariate",
      tooltip:
        "Features that do not change over time but can help distinguish between different groupings (if more than one distinct group is being forecasted).",
    },
    {
      role: "not-included",
      label: "Not Included",
      tooltip: "Columns that should be ignored and not used in the analysis.",
    },
  ];

  let roles = $state<Record<string, TsRole>>({});
  let saving = $state(false);

  $effect(() => {
    const map: Record<string, TsRole> = {};
    for (const col of columns) {
      map[col.column] = "not-included";
    }
    for (const r of initialRoles) {
      if (r.role !== "not-included" && r.column in map) {
        map[r.column] = r.role;
      }
    }
    roles = map;
  });

  function setRole(column: string, role: TsRole) {
    roles = { ...roles, [column]: role };
  }

  const timeComponentCount = $derived(
    Object.values(roles).filter((r) => r === "time-component").length,
  );
  const targetCount = $derived(Object.values(roles).filter((r) => r === "target").length);

  const validationError = $derived.by(() => {
    if (timeComponentCount !== 1) {
      return "You must have exactly one column selected for the Time Component.";
    }
    if (targetCount < 1) {
      return "You must have at least one target variable selected.";
    }
    return "";
  });

  async function save() {
    if (validationError) {
      toasts.error(validationError);
      return;
    }
    saving = true;
    try {
      const assignments: TsRoleAssignment[] = Object.entries(roles)
        .filter(([, role]) => role !== "not-included")
        .map(([column, role]) => ({ column, role }));
      const resp = await datasetsApi.tsRoles(reportId, assignments);
      if (resp.success) {
        toasts.success("Time-series configuration saved");
        oncomplete(assignments);
      } else {
        toasts.error(resp.message || "Failed to save time-series configuration");
      }
    } catch (e) {
      toasts.error(`Failed to save roles: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      saving = false;
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
  <div class="modal-dialog modal-xl modal-dialog-scrollable" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Time-Series Column Roles</h5>
        <button type="button" class="btn-close" onclick={onclose} aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <p class="text-muted small">
          Assign each column a role: exactly one <strong>Time Component</strong>, at least one
          <strong>Target</strong>, and optionally a grouping column and covariates.
        </p>
        <div class="table-scroll-container">
          <table class="table table-striped table-hover align-middle ts-role-table">
            <thead class="table-light">
              <tr>
                <th>Column Name</th>
                {#each ROLE_ORDER as r (r.role)}
                  <th class="text-center" title={r.tooltip}>{r.label}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each columns as col (col.column)}
                <tr>
                  <td>
                    <span class="fw-medium">{col.column}</span>
                    <span class="badge text-bg-light border ms-1">{col.dataType}</span>
                    {#if col.missing}
                      <span class="badge text-bg-warning ms-1" title="Column has missing values"
                        >missing</span
                      >
                    {/if}
                  </td>
                  {#each ROLE_ORDER as r (r.role)}
                    <td class="text-center">
                      <input
                        type="radio"
                        class="btn-check"
                        name={`role-${col.column}`}
                        id={`role-${col.column}-${r.role}`}
                        checked={roles[col.column] === r.role}
                        onchange={() => setRole(col.column, r.role)}
                      />
                      <label
                        class="btn btn-sm radio-bubble-label {roles[col.column] === r.role
                          ? 'radio-bubble-active'
                          : ''}"
                        for={`role-${col.column}-${r.role}`}
                      >
                        {roles[col.column] === r.role ? r.label : "Select"}
                      </label>
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>

        <details class="mt-3">
          <summary class="text-muted" style="cursor: pointer;"
            >Explanations about Covariates</summary
          >
          <div class="small mt-2">
            <p class="mb-2">
              <strong>Past covariates</strong> — values are unknown during the prediction window (e.g.
              measured sensor readings). Models only use their historical values.
            </p>
            <p class="mb-2">
              <strong>Future covariates</strong> — values are known during the prediction window (e.g.
              forecast weather, holidays, month of year).
            </p>
            <p class="mb-0">
              <strong>Static covariates</strong> — constant per grouping (e.g. population of a county);
              they help when forecasting multiple groups at once.
            </p>
          </div>
        </details>
      </div>
      <div class="modal-footer">
        <button class="btn btn-secondary" onclick={onclose}>Cancel</button>
        <button class="btn btn-success" disabled={saving} onclick={save}>
          {#if saving}
            <span class="spinner-border spinner-border-sm me-1"></span>
            Saving...
          {:else}
            Submit Categories
          {/if}
        </button>
      </div>
    </div>
  </div>
</div>
