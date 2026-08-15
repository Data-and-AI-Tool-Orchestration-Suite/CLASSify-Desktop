<script lang="ts">
  import { toasts } from "$lib/stores/app";

  const toastClass: Record<string, string> = {
    success: "alert-success",
    error: "alert-danger",
    warning: "alert-warning",
    info: "alert-info",
  };
</script>

<div class="toast-container position-fixed bottom-0 end-0 p-3" style="z-index: 9999;">
  {#each $toasts as toast (toast.id)}
    <div class="toast show {toastClass[toast.type] ?? 'alert-info'}" role="alert">
      <div class="toast-body d-flex align-items-center">
        <span class="flex-grow-1">{toast.message}</span>
        <button
          type="button"
          class="btn-close btn-close-white ms-2"
          onclick={() => toasts.dismiss(toast.id)}
          aria-label="Close"
        ></button>
      </div>
    </div>
  {/each}
</div>
