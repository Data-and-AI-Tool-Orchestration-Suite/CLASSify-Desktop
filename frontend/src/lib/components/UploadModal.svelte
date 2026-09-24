<script lang="ts">
  import { onMount } from "svelte";
  import { datasets as datasetsApi, type DatasetUploadResponse } from "$lib/api/client";
  import { toasts } from "$lib/stores/app";

  let { onclose, oncomplete } = $props<{
    onclose: () => void;
    oncomplete: (result: DatasetUploadResponse) => void;
  }>();

  let file = $state<File | null>(null);
  let filePath = $state<string | null>(null);
  let uploading = $state(false);

  function isNativeShell(): boolean {
    return "pywebview" in window;
  }

  function basename(path: string): string {
    return path.replace(/\\/g, "/").split("/").pop() ?? path;
  }

  function acceptDroppedPath(path: string) {
    if (!path.toLowerCase().endsWith(".csv")) {
      toasts.error("Please select a CSV file");
      return;
    }
    filePath = path;
    file = null;
  }

  function handleFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      file = input.files[0];
      filePath = null;
    }
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    if (isNativeShell()) {
      return;
    }
    if (event.dataTransfer?.files?.[0]) {
      const f = event.dataTransfer.files[0];
      if (f.name.endsWith(".csv")) {
        file = f;
        filePath = null;
      } else {
        toasts.error("Please select a CSV file");
      }
    }
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
  }

  async function handleUpload() {
    if (!file && !filePath) return;
    uploading = true;
    try {
      const result = filePath
        ? await datasetsApi.uploadPath(filePath)
        : await datasetsApi.upload(file as File);
      oncomplete(result);
    } catch (e) {
      toasts.error(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      uploading = false;
    }
  }

  onMount(() => {
    function onNativeDrop(event: Event) {
      const detail = (event as CustomEvent<string[]>).detail ?? [];
      if (detail.length > 0) acceptDroppedPath(detail[0]);
    }
    window.addEventListener("classify-file-drop", onNativeDrop);
    return () => window.removeEventListener("classify-file-drop", onNativeDrop);
  });
</script>

<div
  class="modal-backdrop fade show"
  onclick={onclose}
  onkeydown={(e) => e.key === "Escape" && onclose()}
  role="presentation"
></div>
<div class="modal fade show d-block" tabindex="-1" role="dialog">
  <div class="modal-dialog modal-lg" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Upload Dataset</h5>
        <button type="button" class="btn-close" onclick={onclose} aria-label="Close"></button>
      </div>
      <div class="modal-body">
        <div
          class="border rounded p-5 text-center"
          style="border-style: dashed; cursor: pointer;"
          ondrop={handleDrop}
          ondragover={handleDragOver}
          onclick={() => document.getElementById("file-input")?.click()}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") document.getElementById("file-input")?.click();
          }}
          role="button"
          tabindex="0"
        >
          {#if file}
            <p class="mb-0 text-success">
              <strong>{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
            </p>
            <p class="text-muted small mt-1">Click to change file</p>
          {:else if filePath}
            <p class="mb-0 text-success"><strong>{basename(filePath)}</strong></p>
            <p class="text-muted small mt-1">Drop another file or click to change</p>
          {:else}
            <p class="text-muted mb-0">Drag &amp; drop a CSV file here, or click to browse</p>
          {/if}
          <input
            id="file-input"
            type="file"
            accept=".csv"
            class="d-none"
            onchange={handleFileSelect}
          />
        </div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" onclick={onclose}>Cancel</button>
        <button
          type="button"
          class="btn btn-primary"
          disabled={!file && !filePath}
          onclick={handleUpload}
        >
          {#if uploading}
            <span class="spinner-border spinner-border-sm me-1"></span>
            Uploading...
          {:else}
            Upload
          {/if}
        </button>
      </div>
    </div>
  </div>
</div>
