<script lang="ts">
  import { onMount } from "svelte";
  import { system } from "$lib/api/client";
  import FirstRunWizard from "$lib/components/FirstRunWizard.svelte";

  let showWizard = $state(false);

  onMount(async () => {
    try {
      const state = await system.firstRun();
      showWizard = state.first_run;
    } catch {
      // If the endpoint fails, just show the app
    }
  });
</script>

{#if showWizard}
  <FirstRunWizard oncomplete={() => (showWizard = false)} />
{:else}
  <slot />
{/if}
