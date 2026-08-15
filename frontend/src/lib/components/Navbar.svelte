<script lang="ts">
  import { system } from "$lib/api/client";
  import { onMount } from "svelte";
  import { location } from "svelte-spa-router";

  let appInfo = $state<{ app: string; version: string; os: string } | null>(null);

  onMount(async () => {
    try {
      appInfo = await system.info();
    } catch (e) {
      console.error("Failed to fetch app info", e);
    }
  });

  const navItems = [
    { href: "#/", label: "Home", icon: "house" },
    { href: "#/results", label: "Results", icon: "table" },
    { href: "#/addons", label: "Add-ons", icon: "box" },
    { href: "#/settings", label: "Settings", icon: "gear" },
  ];

  function isActive(href: string): boolean {
    const path = href.replace("#", "");
    if (path === "/") return $location === "/";
    return $location?.startsWith(path) ?? false;
  }
</script>

<nav class="navbar navbar-expand-lg sticky-top">
  <div class="container-fluid">
    <a class="navbar-brand d-flex align-items-center gap-2" href="#/">
      <span class="brand-text">CLASSify Desktop</span>
    </a>

    <button
      class="navbar-toggler"
      type="button"
      data-bs-toggle="collapse"
      data-bs-target="#navbarNav"
      aria-expanded="false"
      aria-label="Toggle navigation"
    >
      <span class="navbar-toggler-icon"></span>
    </button>

    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav me-auto">
        {#each navItems as item}
          <li class="nav-item">
            <a class="nav-link {isActive(item.href) ? 'active' : ''}" href={item.href}>
              {item.label}
            </a>
          </li>
        {/each}
      </ul>
      <span class="navbar-text text-white-50 small">
        {#if appInfo}
          v{appInfo.version} · {appInfo.os}
        {/if}
      </span>
    </div>
  </div>
</nav>
