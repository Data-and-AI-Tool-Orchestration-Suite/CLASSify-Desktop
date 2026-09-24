<script lang="ts">
  import { system } from "$lib/api/client";
  import { onMount } from "svelte";
  import { location } from "svelte-spa-router";

  let appInfo = $state<{ app: string; version: string; os: string } | null>(null);
  let native = $state(false);

  onMount(async () => {
    native = "pywebview" in window;
    window.addEventListener("pywebviewready", () => (native = true), { once: true });
    try {
      appInfo = await system.info();
    } catch (e) {
      console.error("Failed to fetch app info", e);
    }
  });

  function winApi(): any {
    return (window as any).pywebview?.api ?? null;
  }

  function minimizeWindow() {
    winApi()?.minimize_window?.();
  }

  function toggleMaximizeWindow() {
    winApi()?.toggle_maximize_window?.();
  }

  function closeWindow() {
    winApi()?.close_window?.();
  }

  const navItems = [
    { href: "#/", label: "Datasets", icon: "house" },
    { href: "#/addons", label: "Add-ons", icon: "box" },
    { href: "#/settings", label: "Settings", icon: "gear" },
  ];

  function isActive(href: string): boolean {
    const path = href.replace("#", "");
    if (path === "/") {
      return $location === "/" || $location.startsWith("/results");
    }
    return $location?.startsWith(path) ?? false;
  }
</script>

<nav class="navbar navbar-expand-lg sticky-top">
  <div class="container-fluid">
    <a class="navbar-brand d-flex align-items-center gap-2 pywebview-drag-region" href="#/">
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
      <div
        class="pywebview-drag-region titlebar-drag flex-grow-1 align-self-stretch"
        ondblclick={toggleMaximizeWindow}
        role="presentation"
      ></div>
      <span class="navbar-text text-white-50 small">
        {#if appInfo}
          v{appInfo.version} · {appInfo.os}
        {/if}
      </span>
      {#if native}
        <div class="window-controls d-flex align-self-stretch align-items-center">
          <button type="button" class="wc-btn" onclick={minimizeWindow} title="Minimize">
            <span class="wc-glyph">&#x2500;</span>
          </button>
          <button
            type="button"
            class="wc-btn"
            onclick={toggleMaximizeWindow}
            title="Maximize / Restore"
          >
            <span class="wc-glyph">&#x25A1;</span>
          </button>
          <button type="button" class="wc-btn wc-close" onclick={closeWindow} title="Close">
            <span class="wc-glyph">&#x2715;</span>
          </button>
        </div>
      {/if}
    </div>
  </div>
</nav>
