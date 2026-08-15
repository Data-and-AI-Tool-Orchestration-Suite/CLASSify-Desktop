<script lang="ts">
  import Router from "svelte-spa-router";
  import Home from "./routes/Home.svelte";
  import Prepare from "./routes/Prepare.svelte";
  import ResultsList from "./routes/ResultsList.svelte";
  import ResultsDetail from "./routes/ResultsDetail.svelte";
  import Settings from "./routes/Settings.svelte";
  import Addons from "./routes/Addons.svelte";
  import Navbar from "./lib/components/Navbar.svelte";
  import Toasts from "./lib/components/Toasts.svelte";
  import AppGate from "./lib/components/AppGate.svelte";

  const routes = {
    "/": Home,
    "/prepare/:reportId": Prepare,
    "/results": ResultsList,
    "/results/:reportId": ResultsDetail,
    "/settings": Settings,
    "/addons": Addons,
  };
</script>

<AppGate>
  <Navbar />

  <main class="container-fluid py-3">
    <Router {routes} />
  </main>
</AppGate>

<Toasts />
