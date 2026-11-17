import { defineConfig, devices } from "@playwright/test";

// E2E tests run against the real explorer server on synthetic fixture
// data (tests/explorer_e2e_server.py) serving the built frontend in
// production mode. Run `npm run build` first (or use `just e2e`).
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  // All tests share one fixture server; a single worker keeps runs
  // deterministic.
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://127.0.0.1:8321",
    trace: "retain-on-failure",
    // The host may restrict unprivileged user namespaces (AppArmor);
    // headless Chromium then needs its sandbox off.
    launchOptions: { chromiumSandbox: false },
  },
  projects: [
    // Desktop-only by decision (2026-08-17): no mobile viewports.
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: {
    command:
      "cd .. && uv run --no-sync python tests/explorer_e2e_server.py --port 8321",
    url: "http://127.0.0.1:8321/",
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
