import { defineConfig, devices } from "@playwright/test";
import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

function cachedChromiumExecutable(): string | undefined {
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE) {
    return process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  }
  const cacheRoot = join(homedir(), ".cache", "ms-playwright");
  if (!existsSync(cacheRoot)) return undefined;
  const revisions = readdirSync(cacheRoot)
    .filter((entry) => /^chromium-\d+$/.test(entry))
    .sort((left, right) => Number(right.split("-")[1]) - Number(left.split("-")[1]));
  for (const revision of revisions) {
    for (const relative of ["chrome-linux64/chrome", "chrome-linux/chrome"]) {
      const candidate = join(cacheRoot, revision, relative);
      if (existsSync(candidate)) return candidate;
    }
  }
  return undefined;
}

const executablePath = cachedChromiumExecutable();

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:5174",
    trace: "retain-on-failure",
    launchOptions: executablePath ? { executablePath } : undefined,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: "node e2e/fixture-gateway.mjs",
      url: "http://127.0.0.1:14100/health",
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command: "npm run dev:e2e",
      env: { LLM_LAB_GATEWAY_URL: "http://127.0.0.1:14100" },
      url: "http://127.0.0.1:5174/ui/",
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
