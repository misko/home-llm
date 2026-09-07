import { expect, test } from "@playwright/test";
import { portfolioFixture, runtimeFixture } from "../src/test/fixtures";

const operationId = "op_1234567890abcdef1234567890abcdef";

test("activates Muse when randomUUID is unavailable on an HTTP LAN host", async ({ page }) => {
  let museActive = false;
  let operationPolls = 0;
  let activationRequests = 0;
  let activationRequest: { headers: Record<string, string>; body: unknown } | undefined;
  const pageErrors: string[] = [];

  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.addInitScript(() => {
    // Plain HTTP LAN origins do not expose randomUUID in affected browsers.
    Object.defineProperty(Crypto.prototype, "randomUUID", {
      configurable: true,
      value: undefined,
    });
  });

  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === "/api/v1/events") {
      await route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: "retry: 60000\n\n",
      });
      return;
    }
    if (path === "/api/v1/models") {
      await route.fulfill({ json: portfolioFixture(museActive) });
      return;
    }
    if (path === "/api/v1/runtime") {
      await route.fulfill({
        json: runtimeFixture(museActive ? {
          revision: "rt1-local-agent",
          deployment_id: "muse-glimmer-30b-4090-8k",
          public_alias: "local-agent",
          model_name: "Muse Glimmer 30B",
        } : {}),
      });
      return;
    }
    if (path === "/api/v1/runtime/activations" && request.method() === "POST") {
      activationRequests += 1;
      activationRequest = {
        headers: request.headers(),
        body: request.postDataJSON(),
      };
      await route.fulfill({
        status: 202,
        json: {
          id: operationId,
          kind: "activate",
          state: "queued",
          requested_deployment_id: "muse-glimmer-30b-4090-8k",
          created_at: "2026-09-06T20:00:00Z",
          updated_at: "2026-09-06T20:00:00Z",
          message: "Activation queued",
        },
      });
      return;
    }
    if (path === `/api/v1/operations/${operationId}`) {
      operationPolls += 1;
      const succeeded = operationPolls >= 2;
      museActive = succeeded;
      await route.fulfill({
        json: {
          id: operationId,
          kind: "activate",
          state: succeeded ? "succeeded" : "running",
          requested_deployment_id: "muse-glimmer-30b-4090-8k",
          created_at: "2026-09-06T20:00:00Z",
          updated_at: "2026-09-06T20:00:01Z",
          message: succeeded ? "Muse is ready" : "Verifying model artifact",
          result: succeeded ? { deployment_id: "muse-glimmer-30b-4090-8k" } : null,
        },
      });
      return;
    }
    await route.abort("failed");
  });

  await page.goto("/ui/models");
  const museCard = page.getByTestId("model-muse-glimmer-30b");
  await expect(museCard).toContainText("Muse Glimmer 30B");
  await museCard.getByRole("button", { name: "Activate" }).click();

  const dialog = page.getByRole("dialog", { name: "Activate local-agent" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Close dialog" })).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(dialog.getByRole("button", { name: "Verify and activate" })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(dialog.getByRole("button", { name: "Close dialog" })).toBeFocused();
  await dialog.getByRole("button", { name: "Verify and activate" }).click();

  await expect(dialog).toContainText("Verifying model artifact");
  await expect(dialog).not.toBeVisible({ timeout: 10_000 });
  await expect(museCard).toContainText("Ready");

  expect(activationRequest?.body).toEqual({
    deployment_id: "muse-glimmer-30b-4090-8k",
    catalog_revision: "cat1-console-e2e",
  });
  expect(activationRequest?.headers["if-match"]).toBe("rt1-local-fast");
  expect(activationRequest?.headers["idempotency-key"]).toMatch(
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
  );
  expect(activationRequests).toBe(1);
  expect(operationPolls).toBeGreaterThanOrEqual(2);
  await expect(page.getByText("local-agent", { exact: true }).first()).toBeVisible();

  const lingCard = page.getByTestId("model-ling-3.0-tiny");
  const lingActivate = lingCard.getByRole("button", { name: "Activate" });
  await lingActivate.click();
  const lingDialog = page.getByRole("dialog", { name: "Activate local-fast" });
  await expect(lingDialog.getByRole("button", { name: "Verify and activate" })).toBeEnabled();
  await expect(lingDialog.getByRole("button", { name: "Close dialog" })).toBeFocused();
  await lingDialog.getByRole("button", { name: "Cancel" }).click();
  await expect(lingActivate).toBeFocused();
  expect(pageErrors).toEqual([]);
});
