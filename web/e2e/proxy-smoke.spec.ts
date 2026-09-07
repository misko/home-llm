import { expect, test } from "@playwright/test";

test("loads the catalog through the real Vite-to-gateway proxy", async ({ page }) => {
  const catalogResponse = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/v1/models",
  );

  await page.goto("/ui/models");

  expect((await catalogResponse).status()).toBe(200);
  await expect(page.getByRole("heading", { name: "Models", exact: true })).toBeVisible();
  await expect(page.getByTestId("model-muse-glimmer-30b")).toContainText("Muse Glimmer 30B");
  await expect(page.getByText("Catalog API unavailable")).toHaveCount(0);
});
