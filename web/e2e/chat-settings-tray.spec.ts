import { expect, test } from "@playwright/test";

test("chat tray generation-settings gear is visible and opens the settings dialog", async ({ page }) => {
  await page.goto("/ui/");

  const gear = page.getByRole("button", { name: "Open generation settings" });
  await expect(gear).toBeVisible();
  await expect(gear).toHaveCSS("background-color", "rgba(0, 0, 0, 0)");
  await expect(gear.locator("svg")).toBeVisible();
  await expect(gear).toHaveJSProperty("offsetWidth", 29);
  await expect(gear).toHaveJSProperty("offsetHeight", 29);

  await gear.click();
  await expect(page.getByRole("dialog", { name: "Generation settings" })).toBeVisible();
});
