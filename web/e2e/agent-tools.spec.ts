import { expect, test } from "@playwright/test";

test("executes a read-only web search and renders its cited final answer", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto("/ui/");
  await page.getByRole("button", { name: "Generation settings" }).click();
  await page.getByLabel("System prompt").fill("Use concise language and cite sources.");
  await page.getByRole("button", { name: "Apply settings" }).click();
  const tools = page.getByRole("switch", { name: "Research tools" });
  await expect(tools).not.toBeChecked();
  await expect(page.getByText("Sends queries and requested public pages to the internet · no writes", { exact: true })).toBeVisible();
  await page.locator("label.tool-toggle").click();
  await expect(tools).toBeChecked();

  const agentRequest = page.waitForRequest((request) =>
    request.method() === "POST" && new URL(request.url()).pathname === "/api/v1/agent/turns",
  );
  await page.getByLabel("Message the active model").fill("Find the best way to add private web search");
  await page.getByRole("button", { name: "Send" }).click();

  const request = (await agentRequest).postDataJSON();
  expect(request).toMatchObject({
    instructions: "Use concise language and cite sources.",
    toolset: "standard-readonly",
    temperature: 0,
    max_tokens: 1024,
    stream: true,
    messages: [{ role: "user", content: "Find the best way to add private web search" }],
  });
  expect(request.messages).not.toContainEqual(expect.objectContaining({ role: "system" }));

  const toolCard = page.getByTestId("tool-search-fixture");
  await expect(toolCard).toContainText("Web Search");
  await expect(toolCard).toContainText("Complete");
  const calculatorCard = page.getByTestId("tool-calculator-fixture");
  await expect(calculatorCard).toContainText("Calculator");
  await expect(calculatorCard).toContainText("Runs locally · no network or writes");
  await expect(page.getByText("I found it through the read-only research tool.")).toBeVisible();

  const source = page.getByRole("link", { name: /SearXNG search documentation/ });
  await expect(source).toHaveAttribute("href", "https://docs.searxng.org/dev/search_api.html");
  await expect(source).toHaveAttribute("target", "_blank");
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible();
  expect(pageErrors).toEqual([]);
});
