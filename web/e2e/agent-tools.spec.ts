import { expect, test } from "@playwright/test";

test("shows both explicitly requested remote reviewers and their separate tool results", async ({ page }) => {
  const models = ["anthropic/claude-opus-5", "openai/gpt-6-astra"];
  await page.route("**/api/v1/agent/turns", async (route) => {
    expect(route.request().postDataJSON().messages.at(-1).content).toBe("Can you run against Opus and Astra using open router?");
    const events = [
      { type: "turn.started", model: "local-fast", toolset: "assistant-tools", tools: ["openrouter_delegate"] },
      ...models.flatMap((model, index) => [
        { type: "tool.started", call_id: `review-${index}`, name: "openrouter_delegate", arguments: { model, prompt: "Review the release notes." }, round: index + 1 },
        { type: "tool.completed", call_id: `review-${index}`, name: "openrouter_delegate", result: { model, content: `Review from ${model}` }, round: index + 1, duration_ms: 10 },
      ]),
      { type: "assistant.delta", content: "Both requested reviewers completed their checks.", round: 3 },
      { type: "turn.completed", model: "local-fast", rounds: 3, finish_reason: "stop", tools_used: ["openrouter_delegate"], delegated_models: models, recovery_reasons: [] },
    ];
    await route.fulfill({ contentType: "text/event-stream", body: events.map((event, index) => `data: ${JSON.stringify({ schema_version: 1, run_id: "dual-review", sequence: index + 1, ...event })}\n\n`).join("") });
  });
  await page.goto("/ui/");
  await page.getByLabel("Message the active model").fill("Can you run against Opus and Astra using open router?");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText(`Local model + OpenRouter · ${models.join(", ")}`, { exact: true })).toBeVisible();
  await page.locator("details.tool-activity > summary").click();
  for (const [index, model] of models.entries()) {
    const tool = page.getByTestId(`tool-review-${index}`);
    await expect(tool).toContainText("Complete");
    await tool.locator(":scope > summary").click();
    await expect(tool.locator(".tool-execution-detail")).toContainText(`Review from ${model}`);
  }
});

test("shows recovered delegation evidence in collapsible activity and preserves provenance after reload", async ({ page }) => {
  await page.route("**/api/v1/agent/turns", async (route) => {
    const events = [
      { type: "turn.started", model: "local-fast", toolset: "assistant-tools", tools: ["openrouter_delegate"] },
      { type: "tool.started", call_id: "recovery-delegate", name: "openrouter_delegate", arguments: { model: "qwen/test", prompt: "Review the release notes." }, round: 2 },
      { type: "tool.completed", call_id: "recovery-delegate", name: "openrouter_delegate", result: { model: "qwen/test", content: "The release notes cover all changes." }, round: 2, duration_ms: 10 },
      { type: "assistant.delta", content: "The delegated review confirms the release notes cover all changes.", round: 3 },
      { type: "turn.completed", model: "local-fast", rounds: 3, finish_reason: "stop", tools_used: ["openrouter_delegate"], delegated_models: ["qwen/test"], recovery_reasons: ["repeated_answer"] },
    ];
    await route.fulfill({
      contentType: "text/event-stream",
      body: events.map((event, index) => `data: ${JSON.stringify({ schema_version: 1, run_id: "recovery-fixture", sequence: index + 1, ...event })}\n\n`).join(""),
    });
  });
  await page.goto("/ui/");
  await page.getByLabel("Message the active model").fill("Ask OpenRouter to review the release notes.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("The delegated review confirms the release notes cover all changes.", { exact: true })).toBeVisible();
  await expect(page.getByText("Local model + OpenRouter · qwen/test", { exact: true })).toBeVisible();
  await expect(page.getByText("Repeated answer detected · regenerated", { exact: true })).toBeVisible();
  const activity = page.locator("details.tool-activity");
  await expect(activity).not.toHaveAttribute("open", "");
  await activity.locator(":scope > summary").click();
  const delegate = page.getByTestId("tool-recovery-delegate");
  await expect(delegate).toContainText("Round 2");
  await expect(delegate).toContainText("Complete");
  await delegate.locator(":scope > summary").click();
  await expect(delegate.locator(".tool-execution-detail")).toContainText("The release notes cover all changes.");
  // Wait for the debounced history write before navigating away.
  await expect.poll(() => page.evaluate(async () => {
    const modulePath = "/ui/src/chat/history.ts";
    const { loadChats, loadMessagePage } = await import(modulePath);
    const chats = await loadChats();
    for (const chat of chats) {
      const page = await loadMessagePage(chat.id);
      if (page.messages.some((message: { provenance?: { delegated_models: string[] } }) => message.provenance?.delegated_models.includes("qwen/test"))) return true;
    }
    return false;
  })).toBe(true);
  await page.reload();
  await expect(page.getByText("Local model + OpenRouter · qwen/test", { exact: true })).toBeVisible();
  await expect(page.getByText("Repeated answer detected · regenerated", { exact: true })).toBeVisible();
});

test("executes a read-only web search and renders its cited final answer", async ({ page }) => {
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));

  await page.goto("/ui/");
  await page.getByRole("button", { name: "Generation settings", exact: true }).click();
  const maximumOutput = page.getByLabel("Maximum output tokens");
  await expect(maximumOutput).toHaveValue("32000");
  await expect(maximumOutput).toHaveAttribute("max", "32768");
  await page.getByLabel("System prompt").fill("Use concise language and cite sources.");
  await page.getByRole("button", { name: "Apply settings" }).click();
  const tools = page.getByRole("switch", { name: "Research tools" });
  await expect(tools).toBeChecked();
  await expect(page.getByText("Web research plus the permissions selected in settings", { exact: true })).toBeVisible();

  const agentRequest = page.waitForRequest((request) =>
    request.method() === "POST" && new URL(request.url()).pathname === "/api/v1/agent/turns",
  );
  await page.getByLabel("Message the active model").fill("Find the best way to add private web search");
  await page.getByRole("button", { name: "Send" }).click();

  const request = (await agentRequest).postDataJSON();
  expect(request).toMatchObject({
    instructions: "Use concise language and cite sources.",
    toolset: "assistant-tools",
    enabled_tools: ["web_search", "web_fetch", "calculator", "current_time", "workspace_list", "workspace_read", "workspace_write", "python_sandbox", "openrouter_delegate"],
    allow_workspace_writes: true,
    temperature: 0,
    max_tokens: 32_000,
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
  await expect(page.locator(".message-content").filter({ hasText: "I found it through the read-only research tool." })).toBeVisible();
  await expect(page.getByText("Local model + 2 tools", { exact: true })).toBeVisible();

  const source = page.getByRole("link", { name: /SearXNG search documentation/ });
  await expect(source).toHaveAttribute("href", "https://docs.searxng.org/dev/search_api.html");
  await expect(source).toHaveAttribute("target", "_blank");
  await expect(page.getByRole("button", { name: "Send" })).toBeVisible();
  expect(pageErrors).toEqual([]);
});
