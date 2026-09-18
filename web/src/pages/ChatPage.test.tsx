import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { portfolioFixture, runtimeFixture } from "../test/fixtures";
import { ChatPage } from "./ChatPage";

function agentResponse() {
  const encoder = new TextEncoder();
  const events = [
    { type: "tool.started", call_id: "search-1", name: "web_search", arguments: { query: "current local models" } },
    {
      type: "tool.completed",
      call_id: "search-1",
      name: "web_search",
      result: {
        results: [{
          title: "Local model report",
          url: "https://example.com/report",
          snippet: "A deterministic research result.",
        }],
      },
    },
    { type: "tool.started", call_id: "fetch-1", name: "web_fetch", arguments: { url: "https://example.com/report" } },
    {
      type: "tool.completed",
      call_id: "fetch-1",
      name: "web_fetch",
      result: {
        url: "https://example.com/report",
        title: "Local model report",
        text: "A bounded extract.",
        truncated: true,
        extracted_characters: 15_516,
      },
    },
    { type: "tool.started", call_id: "calc-1", name: "calculator", arguments: { expression: "2 + 2" } },
    {
      type: "tool.completed",
      call_id: "calc-1",
      name: "calculator",
      result: { expression: "2 + 2", result: 4 },
    },
    { type: "assistant.delta", content: "A cited answer." },
    { type: "turn.completed", run_id: "run-1", rounds: 2 },
  ];
  return new Response(new ReadableStream({
    start(controller) {
      for (const event of events) controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      controller.close();
    },
  }), { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function rawChatResponse() {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"Raw answer."}}]}\n\n'));
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  }), { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

describe("ChatPage research tools", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => vi.unstubAllGlobals());

  it("runs the read-only toolset and renders execution evidence and citations", async () => {
    let agentRequest: Record<string, unknown> | undefined;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path === "/api/v1/models") {
        return new Response(JSON.stringify(portfolioFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/runtime") {
        return new Response(JSON.stringify(runtimeFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/agent/turns") {
        agentRequest = JSON.parse(String(init?.body));
        return agentResponse();
      }
      throw new Error(`unexpected request ${path}`);
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(<QueryClientProvider client={client}><ChatPage /></QueryClientProvider>);

    const toolSwitch = await screen.findByRole("switch", { name: "Research tools" });
    await waitFor(() => expect(toolSwitch).toBeEnabled());
    await userEvent.click(screen.getByRole("button", { name: "Generation settings" }));
    const maximumOutput = screen.getByLabelText("Maximum output tokens");
    expect(maximumOutput).toHaveValue(32_000);
    expect(maximumOutput).toHaveAttribute("max", "32768");
    expect(screen.getByText(/32,000 is a ceiling, not a target/)).toHaveTextContent("8,192-token context");
    fireEvent.change(maximumOutput, { target: { value: "100000" } });
    expect(maximumOutput).toHaveValue(32_768);
    fireEvent.change(maximumOutput, { target: { value: "32000" } });
    const instructions = screen.getByLabelText("System prompt");
    expect(instructions).toHaveAttribute("maxlength", "16384");
    fireEvent.change(instructions, { target: { value: "x".repeat(17_000) } });
    expect(instructions).toHaveValue("x".repeat(16_384));
    await userEvent.click(screen.getByRole("button", { name: "Apply settings" }));
    expect(toolSwitch).toBeChecked();
    expect(screen.getByText("Web research plus the permissions selected in settings")).toBeVisible();
    await userEvent.type(screen.getByLabelText("Message the active model"), "Find a current source");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("A cited answer.")).toBeInTheDocument();
    const toolCard = screen.getByTestId("tool-search-1");
    expect(within(toolCard).getByText("Web Search")).toBeInTheDocument();
    expect(within(toolCard).getByText("Complete")).toBeInTheDocument();
    const fetchCard = screen.getByTestId("tool-fetch-1");
    expect(within(fetchCard).getByText("Bounded extract")).toBeInTheDocument();
    expect(within(fetchCard).getByText("Bounded public extract · 15,516 readable characters")).toBeInTheDocument();
    expect(within(screen.getByTestId("tool-calc-1")).getByText("Runs locally · no network or writes")).toBeInTheDocument();
    const source = screen.getByRole("link", { name: /Local model report/ });
    expect(source).toHaveAttribute("href", "https://example.com/report");
    expect(source).toHaveAttribute("target", "_blank");
    expect(screen.getByText("A deterministic research result.")).toBeInTheDocument();
    await waitFor(() => expect(agentRequest).toMatchObject({
      toolset: "assistant-tools",
      enabled_tools: ["web_search", "web_fetch", "calculator", "current_time", "workspace_list", "workspace_read", "workspace_write_proposal", "python_sandbox", "openrouter_delegate"],
      temperature: 0,
      max_tokens: 32_000,
      instructions: "x".repeat(16_384),
      messages: [{ role: "user", content: "Find a current source" }],
    }));
    client.clear();
  });

  it("uses raw chat and never calls the agent endpoint while tools remain disabled", async () => {
    let rawRequest: Record<string, unknown> | undefined;
    let agentCalls = 0;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path === "/api/v1/models") {
        return new Response(JSON.stringify(portfolioFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/runtime") {
        return new Response(JSON.stringify(runtimeFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/v1/chat/completions") {
        rawRequest = JSON.parse(String(init?.body));
        return rawChatResponse();
      }
      if (path === "/api/v1/agent/turns") {
        agentCalls += 1;
        throw new Error("agent endpoint must not be called");
      }
      throw new Error(`unexpected request ${path}`);
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(<QueryClientProvider client={client}><ChatPage /></QueryClientProvider>);

    const toolSwitch = await screen.findByRole("switch", { name: "Research tools" });
    await waitFor(() => expect(toolSwitch).toBeEnabled());
    expect(toolSwitch).toBeChecked();
    await userEvent.click(toolSwitch);
    expect(toolSwitch).not.toBeChecked();
    await userEvent.type(screen.getByLabelText("Message the active model"), "Answer without internet");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Raw answer.")).toBeInTheDocument();
    expect(agentCalls).toBe(0);
    expect(rawRequest).toMatchObject({
      model: "local-fast",
      messages: [{ role: "user", content: "Answer without internet" }],
      max_tokens: 32_000,
      stream: true,
    });
    client.clear();
  });

  it("keeps previous chats and settings behind compact chat controls", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/v1/models") {
        return new Response(JSON.stringify(portfolioFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/runtime") {
        return new Response(JSON.stringify(runtimeFixture()), { headers: { "Content-Type": "application/json" } });
      }
      throw new Error(`unexpected request ${path}`);
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(<QueryClientProvider client={client}><ChatPage /></QueryClientProvider>);

    const history = await screen.findByRole("complementary", { name: "Chat history" });
    const previousChats = screen.getByRole("button", { name: "Previous chats" });
    expect(previousChats).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(previousChats);
    expect(previousChats).toHaveAttribute("aria-expanded", "true");
    expect(history).toHaveClass("open");
    await userEvent.click(screen.getByRole("button", { name: "Close previous chats" }));
    expect(history).not.toHaveClass("open");

    await userEvent.click(screen.getByRole("button", { name: "Open settings" }));
    expect(screen.getByRole("dialog", { name: "Generation settings" })).toBeInTheDocument();
    client.clear();
  });

  it("accepts only supported images and caps the whole conversation at four", async () => {
    let agentRequest: Record<string, unknown> | undefined;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path === "/api/v1/models") {
        return new Response(JSON.stringify(portfolioFixture(true)), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/runtime") {
        return new Response(JSON.stringify(runtimeFixture({
          deployment_id: "muse-glimmer-30b-4090-8k",
          public_alias: "local-agent",
          model_name: "Muse Glimmer 30B",
        })), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/agent/turns") {
        agentRequest = JSON.parse(String(init?.body));
        return agentResponse();
      }
      throw new Error(`unexpected request ${path}`);
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(<QueryClientProvider client={client}><ChatPage /></QueryClientProvider>);

    const imageInput = await screen.findByLabelText("Image") as HTMLInputElement;
    await waitFor(() => expect(imageInput).toBeEnabled());
    expect(imageInput).toHaveAttribute("accept", "image/png,image/jpeg,image/webp");
    fireEvent.change(imageInput, {
      target: { files: [new File(["<svg/>"] , "unsafe.svg", { type: "image/svg+xml" })] },
    });
    expect(await screen.findByRole("alert")).toHaveTextContent("PNG, JPEG, or WebP");

    const images = Array.from({ length: 5 }, (_, index) =>
      new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], `image-${index + 1}.png`, { type: "image/png" })
    );
    fireEvent.change(imageInput, { target: { files: images } });
    await screen.findByText("image-4.png");
    expect(screen.queryByText("image-5.png")).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("at most four images");

    await userEvent.type(screen.getByLabelText("Message the active model"), "Inspect these images");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("A cited answer.");
    await waitFor(() => expect(imageInput).toBeDisabled());

    const messages = agentRequest?.messages as Array<{ role: string; content: unknown[] }>;
    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({ role: "user" });
    expect(messages[0].content).toHaveLength(5);
    expect(messages[0].content.filter((part) => (part as { type?: string }).type === "image_url")).toHaveLength(4);
    client.clear();
  });

  it("rolls back an early failed exchange, restores its input, and retries with valid roles", async () => {
    const agentRequests: Array<Record<string, unknown>> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path === "/api/v1/models") {
        return new Response(JSON.stringify(portfolioFixture(true)), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/runtime") {
        return new Response(JSON.stringify(runtimeFixture({
          deployment_id: "muse-glimmer-30b-4090-8k",
          public_alias: "local-agent",
          model_name: "Muse Glimmer 30B",
        })), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/agent/turns") {
        agentRequests.push(JSON.parse(String(init?.body)));
        if (agentRequests.length === 1) {
          return new Response(JSON.stringify({
            error: { code: "model_unavailable", message: "The model is temporarily unavailable", retryable: true },
          }), { status: 503, headers: { "Content-Type": "application/json" } });
        }
        return agentResponse();
      }
      throw new Error(`unexpected request ${path}`);
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(<QueryClientProvider client={client}><ChatPage /></QueryClientProvider>);

    const imageInput = await screen.findByLabelText("Image") as HTMLInputElement;
    await waitFor(() => expect(imageInput).toBeEnabled());
    const image = new File(
      [new Uint8Array([0x89, 0x50, 0x4e, 0x47])],
      "retry.png",
      { type: "image/png" },
    );
    fireEvent.change(imageInput, { target: { files: [image] } });
    await screen.findByText("retry.png");
    const composer = screen.getByLabelText("Message the active model");
    await userEvent.type(composer, "Retry this exact request");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("temporarily unavailable");
    await waitFor(() => expect(composer).toHaveValue("Retry this exact request"));
    expect(screen.getByText("retry.png")).toBeInTheDocument();
    expect(agentRequests).toHaveLength(1);

    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("A cited answer.")).toBeInTheDocument();
    expect(agentRequests).toHaveLength(2);
    const retriedMessages = agentRequests[1].messages as Array<{ role: string; content: unknown }>;
    expect(retriedMessages.map((message) => message.role)).toEqual(["user"]);
    expect(retriedMessages[0].content).toEqual([
      { type: "text", text: "Retry this exact request" },
      expect.objectContaining({ type: "image_url" }),
    ]);
    client.clear();
  });

  it("creates multiple saved chats and resumes the selected history after remount", async () => {
    const rawRequests: Array<Record<string, unknown>> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path === "/api/v1/models") {
        return new Response(JSON.stringify(portfolioFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/runtime") {
        return new Response(JSON.stringify(runtimeFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/v1/chat/completions") {
        rawRequests.push(JSON.parse(String(init?.body)));
        return rawChatResponse();
      }
      throw new Error(`unexpected request ${path}`);
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    const view = render(<QueryClientProvider client={client}><ChatPage /></QueryClientProvider>);

    const composer = await screen.findByLabelText("Message the active model");
    await userEvent.click(await screen.findByRole("switch", { name: "Research tools" }));
    await userEvent.type(composer, "First saved conversation");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Raw answer.")).toBeInTheDocument();

    const history = screen.getByRole("complementary", { name: "Chat history" });
    await userEvent.click(within(history).getByRole("button", { name: "New chat" }));
    expect(screen.queryByText("First saved conversation", { selector: ".message-content" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("switch", { name: "Research tools" }));
    await userEvent.type(composer, "Second saved conversation");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(rawRequests).toHaveLength(2));

    await userEvent.click(within(history).getByRole("button", { name: /^First saved conversation/ }));
    expect(screen.getByText("First saved conversation", { selector: ".message-content" })).toBeInTheDocument();
    await waitFor(() => {
      const saved = JSON.parse(localStorage.getItem("llm-lab.chat-history.v1") ?? "[]") as unknown[];
      expect(saved).toHaveLength(2);
    });

    view.unmount();
    const resumedClient = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(<QueryClientProvider client={resumedClient}><ChatPage /></QueryClientProvider>);
    const resumedComposer = await screen.findByLabelText("Message the active model");
    expect(await screen.findByText("First saved conversation", { selector: ".message-content" })).toBeInTheDocument();
    await userEvent.type(resumedComposer, "Continue the first chat");
    await userEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(rawRequests).toHaveLength(3));
    expect(rawRequests[2].messages).toMatchObject([
      { role: "user", content: "First saved conversation" },
      { role: "assistant", content: "Raw answer." },
      { role: "user", content: "Continue the first chat" },
    ]);
    client.clear();
    resumedClient.clear();
  });

  it("stops a session at 64 displayed messages with an actionable reset", async () => {
    let rawCalls = 0;
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/api/v1/models") {
        return new Response(JSON.stringify(portfolioFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/runtime") {
        return new Response(JSON.stringify(runtimeFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/v1/chat/completions") {
        rawCalls += 1;
        return rawChatResponse();
      }
      throw new Error(`unexpected request ${path}`);
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(<QueryClientProvider client={client}><ChatPage /></QueryClientProvider>);

    const composer = await screen.findByLabelText("Message the active model");
    await userEvent.click(await screen.findByRole("switch", { name: "Research tools" }));
    const form = composer.closest("form");
    expect(form).not.toBeNull();
    for (let turn = 1; turn <= 32; turn += 1) {
      fireEvent.change(composer, { target: { value: `Turn ${turn}` } });
      fireEvent.submit(form!);
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
      expect(form!.querySelector("button[type='submit']")).not.toBeNull();
    }
    expect(rawCalls).toBe(32);

    fireEvent.change(composer, { target: { value: "One turn too many" } });
    fireEvent.submit(form!);
    expect(await screen.findByRole("alert")).toHaveTextContent("64-message limit");
    expect(screen.getByRole("button", { name: "Clear" })).toBeVisible();
    expect(composer).toHaveValue("One turn too many");
    expect(rawCalls).toBe(32);
    client.clear();
  });
});
