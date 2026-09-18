import { afterEach, describe, expect, it, vi } from "vitest";
import { extractSources, streamAgentTurn } from "./agent";

function eventStream(chunks: string[]) {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

describe("streamAgentTurn", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("assembles fragmented agent events, tool state, and safe sources", async () => {
    const chunks = [
      "id: 1\nevent: turn.started\ndata: {\"schema_version\":1,\"run_id\":\"run-1\",\"sequence\":1,\"type\":\"turn.started\",\"model\":\"local-fast\",\"toolset\":\"standard-readonly\",\"tools\":[\"web_search\",\"web_fetch\",\"calculator\",\"current_time\"]}\n\n",
      ": keepalive\n\nid: 2\nevent: tool.started\ndata: {\"schema_version\":1,\"run_id\":\"run-1\",\"sequence\":2,\"type\":\"tool.started\",\"call_id\":\"search-1\",\"name\":\"web_search\",\"arguments\":{\"query\":\"local LLMs\"},\"round\":1}\n\n",
      "data: not-json\n\n",
      "id: 3\nevent: tool.completed\ndata: {\"schema_version\":1,\"run_id\":\"run-1\",\"sequence\":3,\"type\":\"tool.completed\",\"call_id\":\"search-1\",\"name\":\"web_search\",\"result\":{\"results\":[{\"title\":\"Model guide\",\"url\":\"https://example.com/models\",\"snippet\":\"A useful guide\"},{\"title\":\"Unsafe\",\"url\":\"javascript:alert(1)\"}]},\"round\":1,\"duration_ms\":12.5}\n",
      "\nid: 4\nevent: assistant.delta\ndata: {\"schema_version\":1,\"run_id\":\"run-1\",\"sequence\":4,\"type\":\"assistant.delta\",\"content\":\"I found one source.\",\"reasoning\":\"I checked the source.\",\"round\":2}\n\nid: 5\nevent: turn.completed\ndata: {\"schema_version\":1,\"run_id\":\"run-1\",\"sequence\":5,\"type\":\"turn.completed\",\"model\":\"local-fast\",\"rounds\":2,\"finish_reason\":\"stop\",\"usage\":{\"prompt_tokens\":27,\"completion_tokens\":18,\"total_tokens\":45}}\n\n",
    ];
    let requestBody: Record<string, unknown> | undefined;
    vi.stubGlobal("fetch", vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      requestBody = JSON.parse(String(init?.body));
      return new Response(eventStream(chunks), {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      });
    }));

    const updates: string[] = [];
    const result = await streamAgentTurn(
      [{ id: "m1", role: "user", content: "Research this", created_at: "now" }],
      new AbortController().signal,
      (update) => updates.push(update.content),
      { temperature: 0.2, maxTokens: 256, systemPrompt: "Cite sources." },
    );

    expect(requestBody).toMatchObject({
      instructions: "Cite sources.",
      toolset: "standard-readonly",
      temperature: 0.2,
      max_tokens: 256,
      stream: true,
      messages: [{ role: "user", content: "Research this" }],
    });
    expect(requestBody?.messages).not.toContainEqual(expect.objectContaining({ role: "system" }));
    expect(result).toEqual({
      content: "I found one source.",
      reasoning: "I checked the source.",
      completed: true,
      tools: [{
        id: "search-1",
        name: "web_search",
        arguments: { query: "local LLMs" },
        status: "completed",
        result: {
          results: [
            { title: "Model guide", url: "https://example.com/models", snippet: "A useful guide" },
            { title: "Unsafe", url: "javascript:alert(1)" },
          ],
        },
      }],
      sources: [{ title: "Model guide", url: "https://example.com/models", snippet: "A useful guide" }],
    });
    expect(updates).toEqual(["", "", "I found one source.", "I found one source."]);
  });

  it("accepts compatibility aliases and exposes tool failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(eventStream([
      "data: {\"type\":\"tool.call\",\"data\":{\"tool_call_id\":\"fetch-1\",\"name\":\"web.fetch\",\"arguments\":{\"url\":\"https://example.com\"}}}\n\n",
      "data: {\"type\":\"tool.failed\",\"tool_call_id\":\"fetch-1\",\"name\":\"web.fetch\",\"error\":{\"code\":\"timeout\",\"message\":\"Fetch timed out\",\"retryable\":true}}\n\n",
      "data: {\"type\":\"tool.call\",\"data\":{\"tool_call_id\":\"fetch-2\",\"name\":\"web.fetch\",\"arguments\":{\"url\":\"https://example.com/again\"}}}\n\n",
      "data: {\"type\":\"tool.failed\",\"tool_call_id\":\"fetch-2\",\"name\":\"web.fetch\",\"error\":{\"code\":\"open_world_chain_blocked\",\"message\":\"Further open-world calls are blocked\",\"retryable\":false}}\n\n",
      "data: {\"type\":\"assistant.message\",\"content\":\"I could not fetch that page.\"}\n\n",
      "data: {\"type\":\"run.completed\"}\n\n",
    ]), { status: 200 })));

    const result = await streamAgentTurn(
      [{ id: "m1", role: "user", content: "Fetch", created_at: "now" }],
      new AbortController().signal,
      () => undefined,
      { temperature: 0, maxTokens: 64 },
    );

    expect(result.content).toBe("I could not fetch that page.");
    expect(result.tools[0]).toMatchObject({
      id: "fetch-1",
      name: "web.fetch",
      status: "failed",
      error: { code: "timeout", message: "Fetch timed out", retryable: true },
    });
    expect(result.tools[1]).toMatchObject({
      id: "fetch-2",
      status: "blocked",
      error: { code: "open_world_chain_blocked" },
    });
  });

  it("rejects structured server failures and incomplete streams", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        error: { code: "toolset_unavailable", message: "Search is offline", retryable: true },
      }), { status: 503, headers: { "Content-Type": "application/json" } }))
      .mockResolvedValueOnce(new Response(eventStream([
        "data: {\"type\":\"assistant.delta\",\"content\":\"Partial\"}\n\n",
      ]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const message = [{ id: "m1", role: "user" as const, content: "Search", created_at: "now" }];
    const options = { temperature: 0, maxTokens: 64 };

    await expect(streamAgentTurn(message, new AbortController().signal, () => undefined, options))
      .rejects.toMatchObject({ status: 503, code: "toolset_unavailable", message: "Search is offline", retryable: true });
    await expect(streamAgentTurn(message, new AbortController().signal, () => undefined, options))
      .rejects.toMatchObject({ status: 502, code: "agent_stream_incomplete", retryable: true });
  });

  it("uses only bounded validation messages from FastAPI detail arrays", async () => {
    const reflectedSecret = "do-not-reflect-this-input";
    const longMessage = "x".repeat(700);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: [{
        type: "value_error",
        loc: ["body", "messages", 0, "content"],
        msg: longMessage,
        input: reflectedSecret,
      }],
    }), { status: 422, headers: { "Content-Type": "application/json" } })));

    const pending = streamAgentTurn(
      [{ id: "m1", role: "user", content: "Search", created_at: "now" }],
      new AbortController().signal,
      () => undefined,
      { temperature: 0, maxTokens: 64 },
    );

    await expect(pending).rejects.toMatchObject({
      status: 422,
      code: "agent_failed",
      message: "x".repeat(512),
      retryable: false,
    });
    await expect(pending).rejects.not.toMatchObject({ message: expect.stringContaining(reflectedSecret) });
  });

  it("passes cancellation through to the streaming request", async () => {
    vi.stubGlobal("fetch", vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
      })
    ));
    const controller = new AbortController();
    const pending = streamAgentTurn(
      [{ id: "m1", role: "user", content: "Search", created_at: "now" }],
      controller.signal,
      () => undefined,
      { temperature: 0, maxTokens: 64 },
    );
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("extractSources", () => {
  it("deduplicates links and rejects non-web protocols", () => {
    expect(extractSources({
      sources: [
        { title: "One", url: "https://example.com/a" },
        { title: "Duplicate", href: "https://example.com/a" },
        { title: "Local file", url: "file:///etc/passwd" },
        { title: "Credentials", url: "https://user:secret@example.com/private" },
        { title: "Loopback", url: "http://127.0.0.1/admin" },
        { title: "Abbreviated loopback", url: "http://127.1/admin" },
        { title: "Public literal IP", url: "https://8.8.8.8/dns" },
        { title: "Private IPv4", url: "http://192.168.1.20/status" },
        { title: "Loopback IPv6", url: "http://[::1]/admin" },
        { title: "Local hostname", url: "http://nas.local/files" },
        { title: "Non-web port", url: "https://example.com:8443/admin" },
      ],
    })).toEqual([{ title: "One", url: "https://example.com/a" }]);
  });
});
