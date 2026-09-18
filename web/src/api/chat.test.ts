import { afterEach, describe, expect, it, vi } from "vitest";
import { streamChat } from "./chat";

describe("streamChat", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("assembles fragmented text, reasoning, and tool-call SSE deltas", async () => {
    const encoder = new TextEncoder();
    const chunks = [
      'data: {"choices":[{"delta":{"reasoning_content":"Check "}}]}\n',
      '\ndata: {"choices":[{"delta":{"content":"Hello ","reasoning_content":"the facts."}}]}\n',
      '\ndata: {"choices":[{"delta":{"content":"world","tool_calls":[{"index":0,"id":"call-1","function":{"name":"inspect","arguments":"{\\"id\\":"}}]}}]}\n\n',
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}\n\ndata: [DONE]\n\n',
    ];
    const body = new ReadableStream({
      start(controller) {
        chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    })));
    const updates: string[] = [];
    const result = await streamChat(
      "local-agent",
      [{ id: "m1", role: "user", content: "Hi", created_at: "now" }],
      new AbortController().signal,
      (update) => updates.push(update.content),
      { temperature: 0, maxTokens: 64 },
    );
    expect(result.content).toBe("Hello world");
    expect(result.reasoning).toBe("Check the facts.");
    expect(result.toolCalls).toEqual([{ id: "call-1", name: "inspect", arguments: '{"id":1}' }]);
    expect(updates.at(-1)).toBe("Hello world");
  });
});
