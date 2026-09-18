import type { ChatMessage, ToolCall } from "./types";
import { ApiError } from "./client";

interface StreamDelta {
  content?: string | null;
  reasoning_content?: string | null;
  tool_calls?: Array<{
    index?: number;
    id?: string;
    function?: { name?: string; arguments?: string };
  }>;
}

export interface StreamUpdate {
  content: string;
  reasoning: string;
  toolCalls: ToolCall[];
}

export async function streamChat(
  alias: string,
  messages: ChatMessage[],
  signal: AbortSignal,
  onUpdate: (update: StreamUpdate) => void,
  options: { temperature: number; maxTokens: number; systemPrompt?: string },
): Promise<StreamUpdate> {
  const outbound: Array<{ role: string; content: string | Array<Record<string, unknown>> }> = messages.map(({ role, content, attachments }) => ({
    role,
    content: attachments?.length
      ? [
          { type: "text", text: content },
          ...attachments.map((attachment) => ({
            type: "image_url",
            image_url: { url: attachment.data_url },
          })),
        ]
      : content,
  }));
  if (options.systemPrompt?.trim()) {
    outbound.unshift({ role: "system", content: options.systemPrompt.trim() });
  }
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      model: alias,
      messages: outbound,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stream: true,
    }),
    signal,
  });
  if (!response.ok || !response.body) {
    let message = `Chat request failed (${response.status})`;
    try {
      const payload = await response.json() as { error?: { message?: string; code?: string } };
      message = payload.error?.message ?? message;
      throw new ApiError(response.status, payload.error?.code ?? "chat_failed", message, response.status >= 500);
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(response.status, "chat_failed", message, response.status >= 500);
    }
  }

  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let buffer = "";
  let content = "";
  let reasoning = "";
  const tools = new Map<number, ToolCall>();

  const consume = (line: string) => {
    if (!line.startsWith("data:")) return;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") return;
    const payload = JSON.parse(data) as { choices?: Array<{ delta?: StreamDelta }> };
    const delta = payload.choices?.[0]?.delta;
    if (typeof delta?.content === "string") content += delta.content;
    if (typeof delta?.reasoning_content === "string") reasoning += delta.reasoning_content;
    for (const item of delta?.tool_calls ?? []) {
      const index = item.index ?? tools.size;
      const prior = tools.get(index) ?? { id: item.id ?? `tool-${index}`, name: "", arguments: "" };
      tools.set(index, {
        id: item.id ?? prior.id,
        name: prior.name + (item.function?.name ?? ""),
        arguments: prior.arguments + (item.function?.arguments ?? ""),
      });
    }
    onUpdate({ content, reasoning, toolCalls: [...tools.values()] });
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";
    for (const line of lines) consume(line);
    if (done) break;
  }
  if (buffer) consume(buffer);
  return { content, reasoning, toolCalls: [...tools.values()] };
}
