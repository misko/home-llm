import { ApiError } from "./client";
import type {
  AgentSource,
  AgentToolError,
  AgentToolExecution,
  ChatMessage,
} from "./types";

export const RESEARCH_TOOLSET = "standard-readonly";
export const WORKSPACE_TOOLSET = "workspace-files";
export const PYTHON_SANDBOX_TOOLSET = "python-sandbox";
const MAX_SERVER_ERROR_MESSAGE = 512;
const MAX_INSTRUCTIONS_LENGTH = 16_384;
const POLICY_BLOCK_CODES = new Set([
  "fetch_url_not_approved",
  "open_world_chain_blocked",
  "tool_not_permitted",
  "tool_requires_approval",
]);

export interface AgentTurnUpdate {
  content: string;
  reasoning: string;
  tools: AgentToolExecution[];
  sources: AgentSource[];
  completed: boolean;
}

interface AgentEvent extends Record<string, unknown> {
  type?: string;
  content?: unknown;
  reasoning?: unknown;
  call_id?: unknown;
  tool_call_id?: unknown;
  name?: unknown;
  arguments?: unknown;
  result?: unknown;
  error?: unknown;
  data?: unknown;
}

interface AgentOptions {
  temperature: number;
  maxTokens: number;
  systemPrompt?: string;
  toolset?: string;
}

function errorMessage(value: unknown): AgentToolError {
  if (typeof value === "string") return { message: value };
  if (value && typeof value === "object") {
    const item = value as Record<string, unknown>;
    return {
      code: typeof item.code === "string" ? item.code : undefined,
      message: typeof item.message === "string" ? item.message : "The tool failed.",
      retryable: typeof item.retryable === "boolean" ? item.retryable : undefined,
    };
  }
  return { message: "The tool failed." };
}

function boundedServerMessage(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const message = value.trim();
  return message ? message.slice(0, MAX_SERVER_ERROR_MESSAGE) : undefined;
}

function validationDetailMessage(value: unknown): string | undefined {
  if (!Array.isArray(value)) return boundedServerMessage(value);
  const messages: string[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") continue;
    const message = boundedServerMessage((item as Record<string, unknown>).msg);
    if (message) messages.push(message);
    if (messages.length === 3) break;
  }
  return messages.length
    ? messages.join("; ").slice(0, MAX_SERVER_ERROR_MESSAGE)
    : undefined;
}

function unsafeSourceHostname(value: string) {
  const hostname = value.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.$/, "");
  if (
    hostname === "localhost"
    || hostname.endsWith(".localhost")
    || hostname.endsWith(".local")
    || (!hostname.includes(".") && !hostname.includes(":"))
  ) return true;

  // URL parsing canonicalizes legacy numeric forms such as 127.1 and
  // 0x7f000001. Citation links do not need literal IP destinations, so reject
  // every IPv4/IPv6 literal as a conservative browser-side boundary.
  const ipv4 = hostname.split(".");
  return (ipv4.length === 4 && ipv4.every((part) => /^\d{1,3}$/.test(part)))
    || hostname.includes(":");
}

function safeWebUrl(value: unknown): string | null {
  if (typeof value !== "string") return null;
  try {
    const parsed = new URL(value);
    if (
      (parsed.protocol !== "http:" && parsed.protocol !== "https:")
      || parsed.username
      || parsed.password
      || (parsed.port !== "" && parsed.port !== "80" && parsed.port !== "443")
      || unsafeSourceHostname(parsed.hostname)
    ) return null;
    return parsed.href;
  } catch {
    return null;
  }
}

/** Extracts citation-shaped records from common search/fetch result envelopes. */
export function extractSources(result: unknown): AgentSource[] {
  const candidates: unknown[] = [];
  if (Array.isArray(result)) candidates.push(...result);
  if (result && typeof result === "object") {
    const envelope = result as Record<string, unknown>;
    candidates.push(envelope);
    for (const key of ["results", "sources", "items", "citations"]) {
      if (Array.isArray(envelope[key])) candidates.push(...envelope[key]);
    }
  }

  const sources: AgentSource[] = [];
  const seen = new Set<string>();
  for (const candidate of candidates) {
    if (!candidate || typeof candidate !== "object") continue;
    const item = candidate as Record<string, unknown>;
    const url = safeWebUrl(item.url ?? item.href ?? item.link);
    if (!url || seen.has(url)) continue;
    seen.add(url);
    const title = typeof item.title === "string" && item.title.trim()
      ? item.title.trim()
      : new URL(url).hostname;
    const rawSnippet = item.snippet ?? item.description ?? item.content;
    sources.push({
      title,
      url,
      snippet: typeof rawSnippet === "string" && rawSnippet.trim() ? rawSnippet.trim() : undefined,
    });
    if (sources.length === 12) break;
  }
  return sources;
}

function outboundMessages(messages: ChatMessage[]) {
  const outbound: Array<{ role: string; content: string | Array<Record<string, unknown>> }> = messages.map(({
    role,
    content,
    attachments,
  }) => ({
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
  return outbound;
}

/** A lazily loaded page can start halfway through a turn. Older browser records
 * may also share a timestamp, so select the newest valid user/assistant suffix
 * rather than allowing a malformed stored prefix to violate the agent contract. */
function boundedConversation(messages: ChatMessage[]) {
  const suffix: ChatMessage[] = [];
  let expected: ChatMessage["role"] = "user";
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== expected) continue;
    suffix.push(message);
    expected = expected === "user" ? "assistant" : "user";
  }
  suffix.reverse();
  while (suffix[0]?.role !== "user") suffix.shift();
  return suffix;
}

function eventPayload(raw: unknown, eventName?: string): AgentEvent | null {
  if (!raw || typeof raw !== "object") return null;
  const event = raw as AgentEvent;
  const nested = event.data && typeof event.data === "object"
    ? event.data as Record<string, unknown>
    : undefined;
  return {
    ...event,
    ...nested,
    type: typeof event.type === "string" ? event.type : eventName,
  };
}

export async function streamAgentTurn(
  messages: ChatMessage[],
  signal: AbortSignal,
  onUpdate: (update: AgentTurnUpdate) => void,
  options: AgentOptions,
): Promise<AgentTurnUpdate> {
  const response = await fetch("/api/v1/agent/turns", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      messages: outboundMessages(boundedConversation(messages)),
      instructions: options.systemPrompt?.trim().slice(0, MAX_INSTRUCTIONS_LENGTH) || undefined,
      toolset: options.toolset ?? RESEARCH_TOOLSET,
      temperature: options.temperature,
      max_tokens: options.maxTokens,
      stream: true,
    }),
    signal,
  });
  if (!response.ok || !response.body) {
    let message = `Agent request failed (${response.status})`;
    let code = "agent_failed";
    let retryable = response.status >= 500;
    try {
      const payload = await response.json() as {
        error?: { code?: string; message?: string; retryable?: boolean };
        detail?: unknown;
      };
      message = boundedServerMessage(payload.error?.message)
        ?? validationDetailMessage(payload.detail)
        ?? message;
      code = payload.error?.code ?? code;
      retryable = payload.error?.retryable ?? retryable;
    } catch {
      // Preserve the HTTP status when an intermediary returns a non-JSON body.
    }
    throw new ApiError(response.status, code, message, retryable);
  }

  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  const tools = new Map<string, AgentToolExecution>();
  const sources = new Map<string, AgentSource>();
  let content = "";
  let reasoning = "";
  let completed = false;
  let buffer = "";

  const snapshot = (): AgentTurnUpdate => ({
    content,
    reasoning,
    tools: [...tools.values()],
    sources: [...sources.values()],
    completed,
  });

  const consumeEvent = (raw: unknown, eventName?: string) => {
    const event = eventPayload(raw, eventName);
    if (!event) return;
    const type = event.type;
    if (type === "assistant.delta") {
      if (typeof event.content === "string") content += event.content;
      if (typeof event.reasoning === "string") reasoning += event.reasoning;
    } else if (type === "assistant.message" && typeof event.content === "string") {
      content = event.content;
    } else if (type === "tool.started" || type === "tool.call") {
      const idValue = event.call_id ?? event.tool_call_id;
      const id = typeof idValue === "string" ? idValue : `tool-${tools.size + 1}`;
      tools.set(id, {
        id,
        name: typeof event.name === "string" ? event.name : "tool",
        arguments: event.arguments,
        status: "running",
      });
    } else if (type === "tool.completed" || type === "tool.result") {
      const idValue = event.call_id ?? event.tool_call_id;
      const id = typeof idValue === "string" ? idValue : `tool-${tools.size + 1}`;
      const prior = tools.get(id);
      tools.set(id, {
        id,
        name: typeof event.name === "string" ? event.name : prior?.name ?? "tool",
        arguments: event.arguments ?? prior?.arguments,
        status: "completed",
        result: event.result,
      });
      for (const source of extractSources(event.result)) {
        const priorSource = sources.get(source.url);
        sources.set(source.url, {
          ...priorSource,
          ...source,
          snippet: source.snippet ?? priorSource?.snippet,
        });
      }
    } else if (type === "tool.failed") {
      const idValue = event.call_id ?? event.tool_call_id;
      const id = typeof idValue === "string" ? idValue : `tool-${tools.size + 1}`;
      const prior = tools.get(id);
      const error = errorMessage(event.error);
      tools.set(id, {
        id,
        name: typeof event.name === "string" ? event.name : prior?.name ?? "tool",
        arguments: event.arguments ?? prior?.arguments,
        status: error.code && POLICY_BLOCK_CODES.has(error.code) ? "blocked" : "failed",
        error,
      });
    } else if (type === "turn.completed" || type === "run.completed") {
      completed = true;
    } else if (type === "turn.failed" || type === "error") {
      const failure = errorMessage(event.error ?? event);
      throw new ApiError(502, failure.code ?? "agent_failed", failure.message, failure.retryable);
    } else {
      return;
    }
    onUpdate(snapshot());
  };

  const consumeFrame = (frame: string) => {
    let eventName: string | undefined;
    const data: string[] = [];
    for (const line of frame.split(/\r?\n/)) {
      if (!line || line.startsWith(":")) continue;
      const separator = line.indexOf(":");
      const field = separator >= 0 ? line.slice(0, separator) : line;
      const value = separator >= 0 ? line.slice(separator + 1).replace(/^ /, "") : "";
      if (field === "event") eventName = value;
      if (field === "data") data.push(value);
    }
    if (!data.length || data.join("\n").trim() === "[DONE]") return;
    let payload: unknown;
    try {
      payload = JSON.parse(data.join("\n"));
    } catch {
      // Ignore malformed heartbeat/proxy frames while preserving valid later events.
      return;
    }
    consumeEvent(payload, eventName);
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) consumeFrame(frame);
    if (done) break;
  }
  if (buffer.trim()) consumeFrame(buffer);
  if (!completed) {
    throw new ApiError(
      502,
      "agent_stream_incomplete",
      "The agent stream ended before the turn completed.",
      true,
    );
  }
  return snapshot();
}

export async function approveWorkspaceWrite(proposalId: string) {
  const response = await fetch(`/api/v1/agent/workspace/proposals/${encodeURIComponent(proposalId)}/approve`, {
    method: "POST",
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    let message = `Write approval failed (${response.status})`;
    try {
      const payload = await response.json() as { error?: { message?: string } };
      message = boundedServerMessage(payload.error?.message) ?? message;
    } catch { /* Keep the HTTP-status detail. */ }
    throw new ApiError(response.status, "workspace_approval_failed", message, response.status >= 500);
  }
  return response.json() as Promise<{ path: string; sha256: string; bytes_written: number }>;
}
