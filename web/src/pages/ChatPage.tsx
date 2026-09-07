import { useMemo, useRef, useState, type FormEvent } from "react";
import {
  Bot,
  BrainCircuit,
  CheckCircle2,
  Code2,
  Copy,
  ExternalLink,
  ImagePlus,
  LoaderCircle,
  Paperclip,
  Send,
  Settings2,
  ShieldCheck,
  Square,
  Trash2,
  User,
  Wrench,
  X,
} from "lucide-react";
import { RESEARCH_TOOLSET, streamAgentTurn } from "../api/agent";
import { streamChat } from "../api/chat";
import { createClientId } from "../api/id";
import type { AgentSource, AgentToolExecution, ChatAttachment, ChatMessage, ToolCall } from "../api/types";
import { usePortfolio, useRuntime } from "../hooks/useConsoleData";
import { EmptyState } from "../components/EmptyState";
import { Modal } from "../components/Modal";

const prompts = [
  ["Review a piece of code", "Review this code for correctness, edge cases, and maintainability:\n\n"],
  ["Draft a structured plan", "Create a concrete implementation plan for "],
  ["Compare technical options", "Compare these technical options and recommend one: "],
] as const;
const supportedImageTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
const maximumConversationImages = 4;
const maximumConversationMessages = 64;
const maximumInstructionsLength = 16_384;

function now() {
  return new Date().toISOString();
}

function ToolCallCard({ tool }: { tool: ToolCall }) {
  let formatted = tool.arguments;
  try { formatted = JSON.stringify(JSON.parse(tool.arguments), null, 2); } catch { /* partial stream */ }
  return (
    <details className="tool-call-card">
      <summary><Wrench size={15} /> {tool.name || "Tool call"}</summary>
      <pre>{formatted || "Waiting for arguments…"}</pre>
    </details>
  );
}

function formatValue(value: unknown) {
  if (typeof value === "string") {
    try { return JSON.stringify(JSON.parse(value), null, 2); } catch { return value; }
  }
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}

function toolLabel(name: string) {
  return name.replace(/[._-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function ToolExecutionCard({ tool }: { tool: AgentToolExecution }) {
  const statusLabel = tool.status === "running" ? "Running" : tool.status === "completed" ? "Complete" : "Failed";
  const disclosure = tool.name === "calculator" || tool.name === "current_time"
    ? "Runs locally · no network or writes"
    : tool.name === "web_search" || tool.name === "web_fetch"
      ? "Sends queries and requested public pages to the internet · no writes"
      : "Read-only tool · no writes";
  return (
    <details className={"tool-execution-card " + tool.status} open={tool.status === "running"} data-testid={"tool-" + tool.id}>
      <summary>
        <span className="tool-icon"><Wrench size={14} /></span>
        <span className="tool-identity">
          <strong>{toolLabel(tool.name)}</strong>
          <small>{disclosure}</small>
        </span>
        <span className={"tool-status " + tool.status}>
          {tool.status === "running" ? <LoaderCircle className="spin" size={13} /> : tool.status === "completed" ? <CheckCircle2 size={13} /> : <X size={13} />}
          {statusLabel}
        </span>
      </summary>
      <div className="tool-execution-detail">
        {tool.arguments !== undefined && <div><span>Arguments</span><pre>{formatValue(tool.arguments)}</pre></div>}
        {tool.status === "completed" && tool.result !== undefined && <div><span>Result</span><pre>{formatValue(tool.result)}</pre></div>}
        {tool.error && <div className="tool-error"><span>{tool.error.code ?? "Tool error"}</span><p>{tool.error.message}</p></div>}
      </div>
    </details>
  );
}

function SourceList({ sources }: { sources: AgentSource[] }) {
  if (!sources.length) return null;
  return (
    <section className="source-list" aria-label="Sources">
      <h3>Sources</h3>
      <ol>
        {sources.map((source) => (
          <li key={source.url}>
            <a href={source.url} target="_blank" rel="noreferrer">
              <span><strong>{source.title}</strong>{source.snippet && <small>{source.snippet}</small>}</span>
              <ExternalLink size={13} />
            </a>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function ChatPage() {
  const runtime = useRuntime();
  const portfolio = usePortfolio();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [temperature, setTemperature] = useState(0);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [toolsEnabled, setToolsEnabled] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const activeAlias = runtime.data?.public_alias;
  const activeDeployment = runtime.data?.deployment_id;
  const activeModel = useMemo(() => portfolio.data?.models.find((model) =>
    model.deployments.some((deployment) => deployment.id === activeDeployment)
  ), [portfolio.data, activeDeployment]);
  const supportsImages = activeModel?.modalities.includes("image") ?? false;
  const supportsTools = activeModel?.capabilities.includes("tools") ?? false;
  const conversationImageCount = messages.reduce(
    (count, message) => count + (message.attachments?.length ?? 0),
    0,
  );
  const remainingImageSlots = Math.max(
    0,
    maximumConversationImages - conversationImageCount - attachments.length,
  );

  async function attachFiles(files: FileList | null) {
    if (!files || !supportsImages) return;
    if (remainingImageSlots === 0) {
      setError("A conversation can include at most four images.");
      return;
    }
    const candidates = [...files];
    const selected = candidates.slice(0, remainingImageSlots);
    if (candidates.length > remainingImageSlots) {
      setError("A conversation can include at most four images.");
    }
    const accepted: ChatAttachment[] = [];
    for (const file of selected) {
      if (!supportedImageTypes.has(file.type) || file.size > 10 * 1024 * 1024) {
        setError("Images must be PNG, JPEG, or WebP files no larger than 10 MiB.");
        continue;
      }
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
      accepted.push({ id: createClientId(), name: file.name, mime_type: file.type, size: file.size, data_url: dataUrl });
    }
    setAttachments((current) => {
      const available = Math.max(
        0,
        maximumConversationImages - conversationImageCount - current.length,
      );
      return [...current, ...accepted.slice(0, available)];
    });
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const content = draft.trim();
    if ((!content && !attachments.length) || !activeAlias || !runtime.data?.ready || streaming) return;
    if (messages.length + 2 > maximumConversationMessages) {
      setError("This conversation reached the 64-message limit. Clear the session to start a new one.");
      return;
    }
    const submittedDraft = draft;
    const submittedAttachments = attachments;
    const userMessage: ChatMessage = {
      id: createClientId(), role: "user", content, attachments: submittedAttachments, created_at: now(), deployment_id: activeDeployment ?? undefined,
    };
    const assistantId = createClientId();
    const assistant: ChatMessage = {
      id: assistantId, role: "assistant", content: "", created_at: now(), deployment_id: activeDeployment ?? undefined,
    };
    const requestMessages = [...messages, userMessage];
    setMessages([...requestMessages, assistant]);
    setDraft("");
    setAttachments([]);
    setError(null);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    let assistantHasEvidence = false;
    try {
      if (toolsEnabled && supportsTools) {
        await streamAgentTurn(requestMessages, controller.signal, (update) => {
          assistantHasEvidence ||= Boolean(update.content || update.tools.length || update.sources.length);
          setMessages((current) => current.map((message) => message.id === assistantId
            ? { ...message, content: update.content, tool_executions: update.tools, sources: update.sources }
            : message));
        }, { temperature, maxTokens, systemPrompt, toolset: RESEARCH_TOOLSET });
      } else {
        await streamChat(activeAlias, requestMessages, controller.signal, (update) => {
          assistantHasEvidence ||= Boolean(update.content || update.toolCalls.length);
          setMessages((current) => current.map((message) => message.id === assistantId
            ? { ...message, content: update.content, tool_calls: update.toolCalls }
            : message));
        }, { temperature, maxTokens, systemPrompt });
      }
    } catch (reason) {
      if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "The model request failed.");
        if (!assistantHasEvidence) {
          setMessages((current) => current.filter((message) =>
            message.id !== userMessage.id && message.id !== assistantId
          ));
          setDraft((current) => current.trim() ? current : submittedDraft);
          setAttachments((current) => current.length ? current : submittedAttachments);
        } else {
          setMessages((current) => current.filter((message) =>
            message.id !== assistantId || message.content || message.tool_executions?.length || message.tool_calls?.length
          ));
        }
      }
    } finally {
      abortRef.current = null;
      setStreaming(false);
    }
  }

  const canSend = Boolean(runtime.data?.ready && activeAlias && (draft.trim() || attachments.length) && !streaming);

  return (
    <section className="workspace chat-workspace">
      <div className="page-head conversation-head">
        <div>
          <span className="eyebrow">SESSION · NOT PERSISTED</span>
          <h1>{activeAlias ? `Chat with ${activeAlias}` : "Chat"}</h1>
        </div>
        <div className="head-actions">
          {messages.length > 0 && <button className="ghost-button" onClick={() => setMessages([])}><Trash2 size={15} /> Clear</button>}
          <button className="ghost-button" onClick={() => setSettingsOpen(true)}><Settings2 size={15} /> Generation settings</button>
        </div>
      </div>

      <div className="chat-stage" aria-live="polite">
        {!runtime.isPending && !runtime.data?.ready ? (
          <EmptyState icon={<Bot size={30} />} title="No model is ready" detail="Open Models to activate an installed deployment before starting a conversation." />
        ) : messages.length === 0 ? (
          <div className="welcome-card">
            <span className="orb"><BrainCircuit size={30} /></span>
            <h2>What are we working on?</h2>
            <p>Messages remain in this browser session. The active deployment can stream text{supportsImages ? " and inspect images" : ""}. {supportsTools ? "Research tools are off by default; enabling them sends queries and requested public pages to the internet · no writes." : ""}</p>
            <div className="prompt-grid">
              {prompts.map(([label, prompt], index) => (
                <button key={label} onClick={() => setDraft(prompt)}>
                  {index === 0 ? <Code2 size={17} /> : index === 1 ? <Paperclip size={17} /> : <BrainCircuit size={17} />}
                  {label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="message-avatar">{message.role === "user" ? <User size={16} /> : <Bot size={16} />}</div>
                <div className="message-body">
                  <header>
                    <strong>{message.role === "user" ? "You" : activeAlias}</strong>
                    <span>{message.deployment_id}</span>
                    {message.content && <button className="message-action" aria-label="Copy message" onClick={() => void navigator.clipboard?.writeText(message.content)}><Copy size={14} /></button>}
                  </header>
                  {message.attachments?.length ? <div className="message-images">{message.attachments.map((item) => <img src={item.data_url} alt={item.name} key={item.id} />)}</div> : null}
                  <div className="message-content">{message.content || (streaming ? <span className="typing">Thinking</span> : "")}</div>
                  {message.tool_calls?.map((tool) => <ToolCallCard key={tool.id} tool={tool} />)}
                  {message.tool_executions?.map((tool) => <ToolExecutionCard key={tool.id} tool={tool} />)}
                  <SourceList sources={message.sources ?? []} />
                </div>
              </article>
            ))}
          </div>
        )}
      </div>

      {error && <div className="inline-error" role="alert">{error}<button onClick={() => setError(null)} aria-label="Dismiss error"><X size={15} /></button></div>}
      {attachments.length > 0 && <div className="attachment-tray">{attachments.map((item) => (
        <span key={item.id}><ImagePlus size={14} />{item.name}<button aria-label={`Remove ${item.name}`} onClick={() => setAttachments((all) => all.filter((entry) => entry.id !== item.id))}><X size={13} /></button></span>
      ))}</div>}
      <form className="composer" onSubmit={submit}>
        <label htmlFor="message" className="sr-only">Message the active model</label>
        <textarea
          id="message"
          rows={2}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          placeholder={runtime.data?.ready ? `Message ${activeAlias}…` : "Activate a model to begin…"}
          disabled={!runtime.data?.ready}
        />
        <div className="composer-bar">
          <div>
            <label className={supportsImages ? "attach-button" : "attach-button disabled"} title={supportsImages ? "Attach images" : "This deployment does not support images"}>
              <ImagePlus size={17} /><span>Image</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                multiple
                disabled={!supportsImages || remainingImageSlots === 0}
                onChange={(event) => void attachFiles(event.target.files)}
              />
            </label>
            <label
              className={"tool-toggle " + (toolsEnabled && supportsTools ? "enabled " : "") + (!supportsTools ? "disabled" : "")}
              title={supportsTools ? "Sends queries and requested public pages to the internet · no writes" : "This deployment does not support tools"}
            >
              <input
                type="checkbox"
                role="switch"
                aria-label="Research tools"
                checked={toolsEnabled && supportsTools}
                disabled={!supportsTools || streaming}
                onChange={(event) => setToolsEnabled(event.target.checked)}
              />
              <span className="toggle-track" aria-hidden="true"><span /></span>
              <ShieldCheck size={15} />
              <span className="tool-toggle-copy">
                <strong>Research tools</strong>
                <small>{supportsTools ? "Sends queries and requested public pages to the internet · no writes" : "Unavailable for this deployment"}</small>
              </span>
            </label>
            <span className="composer-meta">Temperature {temperature} · Max {maxTokens}</span>
          </div>
          {streaming
            ? <button type="button" className="stop-button" onClick={() => abortRef.current?.abort()}><Square size={14} /> Stop</button>
            : <button type="submit" disabled={!canSend}><Send size={15} /> Send</button>}
        </div>
      </form>

      <Modal open={settingsOpen} title="Generation settings" onClose={() => setSettingsOpen(false)}>
        <div className="form-stack">
          <label>Temperature <output>{temperature.toFixed(1)}</output><input type="range" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
          <label>Maximum output tokens<input type="number" min="1" max="8192" value={maxTokens} onChange={(event) => setMaxTokens(Math.max(1, Number(event.target.value)))} /></label>
          <label>System prompt<textarea rows={5} maxLength={maximumInstructionsLength} value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value.slice(0, maximumInstructionsLength))} placeholder="Optional instructions for this session" /></label>
          <button className="primary-button" onClick={() => setSettingsOpen(false)}>Apply settings</button>
        </div>
      </Modal>
    </section>
  );
}
