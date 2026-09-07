import { useMemo, useRef, useState, type FormEvent } from "react";
import {
  Bot,
  BrainCircuit,
  Code2,
  Copy,
  ImagePlus,
  Paperclip,
  Send,
  Settings2,
  Square,
  Trash2,
  User,
  Wrench,
  X,
} from "lucide-react";
import { streamChat } from "../api/chat";
import { createClientId } from "../api/id";
import type { ChatAttachment, ChatMessage, ToolCall } from "../api/types";
import { usePortfolio, useRuntime } from "../hooks/useConsoleData";
import { EmptyState } from "../components/EmptyState";
import { Modal } from "../components/Modal";

const prompts = [
  ["Review a piece of code", "Review this code for correctness, edge cases, and maintainability:\n\n"],
  ["Draft a structured plan", "Create a concrete implementation plan for "],
  ["Compare technical options", "Compare these technical options and recommend one: "],
] as const;

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
  const abortRef = useRef<AbortController | null>(null);

  const activeAlias = runtime.data?.public_alias;
  const activeDeployment = runtime.data?.deployment_id;
  const activeModel = useMemo(() => portfolio.data?.models.find((model) =>
    model.deployments.some((deployment) => deployment.id === activeDeployment)
  ), [portfolio.data, activeDeployment]);
  const supportsImages = activeModel?.modalities.includes("image") ?? false;

  async function attachFiles(files: FileList | null) {
    if (!files || !supportsImages) return;
    const selected = [...files].slice(0, 4 - attachments.length);
    const accepted: ChatAttachment[] = [];
    for (const file of selected) {
      if (!file.type.startsWith("image/") || file.size > 10 * 1024 * 1024) {
        setError("Images must be under 10 MiB and use a browser-supported image format.");
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
    setAttachments((current) => [...current, ...accepted]);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const content = draft.trim();
    if ((!content && !attachments.length) || !activeAlias || !runtime.data?.ready || streaming) return;
    const userMessage: ChatMessage = {
      id: createClientId(), role: "user", content, attachments, created_at: now(), deployment_id: activeDeployment ?? undefined,
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
    try {
      await streamChat(activeAlias, requestMessages, controller.signal, (update) => {
        setMessages((current) => current.map((message) => message.id === assistantId
          ? { ...message, content: update.content, tool_calls: update.toolCalls }
          : message));
      }, { temperature, maxTokens, systemPrompt });
    } catch (reason) {
      if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : "The model request failed.");
        setMessages((current) => current.filter((message) => message.id !== assistantId || message.content));
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
            <p>Messages remain in this browser session. The active deployment can stream text{supportsImages ? " and inspect images" : ""}; proposed tool calls are displayed but never executed automatically.</p>
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
              <input type="file" accept="image/*" multiple disabled={!supportsImages} onChange={(event) => void attachFiles(event.target.files)} />
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
          <label>System prompt<textarea rows={5} value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} placeholder="Optional instructions for this session" /></label>
          <button className="primary-button" onClick={() => setSettingsOpen(false)}>Apply settings</button>
        </div>
      </Modal>
    </section>
  );
}
