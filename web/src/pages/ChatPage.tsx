import { useEffect, useLayoutEffect, useMemo, useRef, useState, type FormEvent, type SetStateAction } from "react";
import {
  Bot,
  BrainCircuit,
  CheckCircle2,
  Code2,
  Copy,
  ExternalLink,
  ImagePlus,
  LoaderCircle,
  MessageSquarePlus,
  MessagesSquare,
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
import { approveWorkspaceWrite, ASSISTANT_TOOLSET, streamAgentTurn } from "../api/agent";
import { streamChat } from "../api/chat";
import { createClientId } from "../api/id";
import type { AgentSource, AgentToolExecution, ChatAttachment, ChatMessage, ToolCall } from "../api/types";
import { usePortfolio, useRuntime } from "../hooks/useConsoleData";
import { EmptyState } from "../components/EmptyState";
import { Modal } from "../components/Modal";
import {
  getActiveChatId,
  loadChats,
  loadMessagePage,
  removeChat,
  saveChats,
  setStoredActiveChatId,
  type SavedChat,
} from "../chat/history";

const prompts = [
  ["Review a piece of code", "Review this code for correctness, edge cases, and maintainability:\n\n"],
  ["Draft a structured plan", "Create a concrete implementation plan for "],
  ["Compare technical options", "Compare these technical options and recommend one: "],
] as const;
const supportedImageTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
const maximumConversationImages = 4;
const maximumConversationMessages = 64;
const maximumInstructionsLength = 16_384;
const defaultMaximumOutputTokens = 32_000;
const maximumOutputTokens = 32_768;

function now() {
  return new Date().toISOString();
}

function newChat(): SavedChat {
  const timestamp = now();
  return {
    id: createClientId(),
    title: "New chat",
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: [],
    settings: {
      temperature: 0,
      maxTokens: defaultMaximumOutputTokens,
      maxToolRounds: 128,
      systemPrompt: "",
      toolsEnabled: true, workspaceEnabled: true, pythonEnabled: true, openRouterEnabled: true,
    },
  };
}

function chatTitle(messages: ChatMessage[]) {
  const firstUserMessage = messages.find((message) => message.role === "user");
  const normalized = firstUserMessage?.content.replace(/\s+/g, " ").trim();
  if (normalized) return normalized.length > 52 ? `${normalized.slice(0, 49)}…` : normalized;
  if (firstUserMessage?.attachments?.length) return firstUserMessage.attachments[0].name || "Image conversation";
  return "New chat";
}

function chatPreview(chat: SavedChat) {
  if (chat.latestPreview) return chat.latestPreview;
  const last = [...chat.messages].reverse().find((message) => message.content.trim());
  return last?.content.replace(/\s+/g, " ").trim() || (chat.messages.length ? "Image conversation" : "No messages yet");
}

function chatTime(timestamp: string) {
  const date = new Date(timestamp);
  const today = new Date();
  if (date.toDateString() === today.toDateString()) {
    return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
  }
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
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

function boundedFetchSummary(tool: AgentToolExecution) {
  if (tool.name !== "web_fetch" || tool.status !== "completed" || !tool.result || typeof tool.result !== "object") return null;
  const result = tool.result as Record<string, unknown>;
  if (result.truncated !== true) return null;
  const characters = typeof result.extracted_characters === "number"
    ? `${result.extracted_characters.toLocaleString()} readable characters`
    : "partial page text";
  return `Bounded public extract · ${characters}`;
}

function workspaceProposalId(tool: AgentToolExecution) {
  if (tool.name !== "workspace_write_proposal" || tool.status !== "completed" || !tool.result || typeof tool.result !== "object") return null;
  const proposalId = (tool.result as Record<string, unknown>).proposal_id;
  return typeof proposalId === "string" ? proposalId : null;
}

function ToolExecutionCard({ tool, onApproveWorkspaceWrite }: {
  tool: AgentToolExecution;
  onApproveWorkspaceWrite: (tool: AgentToolExecution, proposalId: string) => void;
}) {
  const boundedFetch = boundedFetchSummary(tool);
  const statusLabel = tool.status === "running" ? "Running" : boundedFetch ? "Bounded extract" : tool.status === "completed" ? "Complete" : tool.status === "blocked" ? "Blocked" : "Failed";
  const disclosure = boundedFetch ?? (tool.name === "calculator" || tool.name === "current_time"
    ? "Runs locally · no network or writes"
    : tool.name === "python_sandbox"
      ? "Disposable Python · no network or host writes"
    : tool.name === "web_search" || tool.name === "web_fetch"
      ? "Sends queries and requested public pages to the internet · no writes"
      : "Read-only tool · no writes");
  const statusClass = boundedFetch ? "bounded" : tool.status;
  const proposalId = workspaceProposalId(tool);
  return (
    <details className={"tool-execution-card " + tool.status} open={tool.status === "running"} data-testid={"tool-" + tool.id}>
      <summary>
        <span className="tool-icon"><Wrench size={14} /></span>
        <span className="tool-identity">
          <strong>{toolLabel(tool.name)}</strong>
          <small>{disclosure}</small>
        </span>
        <span className={"tool-status " + statusClass}>
          {tool.status === "running" ? <LoaderCircle className="spin" size={13} /> : tool.status === "completed" ? <CheckCircle2 size={13} /> : tool.status === "blocked" ? <ShieldCheck size={13} /> : <X size={13} />}
          {statusLabel}
        </span>
      </summary>
      <div className="tool-execution-detail">
        {tool.arguments !== undefined && <div><span>Arguments</span><pre>{formatValue(tool.arguments)}</pre></div>}
        {tool.status === "completed" && tool.result !== undefined && <div><span>Result</span><pre>{formatValue(tool.result)}</pre></div>}
        {proposalId && <button className="primary-button compact" onClick={() => onApproveWorkspaceWrite(tool, proposalId)}>Approve write</button>}
        {tool.error && <div className="tool-error"><span>{tool.error.code ?? "Tool error"}</span><p>{tool.error.message}</p></div>}
      </div>
    </details>
  );
}

function ReasoningTrace({ reasoning, active }: { reasoning?: string; active: boolean }) {
  if (!reasoning && !active) return null;
  return (
    <details className="reasoning-trace">
      <summary>
        <BrainCircuit size={15} />
        <span>{reasoning ? "Model reasoning" : "Thinking"}</span>
        {active && <LoaderCircle className="spin" size={13} />}
        {reasoning && <small>{reasoning.length.toLocaleString()} characters</small>}
      </summary>
      {reasoning ? <pre>{reasoning}</pre> : <p>Waiting for the model’s reasoning stream…</p>}
    </details>
  );
}

function ToolActivity({ calls, executions, onApproveWorkspaceWrite }: {
  calls: ToolCall[];
  executions: AgentToolExecution[];
  onApproveWorkspaceWrite: (tool: AgentToolExecution, proposalId: string) => void;
}) {
  const count = calls.length + executions.length;
  if (!count) return null;
  return (
    <details className="tool-activity">
      <summary><Wrench size={15} /> Tool activity <small>{count}</small></summary>
      <div className="tool-activity-list">
        {calls.map((tool) => <ToolCallCard key={tool.id} tool={tool} />)}
        {executions.map((tool) => <ToolExecutionCard key={tool.id} tool={tool} onApproveWorkspaceWrite={onApproveWorkspaceWrite} />)}
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
  const initialChatRef = useRef<SavedChat | null>(null);
  if (!initialChatRef.current) initialChatRef.current = newChat();
  const [chats, setChats] = useState<SavedChat[]>([initialChatRef.current]);
  const [activeChatId, setActiveChatId] = useState(initialChatRef.current.id);
  const [historyReady, setHistoryReady] = useState(false);
  const [draft, setDraft] = useState("");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(() => new URLSearchParams(window.location.search).get("settings") === "conversation");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [reasoningByMessage, setReasoningByMessage] = useState<Record<string, string>>({});
  const [hasEarlierMessages, setHasEarlierMessages] = useState(false);
  const [loadingEarlierMessages, setLoadingEarlierMessages] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const scrollToBottomAfterLoadRef = useRef(true);
  const persistedChatsRef = useRef(new Map<string, SavedChat>());

  const activeChat = chats.find((chat) => chat.id === activeChatId) ?? chats[0];
  const messages = activeChat?.messages ?? [];
  const temperature = activeChat?.settings.temperature ?? 0;
  const maxTokens = activeChat?.settings.maxTokens ?? defaultMaximumOutputTokens;
  const maxToolRounds = activeChat?.settings.maxToolRounds ?? 128;
  const systemPrompt = activeChat?.settings.systemPrompt ?? "";
  const toolsEnabled = activeChat?.settings.toolsEnabled ?? false;
  const workspaceEnabled = activeChat?.settings.workspaceEnabled ?? true;
  const pythonEnabled = activeChat?.settings.pythonEnabled ?? true;
  const openRouterEnabled = activeChat?.settings.openRouterEnabled ?? true;

  useEffect(() => {
    let cancelled = false;
    void loadChats().then((savedChats) => {
      if (cancelled) return;
      if (savedChats.length) {
        const sorted = [...savedChats].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
        const storedActive = getActiveChatId();
        setChats(sorted);
        setActiveChatId(sorted.some((chat) => chat.id === storedActive) ? storedActive! : sorted[0].id);
      }
      setHistoryReady(true);
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!historyReady) return;
    let cancelled = false;
    scrollToBottomAfterLoadRef.current = true;
    void loadMessagePage(activeChatId).then((page) => {
      if (cancelled) return;
      setHasEarlierMessages(page.hasMore);
      setChats((current) => current.map((chat) => {
        if (chat.id !== activeChatId || chat.messages.length) return chat;
        return { ...chat, messages: page.messages };
      }));
    });
    return () => { cancelled = true; };
  }, [activeChatId, historyReady]);

  useLayoutEffect(() => {
    if (!historyReady || streaming || !scrollToBottomAfterLoadRef.current) return;
    chatEndRef.current?.scrollIntoView?.({ block: "end" });
    scrollToBottomAfterLoadRef.current = false;
  }, [activeChatId, historyReady, messages.length]);

  useEffect(() => {
    if (!historyReady) return;
    const changedChats = chats.filter((chat) => persistedChatsRef.current.get(chat.id) !== chat);
    if (!changedChats.length) return;
    const timer = window.setTimeout(() => {
      void saveChats(changedChats).then(() => {
        for (const chat of changedChats) persistedChatsRef.current.set(chat.id, chat);
      });
    }, 300);
    return () => window.clearTimeout(timer);
  }, [chats, historyReady]);

  useEffect(() => {
    if (historyReady) setStoredActiveChatId(activeChatId);
  }, [activeChatId, historyReady]);

  function updateActiveChat(transform: (chat: SavedChat) => SavedChat) {
    setChats((current) => current.map((chat) => chat.id === activeChatId ? transform(chat) : chat));
  }

  function setMessages(action: SetStateAction<ChatMessage[]>) {
    updateActiveChat((chat) => {
      const nextMessages = typeof action === "function" ? action(chat.messages) : action;
      const latest = [...nextMessages].reverse().find((message) => message.content.trim());
      return {
        ...chat, messages: nextMessages, title: chatTitle(nextMessages), updatedAt: now(),
        messageCount: Math.max(chat.messageCount ?? 0, nextMessages.length),
        latestPreview: latest?.content.replace(/\s+/g, " ").trim() || chat.latestPreview,
      };
    });
  }

  async function loadEarlierMessages() {
    if (loadingEarlierMessages || !hasEarlierMessages || !messages.length) return;
    setLoadingEarlierMessages(true);
    const priorHeight = document.documentElement.scrollHeight;
    const priorTop = window.scrollY;
    try {
      const page = await loadMessagePage(activeChatId, messages[0]);
      setChats((current) => current.map((chat) => chat.id === activeChatId
        ? { ...chat, messages: [...page.messages, ...chat.messages] }
        : chat));
      setHasEarlierMessages(page.hasMore);
      requestAnimationFrame(() => window.scrollTo({ top: priorTop + document.documentElement.scrollHeight - priorHeight }));
    } finally { setLoadingEarlierMessages(false); }
  }

  useEffect(() => {
    const onScroll = () => { if (window.scrollY < 160) void loadEarlierMessages(); };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  });

  function updateSettings(settings: Partial<SavedChat["settings"]>) {
    updateActiveChat((chat) => ({ ...chat, settings: { ...chat.settings, ...settings }, updatedAt: now() }));
  }

  function selectChat(chatId: string) {
    if (streaming) return;
    if (chatId === activeChatId) {
      setHistoryOpen(false);
      return;
    }
    setActiveChatId(chatId);
    setHistoryOpen(false);
    setDraft("");
    setAttachments([]);
    setError(null);
  }

  function createNewChat() {
    if (streaming) return;
    if (!messages.length) return;
    const chat = newChat();
    setChats((current) => [chat, ...current]);
    setActiveChatId(chat.id);
    setDraft("");
    setAttachments([]);
    setError(null);
    setHistoryOpen(false);
  }

  function deleteChat(chatId: string) {
    if (streaming || !window.confirm("Delete this saved chat? This cannot be undone.")) return;
    void removeChat(chatId);
    setChats((current) => {
      const remaining = current.filter((chat) => chat.id !== chatId);
      if (remaining.length) {
        if (chatId === activeChatId) setActiveChatId(remaining[0].id);
        return remaining;
      }
      const replacement = newChat();
      setActiveChatId(replacement.id);
      return [replacement];
    });
    setDraft("");
    setAttachments([]);
    setError(null);
  }

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
      setError("This conversation reached the 64-message limit. Start a new chat to continue.");
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
    setReasoningByMessage((current) => {
      const next = { ...current };
      delete next[assistantId];
      return next;
    });
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
          assistantHasEvidence ||= Boolean(update.content || update.reasoning || update.tools.length || update.sources.length);
          if (update.reasoning) setReasoningByMessage((current) => ({ ...current, [assistantId]: update.reasoning }));
          setMessages((current) => current.map((message) => message.id === assistantId
            ? { ...message, content: update.content, tool_executions: update.tools, sources: update.sources }
            : message));
        }, { temperature, maxTokens, maxToolRounds, systemPrompt, toolset: ASSISTANT_TOOLSET, enabledTools: ["web_search", "web_fetch", "calculator", "current_time", ...(workspaceEnabled ? ["workspace_list", "workspace_read", "workspace_write_proposal"] : []), ...(pythonEnabled ? ["python_sandbox"] : []), ...(openRouterEnabled ? ["openrouter_delegate"] : [])] });
      } else {
        await streamChat(activeAlias, requestMessages, controller.signal, (update) => {
          assistantHasEvidence ||= Boolean(update.content || update.reasoning || update.toolCalls.length);
          if (update.reasoning) setReasoningByMessage((current) => ({ ...current, [assistantId]: update.reasoning }));
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
          setReasoningByMessage((current) => {
            const next = { ...current };
            delete next[assistantId];
            return next;
          });
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

  async function approveWrite(tool: AgentToolExecution, proposalId: string) {
    try {
      const result = await approveWorkspaceWrite(proposalId);
      setMessages((current) => current.map((message) => ({
        ...message,
        tool_executions: message.tool_executions?.map((entry) => entry.id === tool.id
          ? {
              ...entry,
              result: {
                ...(entry.result && typeof entry.result === "object" ? entry.result as Record<string, unknown> : {}),
                approved: true,
                committed: result,
              },
            }
          : entry),
      })));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The workspace write could not be approved.");
    }
  }

  const canSend = Boolean(runtime.data?.ready && activeAlias && (draft.trim() || attachments.length) && !streaming);

  const sortedChats = [...chats].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

  return (
    <section className="workspace workspace-wide chat-workspace">
      {historyOpen && <button className="chat-history-backdrop" aria-label="Dismiss chat history" onClick={() => setHistoryOpen(false)} />}
      <aside className={historyOpen ? "chat-history open" : "chat-history"} aria-label="Chat history">
        <div className="chat-history-head">
          <div><MessagesSquare size={17} /><strong>Chats</strong></div>
          <div className="chat-history-actions">
            <button className="icon-button" aria-label="New chat" title="New chat" disabled={streaming} onClick={createNewChat}><MessageSquarePlus size={16} /></button>
            <button className="icon-button chat-history-close" aria-label="Close previous chats" onClick={() => setHistoryOpen(false)}><X size={16} /></button>
          </div>
        </div>
        <div className="chat-history-list">
          {sortedChats.map((chat) => (
            <div className={`chat-history-item ${chat.id === activeChatId ? "active" : ""}`} key={chat.id}>
              <button className="chat-history-select" disabled={streaming} onClick={() => selectChat(chat.id)} aria-current={chat.id === activeChatId ? "page" : undefined}>
                <span><strong>{chat.title}</strong><time dateTime={chat.updatedAt}>{chatTime(chat.updatedAt)}</time></span>
                <small>Latest: {chatPreview(chat)}</small>
              </button>
              <button className="chat-history-delete" aria-label={`Delete ${chat.title}`} title="Delete chat" disabled={streaming} onClick={() => deleteChat(chat.id)}><Trash2 size={13} /></button>
            </div>
          ))}
        </div>
        <p>Saved in this browser. Select a chat to continue it.</p>
      </aside>

      <div className="chat-conversation">
      <div className="page-head conversation-head">
        <div>
          <span className="eyebrow">SAVED CHAT · {activeChat?.messageCount ?? messages.length} MESSAGES</span>
          <h1>{activeAlias ? `Chat with ${activeAlias}` : "Chat"}</h1>
        </div>
        <div className="head-actions">
          {messages.length > 0 && <button className="ghost-button" disabled={streaming} onClick={() => setMessages([])}><Trash2 size={15} /> Clear</button>}
          <button className="ghost-button" disabled={streaming} onClick={createNewChat}><MessageSquarePlus size={15} /> New chat</button>
          <button className="ghost-button" onClick={() => setSettingsOpen(true)}><Settings2 size={15} /> Generation settings</button>
        </div>
        <div className="mobile-chat-actions" aria-label="Chat controls">
          <button
            className="mobile-chat-action"
            aria-label="Previous chats"
            aria-expanded={historyOpen}
            onClick={() => setHistoryOpen(true)}
          >
            <MessagesSquare size={18} />
            <span>Chats</span>
          </button>
          <button className="mobile-chat-action" aria-label="Open settings" onClick={() => setSettingsOpen(true)}>
            <Settings2 size={18} />
            <span>Settings</span>
          </button>
        </div>
      </div>

      <div className="chat-stage" aria-live="polite">
        {!runtime.isPending && !runtime.data?.ready ? (
          <EmptyState icon={<Bot size={30} />} title="No model is ready" detail="Open Models to activate an installed deployment before starting a conversation." />
        ) : messages.length === 0 ? (
          <div className="welcome-card">
            <span className="orb"><BrainCircuit size={30} /></span>
            <h2>What are we working on?</h2>
            <p>This chat is saved in this browser so you can return and continue later. The active deployment can stream text{supportsImages ? " and inspect images" : ""}. {supportsTools ? "Assistant tools are enabled by default; adjust local files, Python, and OpenRouter permissions in generation settings." : ""}</p>
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
            {hasEarlierMessages && <button className="load-earlier" disabled={loadingEarlierMessages} onClick={() => void loadEarlierMessages()}>{loadingEarlierMessages ? "Loading earlier messages…" : "Load earlier messages"}</button>}
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
                  {message.role === "assistant" && <ReasoningTrace reasoning={reasoningByMessage[message.id]} active={streaming && message.id === messages.at(-1)?.id} />}
                  <div className="message-content">{message.content || (streaming ? <span className="typing">Thinking</span> : "")}</div>
                  <ToolActivity calls={message.tool_calls ?? []} executions={message.tool_executions ?? []} onApproveWorkspaceWrite={approveWrite} />
                  <SourceList sources={message.sources ?? []} />
                </div>
              </article>
            ))}
            <div ref={chatEndRef} />
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
              title={supportsTools ? "Uses the tool permissions selected in generation settings" : "This deployment does not support tools"}
            >
              <input
                type="checkbox"
                role="switch"
                aria-label="Research tools"
                checked={toolsEnabled && supportsTools}
                disabled={!supportsTools || streaming}
                onChange={(event) => updateSettings({ toolsEnabled: event.target.checked })}
              />
              <span className="toggle-track" aria-hidden="true"><span /></span>
              <ShieldCheck size={15} />
              <span className="tool-toggle-copy">
                <strong>Assistant tools</strong>
                <small>{supportsTools ? "Web research plus the permissions selected in settings" : "Unavailable for this deployment"}</small>
              </span>
            </label>
            <span className="composer-meta">Temperature {temperature} · Max {maxTokens}</span>
            <button type="button" className="composer-settings" aria-label="Open generation settings" title="Generation settings" onClick={() => setSettingsOpen(true)}><Settings2 size={16} /></button>
          </div>
          {streaming
            ? <button type="button" className="stop-button" onClick={() => abortRef.current?.abort()}><Square size={14} /> Stop</button>
            : <button type="submit" disabled={!canSend}><Send size={15} /> Send</button>}
        </div>
      </form>

      <Modal open={settingsOpen} title="Generation settings" onClose={() => setSettingsOpen(false)}>
        <div className="form-stack">
          <label>Temperature <output>{temperature.toFixed(1)}</output><input type="range" min="0" max="2" step="0.1" value={temperature} onChange={(event) => updateSettings({ temperature: Number(event.target.value) })} /></label>
          <label>Maximum output tokens<input type="number" min="1" max={maximumOutputTokens} value={maxTokens} onChange={(event) => updateSettings({ maxTokens: Math.min(maximumOutputTokens, Math.max(1, Math.trunc(Number(event.target.value) || 1))) })} /></label>
          <label>Maximum tool rounds <output>{maxToolRounds === 128 ? "Unlimited (128)" : maxToolRounds}</output><input type="range" min="1" max="128" step="1" value={maxToolRounds} onChange={(event) => updateSettings({ maxToolRounds: Number(event.target.value) })} /></label>
          <label><input type="checkbox" checked={workspaceEnabled} onChange={(event) => updateSettings({ workspaceEnabled: event.target.checked })} /> Local file access · write proposals need approval</label>
          <label><input type="checkbox" checked={pythonEnabled} onChange={(event) => updateSettings({ pythonEnabled: event.target.checked })} /> Python sandbox · no network or host writes</label>
          <label><input type="checkbox" checked={openRouterEnabled} onChange={(event) => updateSettings({ openRouterEnabled: event.target.checked })} /> OpenRouter delegation · prompt leaves this machine</label>
          <p className="modal-note">
            {defaultMaximumOutputTokens.toLocaleString()} is a ceiling, not a target. Prompt, history, reasoning, and reply share the active {runtime.data?.context_size?.toLocaleString() ?? "model"}-token context; tool-enabled turns also stop at the configured round cap or total deadline.
          </p>
          <label>System prompt<textarea rows={5} maxLength={maximumInstructionsLength} value={systemPrompt} onChange={(event) => updateSettings({ systemPrompt: event.target.value.slice(0, maximumInstructionsLength) })} placeholder="Optional instructions for this chat" /></label>
          <button className="primary-button" onClick={() => setSettingsOpen(false)}>Apply settings</button>
        </div>
      </Modal>
      </div>
    </section>
  );
}
