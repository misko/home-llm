import type { ChatMessage } from "../api/types";

export interface ChatSettings { temperature: number; maxTokens: number; maxToolRounds?: number; systemPrompt: string; toolsEnabled: boolean; toolset?: "standard-readonly" | "workspace-files" | "python-sandbox" | "openrouter-delegation"; }
export interface SavedChat { id: string; title: string; createdAt: string; updatedAt: string; messages: ChatMessage[]; settings: ChatSettings; messageCount?: number; latestPreview?: string; }
export interface MessagePage { messages: ChatMessage[]; hasMore: boolean; }
type ConversationRecord = Omit<SavedChat, "messages">;
type MessageRecord = { chatId: string; createdAt: string; id: string; message: ChatMessage };

const databaseName = "llm-lab-chat-history", databaseVersion = 2;
const conversationStore = "conversations", messageStore = "messages", messageIndex = "by-chat-created";
const fallbackKey = "llm-lab.chat-history.v1", activeChatKey = "llm-lab.active-chat.v1";

function openDatabase() { return new Promise<IDBDatabase>((resolve, reject) => {
  const request = indexedDB.open(databaseName, databaseVersion);
  request.onupgradeneeded = (event) => {
    const database = request.result, transaction = request.transaction!;
    if (!database.objectStoreNames.contains(conversationStore)) database.createObjectStore(conversationStore, { keyPath: "id" });
    const messages = database.objectStoreNames.contains(messageStore) ? transaction.objectStore(messageStore) : database.createObjectStore(messageStore, { keyPath: ["chatId", "id"] });
    if (!messages.indexNames.contains(messageIndex)) messages.createIndex(messageIndex, ["chatId", "createdAt", "id"]);
    if (event.oldVersion < 2) transaction.objectStore(conversationStore).openCursor().onsuccess = (cursorEvent) => {
      const cursor = (cursorEvent.target as IDBRequest<IDBCursorWithValue | null>).result; if (!cursor) return;
      const legacy = cursor.value as SavedChat, legacyMessages = Array.isArray(legacy.messages) ? legacy.messages : [];
      const { messages: _old, ...metadata } = legacy;
      transaction.objectStore(conversationStore).put({ ...metadata, messageCount: legacyMessages.length, latestPreview: legacyMessages.at(-1)?.content ?? "" });
      for (const message of legacyMessages) messages.put({ chatId: legacy.id, createdAt: message.created_at, id: message.id, message });
      cursor.continue();
    };
  };
  request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error ?? new Error("Could not open chat history."));
}); }
function result<T>(request: IDBRequest<T>) { return new Promise<T>((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error ?? new Error("Could not read chat history.")); }); }
function complete(transaction: IDBTransaction) { return new Promise<void>((resolve, reject) => { transaction.oncomplete = () => resolve(); transaction.onerror = transaction.onabort = () => reject(transaction.error ?? new Error("Could not save chat history.")); }); }
function fallbackChats() { try { const data = JSON.parse(localStorage.getItem(fallbackKey) ?? "[]"); return Array.isArray(data) ? data as SavedChat[] : []; } catch { return []; } }
function saveFallback(chats: SavedChat[]) { try { localStorage.setItem(fallbackKey, JSON.stringify(chats)); } catch { /* optional */ } }

export async function loadChats(): Promise<SavedChat[]> {
  if (!("indexedDB" in globalThis)) return fallbackChats();
  try { const database = await openDatabase(), transaction = database.transaction(conversationStore, "readonly"); const chats = await result(transaction.objectStore(conversationStore).getAll()) as ConversationRecord[]; database.close(); return chats.map((chat) => ({ ...chat, messages: [] })); } catch { return fallbackChats(); }
}
export async function loadMessagePage(chatId: string, before?: ChatMessage, limit = 40): Promise<MessagePage> {
  if (!("indexedDB" in globalThis)) { const messages = fallbackChats().find((chat) => chat.id === chatId)?.messages ?? []; const end = before ? messages.findIndex((message) => message.id === before.id) : messages.length; const start = Math.max(0, end - limit); return { messages: messages.slice(start, end), hasMore: start > 0 }; }
  try {
    const database = await openDatabase(), transaction = database.transaction(messageStore, "readonly"), index = transaction.objectStore(messageStore).index(messageIndex);
    const upper = before ? [chatId, before.created_at, before.id] : [chatId, "\uffff", "\uffff"];
    const range = IDBKeyRange.bound([chatId, "", ""], upper, false, Boolean(before)); const records: MessageRecord[] = [];
    await new Promise<void>((resolve, reject) => { const request = index.openCursor(range, "prev"); request.onerror = () => reject(request.error ?? new Error("Could not load messages.")); request.onsuccess = () => { const cursor = request.result; if (!cursor || records.length > limit) return resolve(); records.push(cursor.value as MessageRecord); cursor.continue(); }; });
    database.close(); return { messages: records.slice(0, limit).reverse().map((record) => record.message), hasMore: records.length > limit };
  } catch { return { messages: [], hasMore: false }; }
}
export async function saveChats(chats: SavedChat[]) {
  if (!("indexedDB" in globalThis)) { const all = new Map(fallbackChats().map((chat) => [chat.id, chat])); for (const chat of chats) all.set(chat.id, chat); saveFallback([...all.values()]); return; }
  try { const database = await openDatabase(), transaction = database.transaction([conversationStore, messageStore], "readwrite"), conversations = transaction.objectStore(conversationStore), messages = transaction.objectStore(messageStore); for (const chat of chats) { const { messages: visible, ...metadata } = chat; conversations.put({ ...metadata, messageCount: chat.messageCount ?? visible.length, latestPreview: chat.latestPreview ?? visible.at(-1)?.content ?? "" }); for (const message of visible) messages.put({ chatId: chat.id, createdAt: message.created_at, id: message.id, message }); } await complete(transaction); database.close(); } catch { const all = new Map(fallbackChats().map((chat) => [chat.id, chat])); for (const chat of chats) all.set(chat.id, chat); saveFallback([...all.values()]); }
}
export async function removeChat(chatId: string) {
  if (!("indexedDB" in globalThis)) { saveFallback(fallbackChats().filter((chat) => chat.id !== chatId)); return; }
  try { const database = await openDatabase(), transaction = database.transaction([conversationStore, messageStore], "readwrite"); transaction.objectStore(conversationStore).delete(chatId); const index = transaction.objectStore(messageStore).index(messageIndex), range = IDBKeyRange.bound([chatId, "", ""], [chatId, "\uffff", "\uffff"]); index.openCursor(range).onsuccess = (event) => { const cursor = (event.target as IDBRequest<IDBCursorWithValue | null>).result; if (cursor) { cursor.delete(); cursor.continue(); } }; await complete(transaction); database.close(); } catch { saveFallback(fallbackChats().filter((chat) => chat.id !== chatId)); }
}
export function getActiveChatId() { try { return localStorage.getItem(activeChatKey); } catch { return null; } }
export function setStoredActiveChatId(chatId: string) { try { localStorage.setItem(activeChatKey, chatId); } catch { /* optional */ } }
