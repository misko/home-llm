import type { ChatMessage } from "../api/types";

export interface ChatSettings {
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  toolsEnabled: boolean;
}

export interface SavedChat {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  settings: ChatSettings;
}

const databaseName = "llm-lab-chat-history";
const databaseVersion = 1;
const conversationStore = "conversations";
const fallbackKey = "llm-lab.chat-history.v1";
const activeChatKey = "llm-lab.active-chat.v1";

function openDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(databaseName, databaseVersion);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(conversationStore)) {
        request.result.createObjectStore(conversationStore, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Could not open chat history."));
  });
}

function requestResult<T>(request: IDBRequest<T>) {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Could not read chat history."));
  });
}

function transactionComplete(transaction: IDBTransaction) {
  return new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("Could not save chat history."));
    transaction.onabort = () => reject(transaction.error ?? new Error("Could not save chat history."));
  });
}

function fallbackChats() {
  try {
    const value = JSON.parse(localStorage.getItem(fallbackKey) ?? "[]");
    return Array.isArray(value) ? value as SavedChat[] : [];
  } catch {
    return [];
  }
}

function saveFallback(chats: SavedChat[]) {
  try {
    localStorage.setItem(fallbackKey, JSON.stringify(chats));
  } catch {
    // Storage may be unavailable or full. The current page remains usable.
  }
}

export async function loadChats(): Promise<SavedChat[]> {
  if (!("indexedDB" in globalThis)) return fallbackChats();
  try {
    const database = await openDatabase();
    const transaction = database.transaction(conversationStore, "readonly");
    const chats = await requestResult(transaction.objectStore(conversationStore).getAll()) as SavedChat[];
    database.close();
    return chats;
  } catch {
    return fallbackChats();
  }
}

export async function saveChats(chats: SavedChat[]) {
  if (!("indexedDB" in globalThis)) {
    const merged = new Map(fallbackChats().map((chat) => [chat.id, chat]));
    for (const chat of chats) merged.set(chat.id, chat);
    saveFallback([...merged.values()]);
    return;
  }
  try {
    const database = await openDatabase();
    const transaction = database.transaction(conversationStore, "readwrite");
    const store = transaction.objectStore(conversationStore);
    for (const chat of chats) store.put(chat);
    await transactionComplete(transaction);
    database.close();
  } catch {
    const merged = new Map(fallbackChats().map((chat) => [chat.id, chat]));
    for (const chat of chats) merged.set(chat.id, chat);
    saveFallback([...merged.values()]);
  }
}

export async function removeChat(chatId: string) {
  if (!("indexedDB" in globalThis)) {
    saveFallback(fallbackChats().filter((chat) => chat.id !== chatId));
    return;
  }
  try {
    const database = await openDatabase();
    const transaction = database.transaction(conversationStore, "readwrite");
    transaction.objectStore(conversationStore).delete(chatId);
    await transactionComplete(transaction);
    database.close();
  } catch {
    saveFallback(fallbackChats().filter((chat) => chat.id !== chatId));
  }
}

export function getActiveChatId() {
  try { return localStorage.getItem(activeChatKey); } catch { return null; }
}

export function setStoredActiveChatId(chatId: string) {
  try { localStorage.setItem(activeChatKey, chatId); } catch { /* optional convenience */ }
}
