import type { ConsoleEvent } from "./types";

export function subscribeToConsoleEvents(
  onEvent: (event: ConsoleEvent) => void,
  onError?: () => void,
): () => void {
  const source = new EventSource("/api/v1/events");
  const receive = (event: MessageEvent<string>) => {
    try {
      onEvent(JSON.parse(event.data) as ConsoleEvent);
    } catch {
      onError?.();
    }
  };
  source.addEventListener("snapshot", receive as EventListener);
  source.addEventListener("runtime.changed", receive as EventListener);
  source.addEventListener("operation.changed", receive as EventListener);
  source.onerror = () => onError?.();
  return () => source.close();
}
