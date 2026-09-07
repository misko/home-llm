import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { consoleApi } from "../api/client";
import { subscribeToConsoleEvents } from "../api/events";

export const consoleKeys = {
  portfolio: ["console", "portfolio"] as const,
  runtime: ["console", "runtime"] as const,
  storage: ["console", "storage"] as const,
  runs: ["console", "runs"] as const,
  system: ["console", "system"] as const,
};

export function usePortfolio() {
  return useQuery({
    queryKey: consoleKeys.portfolio,
    queryFn: consoleApi.portfolio,
    retry: 3,
    refetchInterval: 15_000,
    refetchOnMount: "always",
  });
}

export function useRuntime() {
  return useQuery({ queryKey: consoleKeys.runtime, queryFn: consoleApi.runtime, refetchInterval: 5_000 });
}

export function useStorage() {
  return useQuery({ queryKey: consoleKeys.storage, queryFn: consoleApi.storage });
}

export function useRuns() {
  return useQuery({ queryKey: consoleKeys.runs, queryFn: consoleApi.runs });
}

export function useSystem() {
  return useQuery({ queryKey: consoleKeys.system, queryFn: consoleApi.system });
}

export function useConsoleEvents() {
  const queryClient = useQueryClient();
  useEffect(() => subscribeToConsoleEvents((event) => {
    if (event.runtime) queryClient.setQueryData(consoleKeys.runtime, event.runtime);
    if (event.type === "runtime.changed") {
      void queryClient.invalidateQueries({ queryKey: consoleKeys.portfolio });
    }
    if (event.type === "operation.changed" && event.operation?.state === "succeeded") {
      void queryClient.invalidateQueries({ queryKey: consoleKeys.runtime });
      void queryClient.invalidateQueries({ queryKey: consoleKeys.portfolio });
      void queryClient.invalidateQueries({ queryKey: consoleKeys.runs });
      void queryClient.invalidateQueries({ queryKey: consoleKeys.storage });
    }
  }), [queryClient]);
}
