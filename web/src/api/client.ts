import type {
  BenchmarkRunDetail,
  BenchmarkRuns,
  ModelPortfolio,
  OperationSummary,
  RuntimeStatus,
  StorageStatus,
  SystemStatus,
} from "./types";
import { createClientId } from "./id";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryable: boolean;

  constructor(status: number, code: string, message: string, retryable = false) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let code = "request_failed";
    let message = `${response.status} ${response.statusText}`;
    let retryable = response.status >= 500;
    try {
      const payload = await response.json() as {
        error?: { code?: string; message?: string; retryable?: boolean };
      };
      code = payload.error?.code ?? code;
      message = payload.error?.message ?? message;
      retryable = payload.error?.retryable ?? retryable;
    } catch {
      // The status remains useful when an intermediary returned non-JSON.
    }
    throw new ApiError(response.status, code, message, retryable);
  }
  return await response.json() as T;
}

export const consoleApi = {
  portfolio: () => request<ModelPortfolio>("/api/v1/models"),
  runtime: () => request<RuntimeStatus>("/api/v1/runtime"),
  storage: () => request<StorageStatus>("/api/v1/storage"),
  system: () => request<SystemStatus>("/api/v1/system"),
  runs: () => request<BenchmarkRuns>("/api/v1/runs"),
  run: (runId: string) => request<BenchmarkRunDetail>(`/api/v1/runs/${encodeURIComponent(runId)}`),
  operation: (operationId: string) => request<OperationSummary>(`/api/v1/operations/${encodeURIComponent(operationId)}`),
  activate: (deploymentId: string, catalogRevision: string, runtimeRevision: string, idempotencyKey = createClientId()) =>
    request<OperationSummary>("/api/v1/runtime/activations", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": idempotencyKey,
        "If-Match": runtimeRevision,
      },
      body: JSON.stringify({ deployment_id: deploymentId, catalog_revision: catalogRevision }),
    }),
  stop: (runtimeRevision: string, idempotencyKey = createClientId()) => request<OperationSummary>("/api/v1/runtime/active", {
    method: "DELETE",
    headers: {
      "Idempotency-Key": idempotencyKey,
      "If-Match": runtimeRevision,
    },
  }),
  runBenchmark: (deploymentId: string, suiteId: string, runtimeRevision: string) =>
    request<OperationSummary>("/api/v1/benchmarks", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": createClientId(),
        "If-Match": runtimeRevision,
      },
      body: JSON.stringify({ deployment_id: deploymentId, suite_id: suiteId, telemetry: true }),
    }),
};
