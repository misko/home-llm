import type { ModelPortfolio, OperationSummary, RuntimeStatus } from "./api/types";

interface ModelContextTool {
  name: string;
  title: string;
  description: string;
  inputSchema: object;
  annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
  execute(input: unknown): unknown | Promise<unknown>;
}

interface ModelContext {
  registerTool(tool: ModelContextTool, options?: { signal?: AbortSignal }): void | Promise<void>;
}

declare global {
  interface Document {
    readonly modelContext?: ModelContext;
  }
}

export interface ConsoleWebMcpApi {
  portfolio(): Promise<ModelPortfolio>;
  runtime(): Promise<RuntimeStatus>;
  operation(operationId: string): Promise<OperationSummary>;
  activate(deploymentId: string, catalogRevision: string, runtimeRevision: string): Promise<OperationSummary>;
  stop(runtimeRevision: string): Promise<OperationSummary>;
}

interface ConsoleWebMcpOptions {
  api: ConsoleWebMcpApi;
  navigate(path: string): void;
  refresh(): void | Promise<void>;
  pollIntervalMs?: number;
  timeoutMs?: number;
  reportError?: (error: unknown) => void;
}

const terminalStates = new Set(["succeeded", "failed", "cancelled", "interrupted"]);

function emptyInput(input: unknown): void {
  if (input == null) return;
  if (typeof input === "object" && !Array.isArray(input) && Object.keys(input).length === 0) return;
  throw new TypeError("Input must be an empty object.");
}

function deploymentInput(input: unknown): string {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new TypeError("deploymentId is required.");
  }
  const record = input as Record<string, unknown>;
  if (Object.keys(record).length !== 1 || typeof record.deploymentId !== "string" ||
      !/^[a-z0-9][a-z0-9._-]*$/.test(record.deploymentId)) {
    throw new TypeError("deploymentId must be one reviewed deployment identifier.");
  }
  return record.deploymentId;
}

async function waitForOperation(
  api: ConsoleWebMcpApi,
  operationId: string,
  pollIntervalMs: number,
  timeoutMs: number,
): Promise<OperationSummary> {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    const operation = await api.operation(operationId);
    if (terminalStates.has(operation.state)) {
      if (operation.state !== "succeeded") {
        throw new Error(operation.error?.message ?? `Operation ${operation.state}.`);
      }
      return operation;
    }
    if (Date.now() >= deadline) {
      throw new Error("The operation is still running; check the Models page for status.");
    }
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
  }
}

export function registerConsoleWebMcp(options: ConsoleWebMcpOptions): () => void {
  const context = typeof document === "undefined" ? undefined : document.modelContext;
  if (!context?.registerTool) return () => undefined;
  const lifecycle = new AbortController();
  const pollIntervalMs = options.pollIntervalMs ?? 750;
  const timeoutMs = options.timeoutMs ?? 420_000;
  const reportError = options.reportError ?? ((error) => console.warn("WebMCP registration failed", error));

  const tools: ModelContextTool[] = [
    {
      name: "get_model_runtime",
      title: "Get model runtime",
      description: "Read the active local model and the reviewed deployment choices.",
      inputSchema: { type: "object", properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: true, untrustedContentHint: false },
      async execute(input) {
        emptyInput(input);
        const [portfolio, runtime] = await Promise.all([options.api.portfolio(), options.api.runtime()]);
        return {
          active: runtime.active,
          ready: runtime.ready,
          deploymentId: runtime.deployment_id ?? null,
          publicAlias: runtime.public_alias ?? null,
          deployments: portfolio.models.flatMap((model) => model.deployments.map((deployment) => ({
            id: deployment.id,
            publicAlias: deployment.public_alias,
            model: model.display_name,
            installed: deployment.artifact.registered,
            active: deployment.active,
          }))),
        };
      },
    },
    {
      name: "activate_reviewed_model",
      title: "Activate reviewed model",
      description: "Verify and activate one installed catalog deployment for every connected client.",
      inputSchema: {
        type: "object",
        properties: { deploymentId: { type: "string", pattern: "^[a-z0-9][a-z0-9._-]*$" } },
        required: ["deploymentId"],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      async execute(input) {
        const deploymentId = deploymentInput(input);
        const [portfolio, runtime] = await Promise.all([options.api.portfolio(), options.api.runtime()]);
        const deployment = portfolio.models.flatMap((model) => model.deployments)
          .find((candidate) => candidate.id === deploymentId);
        if (!deployment) throw new Error("Deployment is not in the reviewed catalog.");
        if (!deployment.artifact.registered) throw new Error("Deployment artifact is not installed.");
        if (deployment.active && deployment.ready) {
          return { state: "succeeded", deploymentId, publicAlias: deployment.public_alias, changed: false };
        }
        options.navigate("/models");
        const admitted = await options.api.activate(deploymentId, portfolio.catalog_revision, runtime.revision);
        const completed = await waitForOperation(options.api, admitted.id, pollIntervalMs, timeoutMs);
        await options.refresh();
        return {
          state: completed.state,
          operationId: completed.id,
          deploymentId,
          publicAlias: deployment.public_alias,
          changed: true,
        };
      },
    },
    {
      name: "stop_active_model",
      title: "Stop active model",
      description: "Stop the active local model for every connected client.",
      inputSchema: { type: "object", properties: {}, additionalProperties: false },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      async execute(input) {
        emptyInput(input);
        const runtime = await options.api.runtime();
        if (!runtime.active) return { state: "succeeded", stopped: false };
        options.navigate("/models");
        const admitted = await options.api.stop(runtime.revision);
        const completed = await waitForOperation(options.api, admitted.id, pollIntervalMs, timeoutMs);
        await options.refresh();
        return { state: completed.state, operationId: completed.id, stopped: true };
      },
    },
  ];

  for (const tool of tools) {
    try {
      void Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch(reportError);
    } catch (error) {
      reportError(error);
    }
  }
  return () => lifecycle.abort();
}
