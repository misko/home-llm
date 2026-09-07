import { afterEach, describe, expect, it, vi } from "vitest";
import { portfolioFixture, runtimeFixture } from "./test/fixtures";
import { registerConsoleWebMcp, type ConsoleWebMcpApi } from "./webmcp";

describe("console WebMCP", () => {
  afterEach(() => {
    Reflect.deleteProperty(document, "modelContext");
  });

  it("registers read and lifecycle tools and completes activation", async () => {
    const tools = new Map<string, { execute(input: unknown): unknown | Promise<unknown>; annotations: { readOnlyHint: boolean } }>();
    let signal: AbortSignal | undefined;
    Object.defineProperty(document, "modelContext", {
      configurable: true,
      value: {
        registerTool: vi.fn((tool, options) => {
          tools.set(tool.name, tool);
          signal = options?.signal;
        }),
      },
    });
    const succeeded = {
      id: "op_1234567890abcdef1234567890abcdef",
      kind: "activate" as const,
      state: "succeeded" as const,
      created_at: "2026-09-06T20:00:00Z",
      updated_at: "2026-09-06T20:00:01Z",
    };
    const api: ConsoleWebMcpApi = {
      portfolio: vi.fn(async () => portfolioFixture()),
      runtime: vi.fn(async () => runtimeFixture()),
      operation: vi.fn(async () => succeeded),
      activate: vi.fn(async () => ({ ...succeeded, state: "queued" as const })),
      stop: vi.fn(async () => ({ ...succeeded, kind: "stop" as const, state: "queued" as const })),
    };
    const navigate = vi.fn();
    const refresh = vi.fn();

    const dispose = registerConsoleWebMcp({ api, navigate, refresh, pollIntervalMs: 0 });
    expect([...tools]).toHaveLength(3);
    expect(tools.get("get_model_runtime")?.annotations.readOnlyHint).toBe(true);
    expect(tools.get("activate_reviewed_model")?.annotations.readOnlyHint).toBe(false);

    const read = await tools.get("get_model_runtime")?.execute({});
    expect(read).toMatchObject({ active: true, deploymentId: "ling-3.0-tiny-4090-8k" });
    await expect(tools.get("activate_reviewed_model")?.execute({ deploymentId: "unknown" }))
      .rejects.toThrow("reviewed catalog");
    const activated = await tools.get("activate_reviewed_model")?.execute({
      deploymentId: "muse-glimmer-30b-4090-8k",
    });
    expect(api.activate).toHaveBeenCalledWith(
      "muse-glimmer-30b-4090-8k",
      "cat1-console-e2e",
      "rt1-local-fast",
    );
    expect(activated).toMatchObject({ state: "succeeded", publicAlias: "local-agent", changed: true });
    expect(navigate).toHaveBeenCalledWith("/models");
    expect(refresh).toHaveBeenCalledOnce();

    dispose();
    expect(signal?.aborted).toBe(true);
  });

  it("is a no-op where the proposed browser API is unavailable", () => {
    expect(() => registerConsoleWebMcp({
      api: {} as ConsoleWebMcpApi,
      navigate: () => undefined,
      refresh: () => undefined,
    })()).not.toThrow();
  });
});
