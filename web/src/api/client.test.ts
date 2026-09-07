import { afterEach, describe, expect, it, vi } from "vitest";
import { consoleApi } from "./client";

describe("consoleApi activation", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends concurrency and idempotency headers without randomUUID", async () => {
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => {
        bytes.fill(7);
        return bytes;
      },
    });
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      id: "op_1234567890abcdef1234567890abcdef",
      kind: "activate",
      state: "queued",
      created_at: "2026-09-06T20:00:00Z",
      updated_at: "2026-09-06T20:00:00Z",
    }), { status: 202, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    await consoleApi.activate(
      "muse-glimmer-30b-4090-8k",
      "cat1-current",
      "rt1-current",
    );

    const [path, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(path).toBe("/api/v1/runtime/activations");
    expect(init.method).toBe("POST");
    expect(new Headers(init.headers).get("If-Match")).toBe("rt1-current");
    expect(new Headers(init.headers).get("Idempotency-Key")).toMatch(/^[0-9a-f-]{36}$/);
    expect(JSON.parse(String(init.body))).toEqual({
      deployment_id: "muse-glimmer-30b-4090-8k",
      catalog_revision: "cat1-current",
    });
  });
});
