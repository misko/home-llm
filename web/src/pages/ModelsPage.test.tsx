import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { portfolioFixture, runtimeFixture } from "../test/fixtures";
import { ModelsPage } from "./ModelsPage";

describe("ModelsPage activation", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("activates Muse through the reviewed async operation flow on HTTP", async () => {
    let museActive = false;
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => {
        bytes.fill(9);
        return bytes;
      },
    });
    const requests: Array<{ path: string; init?: RequestInit }> = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      requests.push({ path, init });
      let payload: unknown;
      let status = 200;
      if (path === "/api/v1/models") payload = portfolioFixture(museActive);
      else if (path === "/api/v1/runtime") payload = runtimeFixture(museActive ? {
        revision: "rt1-local-agent",
        deployment_id: "muse-glimmer-30b-4090-8k",
        public_alias: "local-agent",
        model_name: "Muse Glimmer 30B",
      } : {});
      else if (path === "/api/v1/runtime/activations") {
        status = 202;
        payload = { id: "op_1234567890abcdef1234567890abcdef", kind: "activate", state: "queued", created_at: "2026-09-06T20:00:00Z", updated_at: "2026-09-06T20:00:00Z" };
      } else if (path.includes("/api/v1/operations/")) {
        museActive = true;
        payload = { id: "op_1234567890abcdef1234567890abcdef", kind: "activate", state: "succeeded", created_at: "2026-09-06T20:00:00Z", updated_at: "2026-09-06T20:00:01Z", result: { deployment_id: "muse-glimmer-30b-4090-8k" } };
      } else throw new Error(`unexpected request ${path}`);
      return new Response(JSON.stringify(payload), { status, headers: { "Content-Type": "application/json" } });
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(<QueryClientProvider client={client}><ModelsPage /></QueryClientProvider>);
    const card = await screen.findByTestId("model-muse-glimmer-30b");
    await userEvent.click(within(card).getByRole("button", { name: "Activate" }));
    expect(screen.getByRole("dialog", { name: "Activate local-agent" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Verify and activate" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await waitFor(() => expect(within(card).getByText("Ready")).toBeInTheDocument());
    const activation = requests.find((request) => request.path === "/api/v1/runtime/activations");
    expect(new Headers(activation?.init?.headers).get("Idempotency-Key")).toBeTruthy();
    expect(JSON.parse(String(activation?.init?.body))).toEqual({
      deployment_id: "muse-glimmer-30b-4090-8k",
      catalog_revision: "cat1-console-e2e",
    });
    client.clear();
  });

  it("keeps submission errors inside the review and reuses the request key", async () => {
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => {
        bytes.fill(7);
        return bytes;
      },
    });
    const activationKeys: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path === "/api/v1/models") {
        return new Response(JSON.stringify(portfolioFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/runtime") {
        return new Response(JSON.stringify(runtimeFixture()), { headers: { "Content-Type": "application/json" } });
      }
      if (path === "/api/v1/runtime/activations") {
        activationKeys.push(new Headers(init?.headers).get("Idempotency-Key") ?? "");
        return new Response(JSON.stringify({
          error: { code: "gateway_unavailable", message: "The activation request was not acknowledged.", retryable: true },
        }), { status: 503, headers: { "Content-Type": "application/json" } });
      }
      throw new Error(`unexpected request ${path}`);
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
    render(<QueryClientProvider client={client}><ModelsPage /></QueryClientProvider>);
    const card = await screen.findByTestId("model-muse-glimmer-30b");
    await userEvent.click(within(card).getByRole("button", { name: "Activate" }));
    const dialog = screen.getByRole("dialog", { name: "Activate local-agent" });

    await userEvent.click(within(dialog).getByRole("button", { name: "Verify and activate" }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("not acknowledged");
    await userEvent.click(within(dialog).getByRole("button", { name: "Verify and activate" }));
    await waitFor(() => expect(activationKeys).toHaveLength(2));
    expect(activationKeys[0]).toBeTruthy();
    expect(activationKeys[1]).toBe(activationKeys[0]);
    client.clear();
  });
});
