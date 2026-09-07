import { afterEach, describe, expect, it, vi } from "vitest";
import { createClientId } from "./id";

describe("createClientId", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uses randomUUID when the browser exposes it", () => {
    vi.stubGlobal("crypto", {
      randomUUID: () => "12345678-1234-4234-8234-123456789abc",
      getRandomValues: vi.fn(),
    });
    expect(createClientId()).toBe("12345678-1234-4234-8234-123456789abc");
  });

  it("builds an RFC 4122-shaped ID on insecure HTTP without randomUUID", () => {
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => {
        bytes.set(Array.from({ length: 16 }, (_, index) => index));
        return bytes;
      },
    });
    expect(createClientId()).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
    );
  });

  it("still creates distinct printable IDs in an older WebView", () => {
    vi.stubGlobal("crypto", undefined);
    expect(createClientId()).toMatch(/^local-[a-z0-9-]+$/);
  });
});
