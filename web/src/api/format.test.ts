import { describe, expect, it } from "vitest";
import { formatBytes, formatDuration, formatTimestamp, shortHash } from "./format";

describe("console formatters", () => {
  it("formats storage and duration values without false precision", () => {
    expect(formatBytes(18_157_050_004)).toBe("16.9 GiB");
    expect(formatBytes(-1)).toBe("—");
    expect(formatDuration(833)).toBe("833 ms");
    expect(formatDuration(1250)).toBe("1.25 s");
  });

  it("redacts malformed or missing identity values", () => {
    expect(shortHash("abcdef1234567890", 8)).toBe("abcdef12…");
    expect(shortHash()).toBe("—");
    expect(formatTimestamp("not-a-date")).toBe("—");
  });
});
