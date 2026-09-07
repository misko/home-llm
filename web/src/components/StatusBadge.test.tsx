import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusBadge } from "./StatusBadge";

describe("StatusBadge", () => {
  it("distinguishes ready, transitioning, failed, and inactive states", () => {
    const view = render(<StatusBadge ready />);
    expect(screen.getByText("Ready")).toBeInTheDocument();
    view.rerender(<StatusBadge active phase="starting" />);
    expect(screen.getByText("starting")).toBeInTheDocument();
    view.rerender(<StatusBadge active phase="failed" />);
    expect(screen.getByText("failed")).toBeInTheDocument();
    view.rerender(<StatusBadge />);
    expect(screen.getByText("Inactive")).toBeInTheDocument();
  });
});
