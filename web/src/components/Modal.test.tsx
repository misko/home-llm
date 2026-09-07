import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Modal } from "./Modal";

function ModalHarness({ onClose }: { onClose: () => void }) {
  const [open, setOpen] = useState(false);
  const close = () => { onClose(); setOpen(false); };
  return <>
    <button onClick={() => setOpen(true)}>Open review</button>
    <Modal open={open} title="Review activation" onClose={close}>
      <button onClick={close}>Cancel</button>
      <button>Activate</button>
    </Modal>
  </>;
}

describe("Modal", () => {
  it("moves focus into the dialog, traps Tab, and restores focus", async () => {
    const user = userEvent.setup();
    render(<ModalHarness onClose={() => undefined} />);
    const trigger = screen.getByRole("button", { name: "Open review" });
    await user.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "Review activation" });
    expect(screen.getByRole("button", { name: "Close dialog" })).toHaveFocus();

    await user.tab({ shift: true });
    expect(screen.getByRole("button", { name: "Activate" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: "Close dialog" })).toHaveFocus();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(dialog).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("closes on Escape", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<ModalHarness onClose={onClose} />);
    await user.click(screen.getByRole("button", { name: "Open review" }));
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledOnce();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
