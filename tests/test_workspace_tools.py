from __future__ import annotations

import os

import pytest

from llm_lab.tooling.executor import ToolExecutor
from llm_lab.tooling.registry import ToolRegistry, ToolsetDefinition
from llm_lab.tooling.workspace import WorkspaceSettings, WorkspaceToolProvider


@pytest.mark.asyncio
async def test_workspace_tools_read_and_approval_gate_writes(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text("hello", encoding="utf-8")
    provider = WorkspaceToolProvider(WorkspaceSettings(root=root))
    registry = ToolRegistry()
    registry.register_provider(provider)
    registry.register_toolset(ToolsetDefinition(
        id="workspace-files", name="Workspace", description="Workspace", tools=tuple(tool.name for tool in provider.tools),
    ))
    executor = ToolExecutor(registry)
    permitted = tuple(tool.name for tool in registry.resolve("workspace-files"))
    try:
        listed = await executor.execute("workspace_list", {}, permitted=permitted)
        read = await executor.execute("workspace_read", {"path": "notes.txt"}, permitted=permitted)
        proposal = await executor.execute(
            "workspace_write_proposal",
            {"path": "draft.txt", "content": "approved content"},
            permitted=permitted,
        )
        assert listed.ok and listed.value["entries"] == [{"name": "notes.txt", "kind": "file", "size": 5}]
        assert read.ok and read.value["content"] == "hello"
        assert proposal.ok and not (root / "draft.txt").exists()
        committed = await provider.approve(proposal.value["proposal_id"])
        assert committed["bytes_written"] == 16
        assert (root / "draft.txt").read_text() == "approved content"
        with pytest.raises(Exception):
            await provider.approve(proposal.value["proposal_id"])
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_workspace_tools_reject_escapes_symlinks_and_unapproved_overwrite(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.symlink(outside, root / "link.txt")
    provider = WorkspaceToolProvider(WorkspaceSettings(root=root))
    registry = ToolRegistry()
    registry.register_provider(provider)
    registry.register_toolset(ToolsetDefinition(
        id="workspace-files", name="Workspace", description="Workspace", tools=tuple(tool.name for tool in provider.tools),
    ))
    executor = ToolExecutor(registry)
    permitted = tuple(tool.name for tool in registry.resolve("workspace-files"))
    try:
        for path in ("../outside.txt", "/etc/passwd", "link.txt", ".env"):
            result = await executor.execute("workspace_read", {"path": path}, permitted=permitted)
            assert not result.ok
        (root / "exists.txt").write_text("before", encoding="utf-8")
        proposed = await executor.execute(
            "workspace_write_proposal", {"path": "exists.txt", "content": "after"}, permitted=permitted,
        )
        assert proposed.ok
        with pytest.raises(Exception):
            await provider.approve(proposed.value["proposal_id"])
        assert (root / "exists.txt").read_text() == "before"
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_workspace_write_requires_turn_permission_and_stays_bounded(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    provider = WorkspaceToolProvider(WorkspaceSettings(root=root))
    registry = ToolRegistry()
    registry.register_provider(provider)
    registry.register_toolset(ToolsetDefinition(
        id="workspace-files", name="Workspace", description="Workspace", tools=tuple(tool.name for tool in provider.tools),
    ))
    executor = ToolExecutor(registry)
    permitted = tuple(tool.name for tool in registry.resolve("workspace-files"))
    try:
        denied = await executor.execute(
            "workspace_write", {"path": "draft.txt", "content": "bounded write"}, permitted=permitted,
        )
        assert not denied.ok
        assert denied.error and denied.error.code == "tool_requires_approval"
        committed = await executor.execute(
            "workspace_write", {"path": "draft.txt", "content": "bounded write"}, permitted=permitted,
            allow_workspace_writes=True,
        )
        assert committed.ok and committed.value["approval_required"] is False
        assert (root / "draft.txt").read_text() == "bounded write"
        escaped = await executor.execute(
            "workspace_write", {"path": "../outside.txt", "content": "no"}, permitted=permitted,
            allow_workspace_writes=True,
        )
        assert not escaped.ok and escaped.error and escaped.error.code == "workspace_path_denied"
    finally:
        await provider.aclose()
