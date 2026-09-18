"""Bounded, approval-gated access to one operator-approved workspace."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import secrets
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ToolExecutionError, ToolPolicyError
from .registry import ToolDefinition, ToolProvider


_WORKSPACE_ID = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$")
_DENIED_NAMES = frozenset({".env", ".git", "id_rsa", "id_ed25519", "authorized_keys"})
_MAX_READ_BYTES = 64 * 1024
_MAX_WRITE_BYTES = 256 * 1024
_PROPOSAL_TTL_SECONDS = 300


@dataclass(frozen=True)
class WorkspaceSettings:
    root: Path | None = None
    workspace_id: str = "workspace"
    max_read_bytes: int = _MAX_READ_BYTES
    max_write_bytes: int = _MAX_WRITE_BYTES

    @classmethod
    def from_environment(cls) -> "WorkspaceSettings":
        value = os.environ.get("LLM_LAB_WORKSPACE_ROOT", "").strip()
        return cls(root=Path(value) if value else None)

    def __post_init__(self) -> None:
        if not _WORKSPACE_ID.fullmatch(self.workspace_id):
            raise ValueError("workspace id is invalid")
        if not 1024 <= self.max_read_bytes <= _MAX_READ_BYTES:
            raise ValueError("workspace read limit is invalid")
        if not 1024 <= self.max_write_bytes <= _MAX_WRITE_BYTES:
            raise ValueError("workspace write limit is invalid")
        if self.root is not None:
            if not self.root.is_absolute() or not self.root.is_dir() or self.root.is_symlink():
                raise ValueError("workspace root must be an existing non-symlink directory")


@dataclass(frozen=True)
class WorkspaceProposal:
    id: str
    path: str
    content: str
    expected_sha256: str | None
    expires_at: float


class WorkspaceToolProvider(ToolProvider):
    """Native workspace adapter; no arbitrary paths, symlink traversal, or deletes."""

    def __init__(self, settings: WorkspaceSettings) -> None:
        self.settings = settings
        self._root_fd: int | None = None
        self._proposals: dict[str, WorkspaceProposal] = {}
        self._proposal_lock = threading.Lock()
        if settings.root is not None:
            self._root_fd = os.open(settings.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._tools = self._build_tools()

    @property
    def enabled(self) -> bool:
        return self._root_fd is not None

    @property
    def tools(self) -> Sequence[ToolDefinition]:
        return self._tools

    async def aclose(self) -> None:
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None

    def _build_tools(self) -> tuple[ToolDefinition, ...]:
        closed = {"additionalProperties": False}
        return (
            ToolDefinition(
                name="workspace_list",
                description="List one approved workspace directory. Paths are relative and bounded.",
                parameters={"type": "object", **closed, "properties": {"path": {"type": "string", "maxLength": 512, "default": ""}}},
                output_schema={"type": "object", **closed, "required": ["workspace", "path", "entries"], "properties": {"workspace": {"type": "string"}, "path": {"type": "string"}, "entries": {"type": "array", "maxItems": 200, "items": {"type": "object", **closed, "required": ["name", "kind"], "properties": {"name": {"type": "string"}, "kind": {"type": "string", "enum": ["file", "directory"]}, "size": {"type": "integer", "minimum": 0}}}}}},
                handler=self.list_files,
                available=self.enabled,
            ),
            ToolDefinition(
                name="workspace_read",
                description="Read a bounded UTF-8 text file from the approved workspace.",
                parameters={"type": "object", **closed, "required": ["path"], "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 512}}},
                output_schema={"type": "object", **closed, "required": ["workspace", "path", "content", "sha256", "truncated"], "properties": {"workspace": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string", "maxLength": _MAX_READ_BYTES}, "sha256": {"type": "string"}, "truncated": {"type": "boolean"}}},
                handler=self.read_file,
                available=self.enabled,
            ),
            ToolDefinition(
                name="workspace_write_proposal",
                description="Propose one UTF-8 text-file write in the approved workspace. It never writes until a user approves the exact proposal.",
                parameters={"type": "object", **closed, "required": ["path", "content"], "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 512}, "content": {"type": "string", "maxLength": _MAX_WRITE_BYTES}, "expected_sha256": {"type": ["string", "null"], "maxLength": 64}}},
                output_schema={"type": "object", **closed, "required": ["proposal_id", "workspace", "path", "content_sha256", "expires_in_seconds", "approval_required"], "properties": {"proposal_id": {"type": "string"}, "workspace": {"type": "string"}, "path": {"type": "string"}, "content_sha256": {"type": "string"}, "expires_in_seconds": {"type": "integer"}, "approval_required": {"type": "boolean"}}},
                handler=self.propose_write,
                available=self.enabled,
            ),
        )

    def _parts(self, path: Any, *, allow_empty: bool = False) -> tuple[str, ...]:
        if not isinstance(path, str) or len(path) > 512 or "\x00" in path or path.startswith(("/", "\\")):
            raise ToolPolicyError("workspace_path_denied", "Workspace paths must be short relative paths")
        pieces = tuple(path.split("/")) if path else ()
        if not pieces and allow_empty:
            return pieces
        if not pieces or any(piece in {"", ".", ".."} or not _COMPONENT.fullmatch(piece) or piece in _DENIED_NAMES or piece.startswith(".") for piece in pieces):
            raise ToolPolicyError("workspace_path_denied", "Workspace path is not permitted")
        return pieces

    def _parent_fd(self, parts: tuple[str, ...]) -> tuple[int, str]:
        if self._root_fd is None:
            raise ToolPolicyError("workspace_unavailable", "Workspace tools are not configured", retryable=True)
        fd = os.dup(self._root_fd)
        try:
            for component in parts[:-1]:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd, parts[-1]
        except Exception:
            os.close(fd)
            raise

    def _directory_fd(self, parts: tuple[str, ...]) -> int:
        if self._root_fd is None:
            raise ToolPolicyError("workspace_unavailable", "Workspace tools are not configured", retryable=True)
        fd = os.dup(self._root_fd)
        try:
            for component in parts:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _regular(fd: int) -> os.stat_result:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ToolPolicyError("workspace_path_denied", "Only regular files are permitted")
        return info

    async def list_files(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._list_files, arguments.get("path", ""))

    def _list_files(self, path: Any) -> dict[str, Any]:
        parts = self._parts(path, allow_empty=True)
        fd = self._directory_fd(parts)
        try:
            entries: list[dict[str, Any]] = []
            for name in sorted(os.listdir(fd))[:200]:
                if name.startswith(".") or name in _DENIED_NAMES:
                    continue
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    entries.append({"name": name, "kind": "directory", "size": 0})
                elif stat.S_ISREG(info.st_mode):
                    entries.append({"name": name, "kind": "file", "size": info.st_size})
            return {"workspace": self.settings.workspace_id, "path": "/".join(parts), "entries": entries}
        finally:
            os.close(fd)

    async def read_file(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._read_file, arguments["path"])

    def _read_file(self, path: Any) -> dict[str, Any]:
        parts = self._parts(path)
        parent, name = self._parent_fd(parts)
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                self._regular(fd)
                content = os.read(fd, self.settings.max_read_bytes + 1)
            finally:
                os.close(fd)
        finally:
            os.close(parent)
        truncated = len(content) > self.settings.max_read_bytes
        content = content[: self.settings.max_read_bytes]
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError("workspace_not_text", "Workspace files must be valid UTF-8 text") from exc
        return {"workspace": self.settings.workspace_id, "path": "/".join(parts), "content": text, "sha256": hashlib.sha256(content).hexdigest(), "truncated": truncated}

    async def propose_write(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._propose_write, arguments)

    def _propose_write(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        parts = self._parts(arguments["path"])
        content = arguments["content"]
        if not isinstance(content, str) or len(content.encode("utf-8")) > self.settings.max_write_bytes:
            raise ToolPolicyError("workspace_write_denied", "Workspace write exceeds the reviewed size limit")
        expected = arguments.get("expected_sha256")
        if expected is not None and (not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)):
            raise ToolPolicyError("workspace_write_denied", "Expected file hash is invalid")
        proposal = WorkspaceProposal(secrets.token_urlsafe(24), "/".join(parts), content, expected, time.monotonic() + _PROPOSAL_TTL_SECONDS)
        with self._proposal_lock:
            self._proposals[proposal.id] = proposal
        return {"proposal_id": proposal.id, "workspace": self.settings.workspace_id, "path": proposal.path, "content_sha256": hashlib.sha256(content.encode()).hexdigest(), "expires_in_seconds": _PROPOSAL_TTL_SECONDS, "approval_required": True}

    async def approve(self, proposal_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._approve, proposal_id)

    def _approve(self, proposal_id: str) -> dict[str, Any]:
        with self._proposal_lock:
            proposal = self._proposals.pop(proposal_id, None)
        if proposal is None or proposal.expires_at < time.monotonic():
            raise ToolPolicyError("workspace_approval_denied", "Write approval is invalid or expired")
        parts = self._parts(proposal.path)
        parent, name = self._parent_fd(parts)
        temporary = f".llm-lab-{secrets.token_hex(12)}.tmp"
        payload = proposal.content.encode("utf-8")
        try:
            try:
                existing = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            except FileNotFoundError:
                if proposal.expected_sha256 is not None:
                    raise ToolPolicyError("workspace_conflict", "The expected file does not exist")
            else:
                try:
                    self._regular(existing)
                    existing_bytes = os.read(existing, self.settings.max_write_bytes + 1)
                finally:
                    os.close(existing)
                if proposal.expected_sha256 is None or hashlib.sha256(existing_bytes).hexdigest() != proposal.expected_sha256:
                    raise ToolPolicyError("workspace_conflict", "File changed or overwrite was not approved")
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            try:
                offset = 0
                while offset < len(payload):
                    offset += os.write(fd, payload[offset:])
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
        except Exception:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
            raise
        finally:
            os.close(parent)
        return {"workspace": self.settings.workspace_id, "path": proposal.path, "sha256": hashlib.sha256(payload).hexdigest(), "bytes_written": len(payload)}


__all__ = ["WorkspaceSettings", "WorkspaceToolProvider"]
