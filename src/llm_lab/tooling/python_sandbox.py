"""Disposable, networkless Python execution through a fixed Docker contract."""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ToolExecutionError, ToolPolicyError
from .registry import ToolDefinition, ToolProvider


DEFAULT_IMAGE = "python@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534"
_MAX_CODE = 16 * 1024
_MAX_STDIO = 64 * 1024
_MAX_OUTPUT_FILE = 64 * 1024
_MAX_OUTPUT_FILES = 16


@dataclass(frozen=True)
class PythonSandboxSettings:
    enabled: bool = False
    docker_binary: str = "docker"
    image: str = DEFAULT_IMAGE
    timeout_seconds: float = 20.0
    memory: str = "256m"
    cpus: str = "1"
    pids_limit: int = 64

    @classmethod
    def from_environment(cls) -> "PythonSandboxSettings":
        return cls(
            enabled=os.environ.get("LLM_LAB_PYTHON_SANDBOX_ENABLED", "").strip() == "1",
            image=os.environ.get("LLM_LAB_PYTHON_SANDBOX_IMAGE", DEFAULT_IMAGE).strip(),
            timeout_seconds=float(os.environ.get("LLM_LAB_PYTHON_SANDBOX_TIMEOUT_SECONDS", "20")),
        )

    def __post_init__(self) -> None:
        if self.enabled and not shutil.which(self.docker_binary):
            raise ValueError("Python sandbox requires Docker")
        if not self.image.startswith("python@sha256:") or len(self.image) != len("python@sha256:") + 64:
            raise ValueError("Python sandbox image must use a pinned Python digest")
        if not 1 <= self.timeout_seconds <= 600 or not 16 <= self.pids_limit <= 256:
            raise ValueError("Python sandbox limits are invalid")


class PythonSandboxProvider(ToolProvider):
    """A fixed, non-shell Docker invocation; model input never reaches Docker flags."""

    def __init__(self, settings: PythonSandboxSettings) -> None:
        self.settings = settings
        self._tools = self._build_tools()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def tools(self) -> Sequence[ToolDefinition]:
        return self._tools

    async def aclose(self) -> None:
        return None

    def _build_tools(self) -> tuple[ToolDefinition, ...]:
        closed = {"additionalProperties": False}
        return (ToolDefinition(
            name="python_sandbox",
            description=(
                "Run bounded Python in a disposable, networkless sandbox. The host "
                "workspace is never mounted writable; output files are staged only."
            ),
            parameters={
                "type": "object", **closed, "required": ["code"],
                "properties": {
                    "code": {"type": "string", "minLength": 1, "maxLength": _MAX_CODE},
                    "profile": {"type": "string", "enum": ["python-base"], "default": "python-base"},
                },
            },
            output_schema={
                "type": "object", **closed,
                "required": ["profile", "exit_code", "stdout", "stderr", "truncated", "output_files"],
                "properties": {
                    "profile": {"type": "string"}, "exit_code": {"type": "integer"},
                    "stdout": {"type": "string", "maxLength": _MAX_STDIO},
                    "stderr": {"type": "string", "maxLength": _MAX_STDIO},
                    "truncated": {"type": "boolean"},
                    "output_files": {"type": "array", "maxItems": _MAX_OUTPUT_FILES, "items": {
                        "type": "object", **closed, "required": ["path", "sha256", "bytes"],
                        "properties": {"path": {"type": "string"}, "sha256": {"type": "string"}, "bytes": {"type": "integer", "minimum": 0}},
                    }},
                },
            },
            handler=self.run,
            available=self.enabled,
            execution_deadline_seconds=self.settings.timeout_seconds,
        ),)

    async def run(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise ToolPolicyError("python_sandbox_unavailable", "Python sandbox is not configured", retryable=True)
        code = arguments["code"]
        if not isinstance(code, str) or len(code.encode("utf-8")) > _MAX_CODE:
            raise ToolPolicyError("python_sandbox_denied", "Python source exceeds the reviewed size limit")
        profile = arguments.get("profile", "python-base")
        if profile != "python-base":
            raise ToolPolicyError("python_sandbox_denied", "Python profile is not available")
        with tempfile.TemporaryDirectory(prefix="llm-lab-python-") as temporary:
            stage = Path(temporary)
            stage.chmod(0o777)
            name = "llm-lab-python-" + secrets.token_hex(12)
            command = [
                self.settings.docker_binary, "run", "--rm", "--name", name,
                "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--pids-limit", str(self.settings.pids_limit),
                "--memory", self.settings.memory, "--cpus", self.settings.cpus,
                "--user", "65534:65534", "--workdir", "/workspace",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "--mount", f"type=bind,src={stage},dst=/workspace",
                self.settings.image, "python", "-I", "-B", "-c", code,
            ]
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                async with asyncio.timeout(self.settings.timeout_seconds):
                    stdout, stderr = await process.communicate()
            except TimeoutError as exc:
                process.kill()
                await process.wait()
                await self._kill_container(name)
                raise ToolExecutionError("python_sandbox_timeout", "Python execution exceeded its deadline", retryable=True) from exc
            truncated = len(stdout) > _MAX_STDIO or len(stderr) > _MAX_STDIO
            outputs = self._output_files(stage)
        return {
            "profile": profile, "exit_code": process.returncode or 0,
            "stdout": stdout[:_MAX_STDIO].decode("utf-8", "replace"),
            "stderr": stderr[:_MAX_STDIO].decode("utf-8", "replace"),
            "truncated": truncated, "output_files": outputs,
        }

    async def _kill_container(self, name: str) -> None:
        process = await asyncio.create_subprocess_exec(
            self.settings.docker_binary, "kill", name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()

    @staticmethod
    def _output_files(stage: Path) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for candidate in sorted(stage.rglob("*")):
            if len(outputs) == _MAX_OUTPUT_FILES:
                break
            if candidate.is_symlink() or not candidate.is_file():
                continue
            relative = candidate.relative_to(stage).as_posix()
            if candidate.stat().st_size > _MAX_OUTPUT_FILE:
                continue
            payload = candidate.read_bytes()
            outputs.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
        return outputs


__all__ = ["DEFAULT_IMAGE", "PythonSandboxProvider", "PythonSandboxSettings"]
