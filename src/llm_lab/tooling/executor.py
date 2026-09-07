"""Schema-validating, budgeted execution of reviewed read-only tools."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping

import jsonschema

from .errors import ToolExecutionError, ToolPolicyError, ToolingError
from .registry import ToolDefinition, ToolRegistry


@dataclass(frozen=True)
class ToolExecution:
    ok: bool
    value: Any | None = None
    error: ToolingError | None = None


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        timeout_seconds: float = 20.0,
        max_result_bytes: int = 64 * 1024,
    ) -> None:
        if timeout_seconds <= 0 or max_result_bytes < 1024:
            raise ValueError("invalid tool executor limits")
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.max_result_bytes = max_result_bytes

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        permitted: tuple[str, ...],
    ) -> ToolExecution:
        try:
            definition = self.authorize(name, arguments, permitted=permitted)
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    value = await definition.handler(arguments)
            except TimeoutError as exc:
                raise ToolExecutionError(
                    "tool_timeout",
                    f"Tool {name!r} exceeded its execution deadline",
                    retryable=True,
                ) from exc
            except asyncio.CancelledError:
                raise
            except ToolingError:
                raise
            except Exception as exc:
                raise ToolExecutionError(
                    "tool_failed",
                    f"Tool {name!r} failed",
                    retryable=False,
                ) from exc
            value = self._bounded(value)
            try:
                definition.output_validator.validate(value)
            except jsonschema.ValidationError as exc:
                raise ToolExecutionError(
                    "invalid_tool_result",
                    f"Tool {name!r} returned a value outside its reviewed output schema",
                ) from exc
            except Exception as exc:
                raise ToolExecutionError(
                    "tool_schema_error",
                    f"The reviewed output schema for {name!r} could not be evaluated",
                ) from exc
            return ToolExecution(ok=True, value=value)
        except asyncio.CancelledError:
            raise
        except ToolingError as exc:
            return ToolExecution(ok=False, error=exc)

    def authorize(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        permitted: tuple[str, ...],
    ) -> ToolDefinition:
        definition = self.registry.get(name, permitted=permitted)
        try:
            definition.validator.validate(dict(arguments))
        except jsonschema.ValidationError as exc:
            raise ToolPolicyError(
                "invalid_tool_arguments",
                f"Arguments for {name!r} do not match its reviewed schema",
            ) from exc
        except Exception as exc:
            raise ToolPolicyError(
                "tool_schema_error",
                f"The reviewed schema for {name!r} could not be evaluated",
            ) from exc
        return definition

    def _bounded(self, value: Any) -> Any:
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(
                "invalid_tool_result",
                "Tool returned a value that is not valid JSON",
            ) from exc
        if len(encoded) <= self.max_result_bytes:
            return value
        raise ToolExecutionError(
            "tool_result_too_large",
            "Tool result exceeded the configured serialized-byte limit",
        )


__all__ = ["ToolExecution", "ToolExecutor"]
