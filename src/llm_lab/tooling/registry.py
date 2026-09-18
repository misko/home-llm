"""Reviewed tool registry shared by every active model deployment."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import jsonschema

from .errors import ToolPolicyError
from .schema import ToolSummary, ToolsetSummary, ToolsetsResponse


TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TOOLSET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
TOOL_RISKS = frozenset({"low", "medium", "high"})
TOOL_EFFECTS = frozenset(
    {"local", "open_world_search", "open_world_fetch", "open_world"}
)
ToolHandler = Callable[[Mapping[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    handler: ToolHandler
    read_only: bool = True
    risk: Literal["low", "medium", "high"] = "low"
    effect: Literal[
        "local", "open_world_search", "open_world_fetch", "open_world"
    ] = "local"
    execution_deadline_seconds: float | None = None
    available: bool = True
    validator: jsonschema.Draft202012Validator = field(
        init=False, repr=False, compare=False
    )
    output_validator: jsonschema.Draft202012Validator = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.name) is not str or not TOOL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"invalid OpenAI tool name: {self.name!r}")
        if (
            type(self.description) is not str
            or not self.description.strip()
            or len(self.description) > 1024
        ):
            raise ValueError("tool descriptions must contain at most 1024 characters")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("tool input schema must be an object")
        if not isinstance(self.output_schema, Mapping):
            raise ValueError("tool output schema must be an object")
        if not callable(self.handler):
            raise ValueError("tool handler must be callable")
        if type(self.read_only) is not bool:
            raise ValueError("tool read_only must be a boolean")
        if type(self.available) is not bool:
            raise ValueError("tool available must be a boolean")
        if type(self.risk) is not str or self.risk not in TOOL_RISKS:
            raise ValueError("tool risk must be low, medium, or high")
        if type(self.effect) is not str or self.effect not in TOOL_EFFECTS:
            raise ValueError("tool effect is unsupported")
        if self.execution_deadline_seconds is not None and (
            type(self.execution_deadline_seconds) not in (int, float)
            or not 0 < self.execution_deadline_seconds <= 7200
        ):
            raise ValueError("tool execution deadline must be between 0 and 7200 seconds")
        canonical, validator = _compile_schema(
            self.parameters, name=self.name, purpose="input"
        )
        output, output_validator = _compile_schema(
            self.output_schema, name=self.name, purpose="output"
        )
        if canonical.get("type") != "object":
            raise ValueError(f"tool {self.name!r} requires an object input schema")
        # Freeze a detached canonical copy, not a provider-owned mutable map.
        object.__setattr__(self, "parameters", canonical)
        object.__setattr__(self, "validator", validator)
        object.__setattr__(self, "output_schema", output)
        object.__setattr__(self, "output_validator", output_validator)

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class ToolsetDefinition:
    id: str
    name: str
    description: str
    tools: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.id) is not str or not TOOLSET_ID_PATTERN.fullmatch(self.id):
            raise ValueError(f"invalid toolset id: {self.id!r}")
        if (
            type(self.name) is not str
            or not self.name.strip()
            or len(self.name) > 128
        ):
            raise ValueError("toolset names must contain at most 128 characters")
        if (
            type(self.description) is not str
            or not self.description.strip()
            or len(self.description) > 1024
        ):
            raise ValueError(
                "toolset descriptions must contain at most 1024 characters"
            )
        if isinstance(self.tools, (str, bytes)) or not isinstance(
            self.tools, Sequence
        ):
            raise ValueError("toolset tools must be a sequence")
        tools = tuple(self.tools)
        if not tools:
            raise ValueError("toolsets must contain at least one tool")
        if len(tools) > 64:
            raise ValueError("toolsets may contain at most 64 tools")
        if any(
            type(tool) is not str or not TOOL_NAME_PATTERN.fullmatch(tool)
            for tool in tools
        ):
            raise ValueError("toolsets contain an invalid tool name")
        if len(set(tools)) != len(tools):
            raise ValueError("toolsets may not contain duplicate tools")
        object.__setattr__(self, "tools", tools)


class ToolProvider(Protocol):
    @property
    def tools(self) -> Sequence[ToolDefinition]: ...

    async def aclose(self) -> None: ...


class ToolRegistry:
    """Immutable-by-convention registry of reviewed executable handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._toolsets: dict[str, ToolsetDefinition] = {}
        self._providers: list[ToolProvider] = []

    def register(self, definition: ToolDefinition) -> None:
        if not isinstance(definition, ToolDefinition):
            raise TypeError("registry entries must be ToolDefinition instances")
        if definition.name in self._tools:
            raise ValueError(f"duplicate tool name: {definition.name}")
        self._tools[definition.name] = definition

    def register_provider(self, provider: ToolProvider) -> None:
        definitions = tuple(provider.tools)
        if any(not isinstance(item, ToolDefinition) for item in definitions):
            raise TypeError("provider entries must be ToolDefinition instances")
        names = tuple(definition.name for definition in definitions)
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        collisions = set(names).intersection(self._tools)
        if duplicates or collisions:
            conflicts = sorted(duplicates | collisions)
            raise ValueError("duplicate tool name: " + ", ".join(conflicts))
        # Registration is atomic. If preflight fails, the caller retains
        # ownership of the connected provider and must close it.
        for definition in definitions:
            self._tools[definition.name] = definition
        self._providers.append(provider)

    def register_toolset(self, definition: ToolsetDefinition) -> None:
        if not isinstance(definition, ToolsetDefinition):
            raise TypeError("toolsets must be ToolsetDefinition instances")
        if definition.id in self._toolsets:
            raise ValueError(f"duplicate toolset id: {definition.id}")
        unknown = set(definition.tools).difference(self._tools)
        if unknown:
            raise ValueError(
                f"toolset {definition.id!r} references unknown tools: "
                + ", ".join(sorted(unknown))
            )
        self._toolsets[definition.id] = definition

    def resolve(self, toolset_id: str) -> tuple[ToolDefinition, ...]:
        try:
            toolset = self._toolsets[toolset_id]
        except KeyError as exc:
            raise ToolPolicyError(
                "unknown_toolset",
                f"Unknown toolset {toolset_id!r}",
            ) from exc
        return tuple(
            self._tools[name]
            for name in toolset.tools
            if self._tools[name].available
        )

    def get(
        self,
        name: str,
        *,
        permitted: Sequence[str],
        allow_workspace_writes: bool = False,
    ) -> ToolDefinition:
        if name not in permitted:
            raise ToolPolicyError(
                "tool_not_permitted",
                f"Tool {name!r} is not permitted by this toolset",
            )
        try:
            definition = self._tools[name]
        except KeyError as exc:
            raise ToolPolicyError(
                "unknown_tool",
                f"Unknown tool {name!r}",
            ) from exc
        if not definition.available:
            raise ToolPolicyError(
                "tool_unavailable",
                f"Tool {name!r} is not configured",
                retryable=True,
            )
        if (
            not definition.read_only
            or definition.risk != "low"
        ) and not (allow_workspace_writes and name == "workspace_write"):
            raise ToolPolicyError(
                "tool_requires_approval",
                f"Tool {name!r} cannot run in an automatic read-only turn",
            )
        return definition

    def describe(self) -> ToolsetsResponse:
        summaries: list[ToolsetSummary] = []
        for toolset in self._toolsets.values():
            summaries.append(
                ToolsetSummary(
                    id=toolset.id,
                    name=toolset.name,
                    description=toolset.description,
                    tools=tuple(
                        ToolSummary(
                            name=self._tools[name].name,
                            description=self._tools[name].description,
                            risk=self._tools[name].risk,
                            read_only=self._tools[name].read_only,
                            available=self._tools[name].available,
                            effect=self._tools[name].effect,
                        )
                        for name in toolset.tools
                    ),
                )
            )
        return ToolsetsResponse(toolsets=tuple(summaries))

    async def aclose(self) -> None:
        for provider in reversed(self._providers):
            await provider.aclose()


def _validate_safe_schema(schema: Any) -> None:
    """Reject expensive or network-capable JSON Schema features.

    Schemas are copied, size-limited, and walked before compilation. Tool input
    validation intentionally supports a conservative subset; remote references
    and regular expressions are unnecessary for function arguments and can
    perform I/O or unbounded work in third-party validator extensions.
    """

    forbidden = {
        "$defs",
        "$dynamicRef",
        "$recursiveRef",
        "$ref",
        "allOf",
        "anyOf",
        "contains",
        "definitions",
        "dependentSchemas",
        "else",
        "if",
        "not",
        "oneOf",
        "pattern",
        "patternProperties",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
    }
    nodes = 0

    def visit(value: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 512 or depth > 16:
            raise ValueError("tool schema exceeds structural limits")
        if isinstance(value, dict):
            blocked = forbidden.intersection(value)
            if blocked:
                raise ValueError(
                    "tool schema uses unsupported keywords: "
                    + ", ".join(sorted(blocked))
                )
            for key, child in value.items():
                if not isinstance(key, str) or len(key) > 256:
                    raise ValueError("tool schema contains an invalid key")
                visit(child, depth + 1)
        elif isinstance(value, list):
            if len(value) > 256:
                raise ValueError("tool schema array exceeds structural limits")
            for child in value:
                visit(child, depth + 1)
        elif isinstance(value, str) and len(value) > 4096:
            raise ValueError("tool schema string exceeds structural limits")

    visit(schema, 0)


def _compile_schema(
    schema: Mapping[str, Any],
    *,
    name: str,
    purpose: str,
) -> tuple[dict[str, Any], jsonschema.Draft202012Validator]:
    try:
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tool {name!r} {purpose} schema must be JSON") from exc
    if len(encoded) > 32 * 1024:
        raise ValueError(f"tool {name!r} {purpose} schema exceeds 32 KiB")
    _validate_safe_schema(canonical)
    try:
        jsonschema.Draft202012Validator.check_schema(canonical)
        validator = jsonschema.Draft202012Validator(canonical)
    except jsonschema.SchemaError as exc:
        raise ValueError(f"tool {name!r} has an invalid {purpose} schema") from exc
    return canonical, validator


__all__ = [
    "TOOL_NAME_PATTERN",
    "TOOLSET_ID_PATTERN",
    "ToolDefinition",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "ToolsetDefinition",
]
