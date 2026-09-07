"""Reviewed MCP servers adapted into the common model tool registry.

Transport configuration is server-owned and never accepted by the agent turn
endpoint.  Every remote tool must also be explicitly allow-listed with a local
risk/read-only policy; discovery alone cannot make a tool executable.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, Protocol, Self
from urllib.parse import urlsplit

import httpx2
from mcp import Client as OfficialMCPClient
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ToolExecutionError
from .registry import TOOL_NAME_PATTERN, ToolDefinition, ToolProvider


class _MCPConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MCPAllowedTool(_MCPConfigModel):
    remote_name: str = Field(min_length=1, max_length=128)
    public_name: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$"
    )
    description: str = Field(min_length=1, max_length=1024)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    read_only: bool = True
    risk: Literal["low", "medium", "high"] = "low"
    # MCP runs outside this process's native-tool trust boundary. Even a
    # loopback server may have network or filesystem access, so it can never be
    # classified as a local/pure automatic tool.
    effect: Literal["open_world_search", "open_world_fetch", "open_world"] = (
        "open_world"
    )

    @model_validator(mode="after")
    def schema_is_an_object(self) -> Self:
        if self.input_schema.get("type") != "object":
            raise ValueError("reviewed MCP input schema must have type object")
        return self


class MCPServerConfig(_MCPConfigModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    namespace: str = Field(pattern=r"^[a-z][a-z0-9_]{0,23}$")
    transport: Literal["streamable_http"] = "streamable_http"
    url: str
    bearer_token_env: str | None = Field(
        default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$"
    )
    tools: tuple[MCPAllowedTool, ...] = Field(min_length=1, max_length=64)
    read_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    @model_validator(mode="after")
    def transport_is_closed(self) -> Self:
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("MCP URL must be an HTTP(S) endpoint without credentials")
        is_loopback = parsed.hostname == "localhost"
        try:
            is_loopback = is_loopback or ipaddress.ip_address(
                parsed.hostname
            ).is_loopback
        except ValueError:
            pass
        if parsed.scheme == "http" and not is_loopback:
            raise ValueError("plain HTTP MCP endpoints must use a loopback host")
        if not is_loopback and not self.bearer_token_env:
            raise ValueError("remote MCP endpoints require reviewed bearer-token auth")
        remote_names = [tool.remote_name for tool in self.tools]
        if len(remote_names) != len(set(remote_names)):
            raise ValueError("MCP tool allow-list contains duplicate remote names")
        public_names = [
            tool.public_name or _public_tool_name(self.namespace, tool.remote_name)
            for tool in self.tools
        ]
        if len(public_names) != len(set(public_names)):
            raise ValueError("MCP tool allow-list produces duplicate public names")
        return self


class MCPListToolsResult(Protocol):
    tools: Sequence[Any]
    next_cursor: str | None


class MCPCallToolResult(Protocol):
    content: Sequence[Any]
    structured_content: Any
    is_error: bool
    result_type: str


class MCPClient(Protocol):
    async def __aenter__(self) -> "MCPClient": ...

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...

    async def list_tools(self, *, cursor: str | None = None) -> MCPListToolsResult: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> MCPCallToolResult: ...


class MCPClientFactory(Protocol):
    def __call__(self, config: MCPServerConfig) -> MCPClient: ...


def _public_tool_name(namespace: str, remote_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", remote_name).strip("_")
    candidate = f"{namespace}__{normalized}"
    if not normalized or not TOOL_NAME_PATTERN.fullmatch(candidate):
        raise ValueError(
            f"MCP tool {remote_name!r} does not produce a valid namespaced name; "
            "configure public_name explicitly"
        )
    return candidate


class _OfficialMCPContext:
    """Own the official SDK client and any authenticated HTTP transport."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._http: httpx2.AsyncClient | None = None
        self._client: OfficialMCPClient | None = None

    async def __aenter__(self) -> MCPClient:
        headers: dict[str, str] = {}
        if self.config.bearer_token_env:
            token = os.environ.get(self.config.bearer_token_env)
            if not token:
                raise ToolExecutionError(
                    "mcp_auth_missing",
                    f"MCP server {self.config.id!r} authentication is not configured",
                )
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx2.AsyncClient(
            headers=headers,
            follow_redirects=False,
            trust_env=False,
        )
        await self._http.__aenter__()
        transport: Any = streamable_http_client(
            self.config.url, http_client=self._http
        )
        self._client = OfficialMCPClient(
            transport,
            read_timeout_seconds=self.config.read_timeout_seconds,
        )
        try:
            return await self._client.__aenter__()
        except BaseException:
            if self._http is not None:
                await self._http.__aexit__(None, None, None)
                self._http = None
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self._client is not None:
                await self._client.__aexit__(exc_type, exc, traceback)
        finally:
            self._client = None
            if self._http is not None:
                await self._http.__aexit__(exc_type, exc, traceback)
                self._http = None


def official_mcp_client(config: MCPServerConfig) -> MCPClient:
    """Build an official SDK context from trusted server-side configuration."""

    return _OfficialMCPContext(config)  # type: ignore[return-value]


class MCPAdapter(ToolProvider):
    """One connected, namespaced, operator-allow-listed MCP provider."""

    def __init__(
        self,
        config: MCPServerConfig,
        client: MCPClient,
        context: AbstractAsyncContextManager[MCPClient],
        discovered: Mapping[str, Any],
    ) -> None:
        self.config = config
        self._client = client
        self._context = context
        self._closed = False
        definitions: list[ToolDefinition] = []
        for policy in config.tools:
            remote = discovered[policy.remote_name]
            public_name = policy.public_name or _public_tool_name(
                config.namespace, policy.remote_name
            )
            input_schema = getattr(remote, "input_schema", None)
            if not isinstance(input_schema, Mapping):
                raise ValueError(f"MCP tool {policy.remote_name!r} has no input schema")
            if not _reviewed_schema_matches(input_schema, policy.input_schema):
                raise ValueError(
                    f"MCP tool {policy.remote_name!r} input schema drifted from "
                    "its reviewed configuration"
                )
            output_schema = getattr(remote, "output_schema", None)
            if not isinstance(output_schema, Mapping) or not _reviewed_schema_matches(
                output_schema, policy.output_schema
            ):
                raise ValueError(
                    f"MCP tool {policy.remote_name!r} output schema drifted from "
                    "its reviewed configuration"
                )

            async def invoke(
                arguments: Mapping[str, Any],
                *,
                remote_name: str = policy.remote_name,
            ) -> Any:
                return await self._invoke(remote_name, arguments)

            definitions.append(
                ToolDefinition(
                    name=public_name,
                    # Never inject remote descriptions into the model prompt.
                    description=policy.description,
                    parameters=policy.input_schema,
                    output_schema=policy.output_schema,
                    handler=invoke,
                    read_only=policy.read_only,
                    risk=policy.risk,
                    effect=policy.effect,
                )
            )
        self._tools = tuple(definitions)

    @classmethod
    async def connect(
        cls,
        config: MCPServerConfig,
        *,
        client_factory: MCPClientFactory = official_mcp_client,
    ) -> "MCPAdapter":
        context = client_factory(config)
        client = await context.__aenter__()
        try:
            discovered: dict[str, Any] = {}
            approved_names = {policy.remote_name for policy in config.tools}
            cursor: str | None = None
            for _ in range(16):
                page = await client.list_tools(cursor=cursor)
                for tool in page.tools:
                    name = getattr(tool, "name", None)
                    if isinstance(name, str) and name in approved_names:
                        discovered[name] = tool
                cursor = getattr(page, "next_cursor", None)
                if not cursor:
                    break
            else:
                raise ValueError("MCP tool discovery exceeded 16 pages")
            missing = {
                policy.remote_name for policy in config.tools
            }.difference(discovered)
            if missing:
                raise ValueError(
                    f"MCP server {config.id!r} is missing allow-listed tools: "
                    + ", ".join(sorted(missing))
                )
            return cls(config, client, context, discovered)
        except BaseException:
            await context.__aexit__(None, None, None)
            raise

    @property
    def tools(self) -> Sequence[ToolDefinition]:
        return self._tools

    async def _invoke(self, remote_name: str, arguments: Mapping[str, Any]) -> Any:
        if self._closed:
            raise ToolExecutionError("mcp_closed", "MCP server connection is closed")
        try:
            result = await self._client.call_tool(
                remote_name,
                arguments=dict(arguments),
                read_timeout_seconds=self.config.read_timeout_seconds,
            )
        except Exception as exc:
            raise ToolExecutionError(
                "mcp_unavailable",
                f"MCP server {self.config.id!r} is unavailable",
                retryable=True,
            ) from exc
        if getattr(result, "result_type", "complete") != "complete":
            raise ToolExecutionError(
                "mcp_input_required",
                "Interactive MCP elicitation is not enabled for automatic tool turns",
            )
        if result.is_error:
            raise ToolExecutionError(
                "mcp_tool_error",
                f"MCP tool {remote_name!r} reported an error",
            )
        if result.structured_content is not None:
            return result.structured_content
        raise ToolExecutionError(
            "mcp_unstructured_result",
            "MCP tool did not return its reviewed structured output",
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._context.__aexit__(None, None, None)


def _reviewed_schema_matches(
    discovered: Mapping[str, Any], reviewed: Mapping[str, Any]
) -> bool:
    """Bound drift comparison before retaining server-supplied schema objects."""

    try:
        encoded = json.dumps(
            discovered,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return False
    return len(encoded) <= 32 * 1024 and dict(discovered) == dict(reviewed)


__all__ = [
    "MCPAdapter",
    "MCPAllowedTool",
    "MCPClient",
    "MCPClientFactory",
    "MCPServerConfig",
    "official_mcp_client",
]
