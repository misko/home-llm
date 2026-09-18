"""Strict public request and event schemas for model-managed tool turns."""

from __future__ import annotations

import base64
import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentDTO(BaseModel):
    """Closed, immutable wire objects for the agent API."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AgentTextPart(AgentDTO):
    type: Literal["text"]
    text: str = Field(max_length=131_072)


class AgentImageURL(AgentDTO):
    url: str = Field(max_length=14 * 1024 * 1024)

    @model_validator(mode="after")
    def safe_embedded_image(self) -> "AgentImageURL":
        header, separator, payload = self.url.partition(",")
        allowed_headers = {
            "data:image/jpeg;base64",
            "data:image/png;base64",
            "data:image/webp;base64",
        }
        if not separator or header.lower() not in allowed_headers:
            raise ValueError(
                "agent images must be embedded PNG, JPEG, or WebP data URLs"
            )
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("agent image data URL contains invalid base64") from exc
        if len(decoded) > 10 * 1024 * 1024:
            raise ValueError("agent images may not exceed 10 MiB")
        signatures = {
            "data:image/jpeg;base64": lambda value: value.startswith(b"\xff\xd8\xff"),
            "data:image/png;base64": lambda value: value.startswith(b"\x89PNG\r\n\x1a\n"),
            "data:image/webp;base64": lambda value: (
                len(value) >= 12
                and value.startswith(b"RIFF")
                and value[8:12] == b"WEBP"
            ),
        }
        if not signatures[header.lower()](decoded):
            raise ValueError("agent image bytes do not match the declared media type")
        return self


class AgentImagePart(AgentDTO):
    type: Literal["image_url"]
    image_url: AgentImageURL


AgentContentPart = AgentTextPart | AgentImagePart
AgentContentParts = Annotated[
    tuple[AgentContentPart, ...],
    Field(min_length=1, max_length=16),
]


class AgentMessage(AgentDTO):
    """An OpenAI-compatible conversation message.

    Agent turns allow only inline image data. In particular, remote image/file
    URLs are rejected so a caller cannot make the model backend perform SSRF.
    Raw ``/v1`` remains available for caller-managed OpenAI-compatible shapes.
    """

    role: Literal["user", "assistant"]
    content: str | AgentContentParts

    @model_validator(mode="after")
    def content_is_bounded(self) -> "AgentMessage":
        if isinstance(self.content, str):
            if len(self.content) > 131_072:
                raise ValueError("message text exceeds 131072 characters")
            return self
        return self


class AgentTurnRequest(AgentDTO):
    messages: tuple[AgentMessage, ...] = Field(min_length=1, max_length=64)
    toolset: str = Field(
        default="standard-readonly",
        pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$",
    )
    instructions: str | None = Field(default=None, max_length=16_384)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=32_000, ge=1, le=32_768)
    max_rounds: int | None = Field(default=None, ge=1, le=128)
    enabled_tools: tuple[str, ...] | None = Field(default=None, max_length=16)
    allow_workspace_writes: bool = False
    stream: Literal[True] = True

    @model_validator(mode="after")
    def request_is_bounded(self) -> "AgentTurnRequest":
        if self.enabled_tools is not None:
            if len(set(self.enabled_tools)) != len(self.enabled_tools):
                raise ValueError("enabled_tools must not contain duplicates")
            if any(
                re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", name) is None
                for name in self.enabled_tools
            ):
                raise ValueError("enabled_tools contains an invalid tool name")
        for index, message in enumerate(self.messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError(
                    "agent messages must start with user and strictly alternate "
                    "user/assistant roles"
                )
            if message.role == "assistant" and not isinstance(message.content, str):
                if any(isinstance(part, AgentImagePart) for part in message.content):
                    raise ValueError("image content parts are permitted only on user messages")
        if self.messages[-1].role != "user":
            raise ValueError("the final agent message must have role user")
        # Four 10 MiB decoded images occupy roughly 53.4 MiB as base64. Keep a
        # separate 56 MiB parsed-message ceiling below the 64 MiB raw ASGI cap.
        if self.instructions is not None:
            try:
                self.instructions.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("instructions must contain valid Unicode") from exc
        encoded = json.dumps(
            {
                "messages": [
                    message.model_dump(mode="json") for message in self.messages
                ],
                "instructions": self.instructions,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(encoded) > 56 * 1024 * 1024:
            raise ValueError("messages exceed the 56 MiB agent request limit")
        image_count = sum(
            1
            for message in self.messages
            if not isinstance(message.content, str)
            for part in message.content
            if isinstance(part, AgentImagePart)
        )
        if image_count > 4:
            raise ValueError("an agent turn may contain at most four images")
        return self


class ToolErrorDTO(AgentDTO):
    code: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")
    message: str = Field(max_length=1024)
    retryable: bool = False


class ToolSummary(AgentDTO):
    name: str
    description: str
    risk: Literal["low", "medium", "high"]
    read_only: bool
    available: bool
    effect: Literal["local", "open_world_search", "open_world_fetch", "open_world"]


class ToolsetSummary(AgentDTO):
    id: str
    name: str
    description: str
    tools: tuple[ToolSummary, ...]


class ToolsetsResponse(AgentDTO):
    schema_version: Literal[1] = 1
    toolsets: tuple[ToolsetSummary, ...]


class AgentEvent(AgentDTO):
    schema_version: Literal[1] = 1
    run_id: str
    sequence: int = Field(ge=1)
    type: str


class TurnStartedEvent(AgentEvent):
    type: Literal["turn.started"] = "turn.started"
    model: str
    toolset: str
    tools: tuple[str, ...]


class AssistantDeltaEvent(AgentEvent):
    type: Literal["assistant.delta"] = "assistant.delta"
    content: str
    reasoning: str | None = Field(default=None, max_length=262_144)
    round: int = Field(ge=1)


class ToolStartedEvent(AgentEvent):
    type: Literal["tool.started"] = "tool.started"
    call_id: str
    name: str
    arguments: dict[str, Any] | str
    round: int = Field(ge=1)


class ToolCompletedEvent(AgentEvent):
    type: Literal["tool.completed"] = "tool.completed"
    call_id: str
    name: str
    result: Any
    round: int = Field(ge=1)
    duration_ms: float = Field(ge=0)


class ToolFailedEvent(AgentEvent):
    type: Literal["tool.failed"] = "tool.failed"
    call_id: str
    name: str
    error: ToolErrorDTO
    round: int = Field(ge=1)
    duration_ms: float = Field(ge=0)


class TurnCompletedEvent(AgentEvent):
    type: Literal["turn.completed"] = "turn.completed"
    model: str
    rounds: int = Field(ge=1)
    finish_reason: str
    usage: dict[str, int]


class ErrorEvent(AgentEvent):
    type: Literal["error"] = "error"
    error: ToolErrorDTO


__all__ = [
    "AgentEvent",
    "AgentMessage",
    "AgentTurnRequest",
    "AssistantDeltaEvent",
    "ErrorEvent",
    "ToolCompletedEvent",
    "ToolErrorDTO",
    "ToolFailedEvent",
    "ToolStartedEvent",
    "ToolSummary",
    "ToolsetSummary",
    "ToolsetsResponse",
    "TurnCompletedEvent",
    "TurnStartedEvent",
]
