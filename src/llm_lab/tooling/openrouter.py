"""Bounded server-side delegation to operator-approved OpenRouter models."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from .errors import ToolExecutionError, ToolPolicyError
from .registry import ToolDefinition, ToolProvider


_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_MAX_PROMPT_CHARACTERS = 32_768
_MAX_OUTPUT_CHARACTERS = 65_536
_FINAL_ANSWER_RETRY_SUFFIX = (
    "\n\nReturn a concise final answer now. Do not provide reasoning, tool calls, "
    "or analysis; put the answer in the response content."
)


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str = ""
    allowed_models: tuple[str, ...] = ()
    timeout_seconds: float = 90.0
    execution_deadline_seconds: float = 120.0
    max_output_tokens: int = 4_096

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.allowed_models)

    @classmethod
    def from_environment(cls) -> "OpenRouterSettings":
        models = tuple(item.strip() for item in os.environ.get("LLM_LAB_OPENROUTER_ALLOWED_MODELS", "").split(",") if item.strip())
        value = cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
            allowed_models=models,
            timeout_seconds=float(os.environ.get("LLM_LAB_OPENROUTER_TIMEOUT_SECONDS", "90")),
            execution_deadline_seconds=float(os.environ.get("LLM_LAB_OPENROUTER_EXECUTION_TIMEOUT_SECONDS", "120")),
            max_output_tokens=int(os.environ.get("LLM_LAB_OPENROUTER_MAX_OUTPUT_TOKENS", "4096")),
        )
        if not 1 <= value.timeout_seconds <= 7200 or not 1 <= value.execution_deadline_seconds <= 7200 or not 1 <= value.max_output_tokens <= 16_384:
            raise ValueError("OpenRouter limits are outside reviewed bounds")
        if value.execution_deadline_seconds < value.timeout_seconds:
            raise ValueError("OpenRouter execution deadline must cover its HTTP timeout")
        if len(value.allowed_models) > 128 or any(len(model) > 128 for model in value.allowed_models):
            raise ValueError("OpenRouter model allow-list is invalid")
        return value


class OpenRouterProvider(ToolProvider):
    def __init__(self, settings: OpenRouterSettings, *, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.timeout_seconds)
        self._owns_client = client is None
        self._tools = self._build_tools()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def tools(self) -> Sequence[ToolDefinition]:
        return self._tools

    def _build_tools(self) -> tuple[ToolDefinition, ...]:
        closed = {"additionalProperties": False}
        return (ToolDefinition(
            name="openrouter_delegate",
            description="Ask one operator-approved remote OpenRouter model for a bounded second opinion. Choose model from the supplied enum; do not invent provider or model IDs. Send only the minimum task context needed; this sends the prompt to a third party.",
            parameters={"type": "object", **closed, "required": ["model", "prompt"], "properties": {"model": {"type": "string", "maxLength": 128, "enum": list(self.settings.allowed_models), "description": "One of the operator-approved OpenRouter model IDs."}, "prompt": {"type": "string", "minLength": 1, "maxLength": _MAX_PROMPT_CHARACTERS}}},
            output_schema={"type": "object", **closed, "required": ["model", "content", "usage"], "properties": {"model": {"type": "string"}, "content": {"type": "string", "maxLength": _MAX_OUTPUT_CHARACTERS}, "usage": {"type": "object", **closed, "properties": {"prompt_tokens": {"type": "integer", "minimum": 0}, "completion_tokens": {"type": "integer", "minimum": 0}, "total_tokens": {"type": "integer", "minimum": 0}}}}},
            handler=self.delegate,
            # It is a read-only, bounded request in a dedicated toolset. The
            # provider boundary remains visible through the open-world effect.
            risk="low",
            effect="open_world",
            execution_deadline_seconds=self.settings.execution_deadline_seconds,
            available=self.enabled,
        ),)

    async def delegate(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise ToolPolicyError("openrouter_unavailable", "OpenRouter delegation is not configured", retryable=True)
        model, prompt = arguments["model"], arguments["prompt"]
        if model not in self.settings.allowed_models:
            raise ToolPolicyError("openrouter_model_denied", "Requested OpenRouter model is not operator-approved")
        content, usage = await self._request(model, prompt)
        if not content.strip():
            content, usage = await self._request(model, prompt + _FINAL_ANSWER_RETRY_SUFFIX)
        if not content.strip():
            raise ToolExecutionError(
                "openrouter_no_final_answer",
                "OpenRouter returned no final answer after a bounded retry.",
                retryable=True,
            )
        return {"model": model, "content": content[:_MAX_OUTPUT_CHARACTERS], "usage": usage}

    async def _request(self, model: str, prompt: str) -> tuple[str, dict[str, int]]:
        try:
            response = await self._client.post(_API_URL, headers={"Authorization": f"Bearer {self.settings.api_key}", "Content-Type": "application/json"}, json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": self.settings.max_output_tokens, "stream": False})
        except httpx.TimeoutException as exc:
            raise ToolExecutionError("openrouter_timeout", "OpenRouter delegation exceeded its deadline", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ToolExecutionError("openrouter_unavailable", "OpenRouter delegation is unavailable", retryable=True) from exc
        if response.status_code >= 400:
            raise ToolExecutionError("openrouter_rejected", "OpenRouter rejected the delegated request", retryable=response.status_code >= 500)
        try:
            document = response.json(); choice = document["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ToolExecutionError("openrouter_invalid_response", "OpenRouter returned an invalid response", retryable=True) from exc
        refusal = choice.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise ToolExecutionError("openrouter_refusal", "The delegated OpenRouter model declined this request.")
        content = choice.get("content", "")
        if not isinstance(content, str):
            content = ""
        usage = document.get("usage") if isinstance(document, Mapping) else {}
        normalized_usage = {key: value for key, value in (usage.items() if isinstance(usage, Mapping) else ()) if key in {"prompt_tokens", "completion_tokens", "total_tokens"} and isinstance(value, int) and value >= 0}
        return content, normalized_usage

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
