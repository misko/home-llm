"""Per-turn effect and provenance policy for automatic tool execution."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .errors import ToolPolicyError
from .registry import ToolDefinition
from .schema import AgentTurnRequest


_PUBLIC_URL = re.compile(r"https?://[^\s<>\"']+", flags=re.IGNORECASE)


def _literal_user_urls(request: AgentTurnRequest) -> set[str]:
    latest = next(
        (message for message in reversed(request.messages) if message.role == "user"),
        None,
    )
    if latest is None:
        return set()
    if isinstance(latest.content, str):
        texts = [latest.content]
    else:
        texts = [
            part.text
            for part in latest.content
            if getattr(part, "type", None) == "text"
        ]
    return {
        match.group(0).rstrip(".,;:!?)]}")
        for text in texts
        for match in _PUBLIC_URL.finditer(text)
    }


class TurnToolPolicy:
    """Prevent untrusted open-world output from choosing the next destination."""

    def __init__(self, request: AgentTurnRequest) -> None:
        self._approved_fetch_urls = _literal_user_urls(request)
        self._search_completed = False
        self._open_world_closed = False

    def before(
        self,
        definition: ToolDefinition,
        arguments: Mapping[str, Any],
    ) -> None:
        effect = definition.effect
        if effect == "local":
            return
        if self._open_world_closed:
            raise ToolPolicyError(
                "open_world_chain_blocked",
                "Further open-world tool calls are blocked after an open-world fetch",
            )
        if effect == "open_world_search" and self._search_completed:
            raise ToolPolicyError(
                "open_world_chain_blocked",
                "A web search has already returned untrusted open-world content",
            )
        if effect == "open_world_fetch":
            url = arguments.get("url")
            if not isinstance(url, str) or url not in self._approved_fetch_urls:
                raise ToolPolicyError(
                    "fetch_url_not_approved",
                    "Web fetch URL must exactly match a URL from web search or the latest user message",
                )
        elif effect == "open_world" and self._search_completed:
            raise ToolPolicyError(
                "open_world_chain_blocked",
                "Another open-world tool has already returned untrusted content",
            )

    def available_to_model(self, definition: ToolDefinition) -> bool:
        """Hide exhausted open-world capabilities from later model rounds."""

        effect = definition.effect
        if self._open_world_closed:
            return False
        if effect == "local":
            return True
        if self._search_completed and effect in {"open_world_search", "open_world"}:
            return False
        return True

    def observe(self, definition: ToolDefinition, result: Any) -> None:
        effect = definition.effect
        if effect == "open_world_search":
            self._search_completed = True
            if isinstance(result, Mapping):
                candidates = result.get("results")
                if isinstance(candidates, list):
                    for candidate in candidates:
                        if isinstance(candidate, Mapping):
                            url = candidate.get("url")
                            if isinstance(url, str):
                                self._approved_fetch_urls.add(url)
            return
        if effect in {"open_world_fetch", "open_world"}:
            self._open_world_closed = True


__all__ = ["TurnToolPolicy"]
