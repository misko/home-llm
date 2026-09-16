"""Expose the active LLM Lab model to Codex as a local MCP tool."""

from __future__ import annotations

import ipaddress
import os
from typing import Annotated, Any
from urllib.parse import urlsplit

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

DEFAULT_GATEWAY_URL = "http://127.0.0.1:14000"
DEFAULT_TIMEOUT_SECONDS = 300.0

server = MCPServer(
    "llm-lab-local",
    title="LLM Lab local model",
    description="Send bounded analysis tasks to the active private LLM Lab model.",
    instructions=(
        "Use ask_local_llm for bounded, read-only analysis, summarization, "
        "brainstorming, and second-opinion review. Provide all required context "
        "in the tool call and independently verify important conclusions."
    ),
)


def _gateway_url() -> str:
    raw = os.environ.get("LLM_LAB_GATEWAY_URL", DEFAULT_GATEWAY_URL).rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ToolError("LLM_LAB_GATEWAY_URL must be an HTTP(S) URL")
    try:
        is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        is_loopback = parsed.hostname == "localhost"
    if parsed.scheme == "http" and not is_loopback:
        raise ToolError("Plain HTTP LLM Lab gateways must use a loopback address")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ToolError("LLM_LAB_GATEWAY_URL must not contain credentials or query data")
    return raw


def _headers() -> dict[str, str]:
    token = os.environ.get("LLM_LAB_GATEWAY_API_KEY")
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _active_model(client: httpx.AsyncClient) -> str:
    configured = os.environ.get("LLM_LAB_CODEX_MODEL")
    if configured:
        return configured
    response = await client.get("/health")
    response.raise_for_status()
    payload = response.json()
    model = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(model, str) or not model:
        raise ToolError("LLM Lab did not report an active model")
    return model


def _response_text(payload: Any) -> str:
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ToolError("LLM Lab returned an invalid chat completion") from exc
    if not isinstance(text, str) or not text.strip():
        raise ToolError("LLM Lab returned an empty chat completion")
    return text.strip()


async def query_local_llm(
    prompt: str,
    *,
    context: str = "",
    max_tokens: int = 2048,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Query the active local model, optionally using an injected test client."""

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            base_url=_gateway_url(),
            headers=_headers(),
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=False,
            trust_env=False,
        )
    try:
        model = await _active_model(client)
        user_content = prompt
        if context:
            user_content = f"Context:\n{context}\n\nTask:\n{prompt}"
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a private local assistant working for a primary "
                            "coding agent. Complete only the bounded task provided. "
                            "Be concise, identify uncertainty, and do not claim to have "
                            "read files or run commands unless their results are included "
                            "in the supplied context."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()
        return _response_text(response.json())
    except httpx.HTTPStatusError as exc:
        raise ToolError(
            f"LLM Lab request failed with HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ToolError(f"LLM Lab request failed: {exc}") from exc
    finally:
        if owns_client:
            await client.aclose()


@server.tool(
    name="ask_local_llm",
    description=(
        "Ask the active private LLM Lab model to analyze supplied text. Best for "
        "summaries, brainstorming, first-pass review, classification, and bounded "
        "second opinions. The local model cannot read files or run commands, so pass "
        "all necessary source text in context and independently verify its conclusions."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def ask_local_llm(
    prompt: Annotated[str, Field(min_length=1, max_length=20_000)],
    context: Annotated[str, Field(max_length=100_000)] = "",
) -> str:
    """Send a bounded task and optional context to the active local model."""

    return await query_local_llm(prompt, context=context)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
