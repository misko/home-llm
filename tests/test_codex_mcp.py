from __future__ import annotations

import httpx
import pytest

from llm_lab.codex_mcp import _gateway_url, query_local_llm
from mcp.server.mcpserver.exceptions import ToolError


@pytest.mark.asyncio
async def test_query_local_llm_uses_active_model_and_context() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"model": "local-test"})
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  useful answer  "}}]},
        )

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handle)
    ) as client:
        result = await query_local_llm(
            "Find the risk.", context="changed code", max_tokens=321, client=client
        )

    assert result == "useful answer"
    payload = __import__("json").loads(requests[1].content)
    assert payload["model"] == "local-test"
    assert payload["max_tokens"] == 321
    assert payload["messages"][1]["content"] == (
        "Context:\nchanged code\n\nTask:\nFind the risk."
    )


@pytest.mark.asyncio
async def test_query_local_llm_rejects_invalid_completion() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"model": "local-test"})
        return httpx.Response(200, json={"choices": []})

    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(ToolError, match="invalid chat completion"):
            await query_local_llm("hello", client=client)


def test_gateway_rejects_non_loopback_plain_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_LAB_GATEWAY_URL", "http://llm.example.test:14000")
    with pytest.raises(ToolError, match="loopback"):
        _gateway_url()


def test_gateway_allows_opted_in_private_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_LAB_GATEWAY_URL", "http://192.168.1.141:14000")
    monkeypatch.setenv("LLM_LAB_ALLOW_INSECURE_LAN", "1")
    assert _gateway_url() == "http://192.168.1.141:14000"


def test_gateway_rejects_opted_in_public_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_LAB_GATEWAY_URL", "http://8.8.8.8:14000")
    monkeypatch.setenv("LLM_LAB_ALLOW_INSECURE_LAN", "1")
    with pytest.raises(ToolError, match="loopback"):
        _gateway_url()
