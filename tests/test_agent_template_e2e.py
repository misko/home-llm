from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from llm_lab.gateway import create_app
from llm_lab.paths import LabPaths
from llm_lab.runtime import (
    LaunchRecord,
    RuntimeState,
    build_backend_command,
    write_active_state,
)
from llm_lab.schema import BackendKind, DeploymentSpec
from llm_lab.tooling.builtins import BuiltinToolSettings, create_builtin_registry
from llm_lab.tooling.orchestrator import AgentRunner, OpenAIChatBackend
from llm_lab.tooling.registry import ToolDefinition, ToolRegistry, ToolsetDefinition


def _active_paths(tmp_path: Path) -> LabPaths:
    repo = tmp_path / "repo"
    repo.mkdir()
    paths = LabPaths.discover(repo_root=repo, data_root=tmp_path / "data")
    paths.initialize()
    deployment = DeploymentSpec(
        id="qwen-template-test",
        artifact_id="test-artifact",
        public_alias="local-qwen-test",
        backend=BackendKind.EXTERNAL,
        external_base_url="http://backend.invalid/v1",
        startup_timeout_seconds=1,
    )
    write_active_state(
        paths,
        RuntimeState(
            schema_version=1,
            phase="ready",
            deployment=deployment,
            artifact_path=None,
            plan=build_backend_command(deployment, paths=paths),
            launch=LaunchRecord(
                kind="external",
                command=(),
                started_at="2026-09-14T00:00:00+00:00",
            ),
            activated_at="2026-09-14T00:00:00+00:00",
        ),
    )
    return paths


def _registry() -> ToolRegistry:
    async def search(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "query": arguments["query"],
            "results": [
                {
                    "title": "Approved source",
                    "url": "https://approved.example/image-models",
                    "snippet": "Benchmark evidence",
                }
            ],
            "result_count": 1,
        }

    async def fetch(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "url": arguments["url"],
            "status": 200,
            "content_type": "text/plain",
            "title": "Approved source",
            "text": "The benchmark evidence is available.",
            "truncated": False,
        }

    closed = {"additionalProperties": False}
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="web_search",
            description="Search",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                **closed,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "results": {"type": "array", "items": {"type": "object"}},
                    "result_count": {"type": "integer"},
                },
                "required": ["query", "results", "result_count"],
                **closed,
            },
            handler=search,
            effect="open_world_search",
        )
    )
    registry.register(
        ToolDefinition(
            name="web_fetch",
            description="Fetch",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                **closed,
            },
            output_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "status": {"type": "integer"},
                    "content_type": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "text": {"type": "string"},
                    "truncated": {"type": "boolean"},
                },
                "required": [
                    "url",
                    "status",
                    "content_type",
                    "title",
                    "text",
                    "truncated",
                ],
                **closed,
            },
            handler=fetch,
            effect="open_world_fetch",
        )
    )
    registry.register_toolset(
        ToolsetDefinition(
            id="standard-readonly",
            name="Standard",
            description="Standard",
            tools=("web_search", "web_fetch"),
        )
    )
    return registry


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def _completion(content: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }


def _sse_events(response: httpx.Response) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_qwen_template_accepts_textual_tool_call_synthesis_retry(
    tmp_path: Path,
) -> None:
    paths = _active_paths(tmp_path)
    backend_app = FastAPI()
    backend_requests: list[dict[str, Any]] = []

    @backend_app.post("/v1/chat/completions", response_model=None)
    async def completion(request: Request) -> dict[str, Any] | JSONResponse:
        payload = await request.json()
        backend_requests.append(payload)
        system_positions = [
            index
            for index, message in enumerate(payload["messages"])
            if message["role"] == "system"
        ]
        if system_positions != [0]:
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": "Jinja Exception: System message must be at the beginning.",
                        "type": "server_error",
                    }
                },
            )
        match len(backend_requests):
            case 1:
                return _tool_call(
                    "web_search", {"query": "best image models"}, "search"
                )
            case 2:
                return _tool_call(
                    "web_fetch",
                    {"url": "https://approved.example/image-models"},
                    "fetch",
                )
            case 3:
                return _completion(
                    "<tool_call><function=web_fetch></function></tool_call>"
                )
            case _:
                final_message = payload["messages"][-1]
                if (
                    final_message["role"] == "user"
                    and "Tool execution is complete" in final_message["content"]
                ):
                    return _completion("The evidence supports the final answer.")
                return _completion(
                    "<tool_call><function=web_fetch></function></tool_call>"
                )

    backend_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=backend_app),
        base_url="http://backend.invalid",
    )
    runner = AgentRunner(
        _registry(), OpenAIChatBackend(lambda: backend_client)
    )
    app = create_app(paths, enable_console=False, agent_runner=runner)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.invalid",
        ) as client:
            response = await client.post(
                "/api/v1/agent/turns",
                headers={"Accept": "text/event-stream"},
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Review the best image-generation models.",
                        }
                    ]
                },
            )
    finally:
        await backend_client.aclose()

    events = _sse_events(response)
    assert [event["type"] for event in events] == [
        "turn.started",
        "tool.started",
        "tool.completed",
        "tool.started",
        "tool.completed",
        "assistant.delta",
        "turn.completed",
    ]
    assert events[-2]["content"] == "The evidence supports the final answer."
    assert len(backend_requests) == 4
    assert backend_requests[-1]["messages"][-1]["role"] == "user"
    assert (
        "Tool execution is complete"
        in backend_requests[-1]["messages"][-1]["content"]
    )
    assert all(
        [
            index
            for index, message in enumerate(payload["messages"])
            if message["role"] == "system"
        ]
        == [0]
        for payload in backend_requests
    )


@pytest.mark.asyncio
async def test_agent_completes_oversized_web_fetch_with_bounded_text(
    tmp_path: Path,
) -> None:
    paths = _active_paths(tmp_path)
    large_page = (
        b"<html><head><title>Large source</title></head>"
        b"<body><p>Useful benchmark evidence.</p>"
        + (b"x" * 4096)
        + b"</body></html>"
    )

    async def web_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Large source",
                            "url": "https://example.com/large",
                            "content": "Benchmark evidence",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "Content-Length": str(len(large_page)),
            },
            content=large_page,
        )

    class ScriptedModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, **kwargs: Any) -> Mapping[str, Any]:
            del kwargs
            self.calls += 1
            if self.calls == 1:
                return _tool_call(
                    "web_search", {"query": "best image models"}, "search"
                )
            if self.calls == 2:
                return _tool_call(
                    "web_fetch", {"url": "https://example.com/large"}, "fetch"
                )
            return _completion("The bounded evidence supports the answer.")

    web_client = httpx.AsyncClient(transport=httpx.MockTransport(web_handler))
    registry = create_builtin_registry(
        settings=BuiltinToolSettings(max_fetch_response_bytes=1024),
        client=web_client,
        resolver=lambda host, port: _public_resolution(host, port),
    )
    app = create_app(
        paths,
        enable_console=False,
        agent_runner=AgentRunner(registry, ScriptedModel()),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.invalid",
        ) as client:
            response = await client.post(
                "/api/v1/agent/turns",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Review the best image-generation models.",
                        }
                    ]
                },
            )
    finally:
        await web_client.aclose()

    events = _sse_events(response)
    assert all(event["type"] != "tool.failed" for event in events)
    fetch = next(
        event
        for event in events
        if event["type"] == "tool.completed" and event["name"] == "web_fetch"
    )
    assert fetch["result"]["title"] == "Large source"
    assert "Useful benchmark evidence." in fetch["result"]["text"]
    assert fetch["result"]["truncated"] is True
    assert fetch["result"]["source_host"] == "example.com"
    assert fetch["result"]["response_bytes_read"] == 1025
    assert fetch["result"]["response_bytes_declared"] == len(large_page)
    assert fetch["result"]["transport_truncated"] is True
    assert fetch["result"]["extraction_quality"] == "usable"
    assert events[-2]["content"] == "The bounded evidence supports the answer."
    assert events[-1]["type"] == "turn.completed"


async def _public_resolution(host: str, port: int) -> tuple[str, ...]:
    del host, port
    return ("93.184.216.34",)
