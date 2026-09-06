from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from llm_lab.attestation import verify_gateway_attestation
from llm_lab.benchmark import BenchmarkRunner
from llm_lab.catalog import Catalog
from llm_lab.errors import IntegrityError
from llm_lab.gateway import create_app
from llm_lab.mock_backend import create_mock_app
from llm_lab.paths import LabPaths
from llm_lab.runtime import (
    LaunchRecord,
    RuntimeManager,
    RuntimeState,
    build_backend_command,
    write_active_state,
)
from llm_lab.schema import BackendKind, DeploymentSpec


def _paths(tmp_path: Path) -> LabPaths:
    repo = tmp_path / "repo"
    repo.mkdir()
    paths = LabPaths.discover(repo_root=repo, data_root=tmp_path / "data")
    paths.initialize()
    return paths


def _publish_external_state(
    paths: LabPaths,
    *,
    phase: str = "ready",
    base_url: str = "http://backend.invalid/v1",
) -> RuntimeState:
    deployment = DeploymentSpec(
        id="test-active",
        artifact_id="test-artifact",
        public_alias="local-test",
        backend=BackendKind.EXTERNAL,
        external_base_url=base_url,
        startup_timeout_seconds=1,
    )
    plan = build_backend_command(deployment, paths=paths)
    state = RuntimeState(
        schema_version=1,
        phase=phase,  # type: ignore[arg-type]
        deployment=deployment,
        artifact_path=None,
        plan=plan,
        launch=LaunchRecord(
            kind="external",
            command=(),
            started_at="2026-09-03T00:00:00+00:00",
        ),
        activated_at="2026-09-03T00:00:00+00:00",
    )
    write_active_state(paths, state)
    return state


@pytest.mark.asyncio
async def test_inactive_gateway_returns_clear_503(tmp_path: Path) -> None:
    app = create_app(_paths(tmp_path))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.invalid",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "local-test", "messages": []},
        )
        health = await client.get("/health")

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "No active model deployment"
    assert health.status_code == 200
    assert health.json()["active"] is False


@pytest.mark.asyncio
async def test_gateway_health_proves_identity_with_a_fresh_challenge(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _publish_external_state(paths)
    origin = "http://gateway.invalid"
    app = create_app(
        paths,
        advertised_origin=origin,
        runtime_manager=RuntimeManager(paths, health_checker=lambda *_: None),
    )
    challenge = "a" * 64
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=origin
    ) as client:
        response = await client.get("/health", params={"challenge": challenge})

    assert response.status_code == 200
    document = response.json()
    verified = verify_gateway_attestation(
        paths,
        document["attestation"],
        challenge=challenge,
        origin=origin,
        deployment="test-active",
        model="local-test",
    )
    assert verified["payload"]["ready"] is True

    tampered = dict(document["attestation"])
    tampered["hmac_sha256"] = "0" * 64
    with pytest.raises(IntegrityError, match="signature is invalid"):
        verify_gateway_attestation(
            paths,
            tampered,
            challenge=challenge,
            origin=origin,
            deployment="test-active",
            model="local-test",
        )


@pytest.mark.asyncio
async def test_gateway_bearer_auth_is_optional_and_enforced(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _publish_external_state(paths)
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_mock_app("local-test")),
        base_url="http://backend.invalid",
    )
    app = create_app(paths, api_key="secret-key", client=upstream)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.invalid",
        ) as client:
            missing = await client.get("/v1/models")
            wrong = await client.get(
                "/v1/models", headers={"Authorization": "Bearer wrong"}
            )
            accepted = await client.get(
                "/v1/models", headers={"Authorization": "Bearer secret-key"}
            )
            unprotected_health = await client.get("/health")
    finally:
        await upstream.aclose()

    assert missing.status_code == wrong.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert accepted.status_code == 200
    assert accepted.json()["data"][0]["id"] == "local-test"
    assert unprotected_health.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "payload", "expected"),
    [
        (
            "/v1/chat/completions",
            {"model": "local-test", "messages": [{"role": "user", "content": "hi"}]},
            lambda body: body["choices"][0]["message"]["content"],
        ),
        (
            "/v1/completions",
            {"model": "local-test", "prompt": "hi"},
            lambda body: body["choices"][0]["text"],
        ),
        (
            "/v1/responses",
            {"model": "local-test", "input": "hi"},
            lambda body: body["output_text"],
        ),
    ],
)
async def test_gateway_proxies_openai_json_shapes(
    tmp_path: Path,
    route: str,
    payload: dict,
    expected,
) -> None:
    paths = _paths(tmp_path)
    # The /v1 suffix exercises the no-double-/v1 URL join path.
    _publish_external_state(paths, base_url="http://backend.invalid/v1")
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_mock_app("local-test")),
        base_url="http://backend.invalid",
    )
    app = create_app(paths, client=upstream)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.invalid",
        ) as client:
            first = await client.post(route, json=payload)
            second = await client.post(route, json=payload)
    finally:
        await upstream.aclose()

    assert first.status_code == 200
    assert expected(first.json()) == "mock response: hi"
    assert first.content == second.content
    assert first.headers["x-llm-lab-mock"] == "true"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route",
    ("/v1/chat/completions", "/v1/completions", "/v1/responses"),
)
@pytest.mark.parametrize("requested_model", (None, "anything-else"))
async def test_gateway_rejects_missing_or_mismatched_model_alias(
    tmp_path: Path,
    route: str,
    requested_model: str | None,
) -> None:
    paths = _paths(tmp_path)
    _publish_external_state(paths)
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_mock_app("local-test")),
        base_url="http://backend.invalid",
    )
    app = create_app(paths, client=upstream)
    payload: dict[str, object] = {"messages": []}
    if requested_model is not None:
        payload["model"] = requested_model
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.invalid",
        ) as client:
            response = await client.post(route, json=payload)
    finally:
        await upstream.aclose()

    assert response.status_code == 400
    assert "active alias" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_gateway_streams_sse_without_reformatting(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _publish_external_state(paths)
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_mock_app("local-test")),
        base_url="http://backend.invalid",
    )
    app = create_app(paths, client=upstream)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.invalid",
        ) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "local-test",
                    "messages": [{"role": "user", "content": "stream me"}],
                    "stream": True,
                },
            ) as response:
                chunks = [chunk async for chunk in response.aiter_bytes()]
    finally:
        await upstream.aclose()

    body = b"".join(chunks)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert b'"object":"chat.completion.chunk"' in body
    assert b'"content":"mock response: stream me"' in body
    assert body.endswith(b"data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_mock_satisfies_structured_and_forced_tool_smoke_cases(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _publish_external_state(paths)
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_mock_app("local-test")),
        base_url="http://backend.invalid",
    )
    app = create_app(paths, client=upstream)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.invalid",
        ) as client:
            planet = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "local-test",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Name the planet humans live on.",
                        }
                    ],
                },
            )
            structured = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "local-test",
                    "messages": [{"role": "user", "content": "return JSON"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "smoke_result",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string", "const": "ok"},
                                    "count": {"type": "integer", "const": 2},
                                },
                                "required": ["status", "count"],
                            },
                        },
                    },
                },
            )
            tool = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "local-test",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the weather in Seattle? Use the weather tool.",
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            },
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "get_weather"},
                    },
                },
            )
    finally:
        await upstream.aclose()

    assert planet.json()["choices"][0]["message"]["content"] == "Earth"
    structured_text = structured.json()["choices"][0]["message"]["content"]
    assert json.loads(structured_text) == {"status": "ok", "count": 2}
    choice = tool.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    function = choice["message"]["tool_calls"][0]["function"]
    assert function["name"] == "get_weather"
    assert json.loads(function["arguments"]) == {"city": "Seattle"}


@pytest.mark.asyncio
async def test_checked_in_smoke_suite_passes_through_gateway(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _publish_external_state(paths)
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_mock_app("local-test")),
        base_url="http://backend.invalid",
    )
    gateway = create_app(paths, client=upstream)
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway),
        base_url="http://gateway.invalid",
    )
    runner = BenchmarkRunner(model="local-test", client=gateway_client)
    try:
        suite = Catalog.load(Path("catalog")).get_suite("smoke")
        result = await runner.run(
            suite,
            available_capabilities=("tools",),
            collect_telemetry=False,
            run_id="mock-gateway-smoke",
        )
    finally:
        await gateway_client.aclose()
        await upstream.aclose()

    assert result.summary["sample_count"] == 3
    assert result.summary["error_count"] == 0
    assert all(sample.passed for sample in result.samples)


@pytest.mark.asyncio
async def test_gateway_rejects_non_ready_and_corrupt_state(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _publish_external_state(paths, phase="starting")
    app = create_app(paths)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.invalid",
    ) as client:
        starting = await client.get("/v1/models")
        paths.active_state_path.write_text("{not-json")
        corrupt = await client.get("/v1/models")

    assert starting.status_code == 503
    assert "phase=starting" in starting.json()["error"]["message"]
    assert corrupt.status_code == 503
    assert "active state is invalid" in corrupt.json()["error"]["message"]
