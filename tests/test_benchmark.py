from __future__ import annotations

import json

import httpx
import pytest

from llm_lab.benchmark import BenchmarkRunner, score_expectation
from llm_lab.schema import (
    BenchmarkCase,
    BenchmarkSuite,
    ChatMessage,
    Expectation,
)


@pytest.mark.parametrize(
    ("expectation", "response", "passed"),
    [
        (Expectation(kind="exact", value="READY"), "ready", True),
        (Expectation(kind="exact", value="READY", case_sensitive=True), "ready", False),
        (Expectation(kind="contains", value="brown fox"), "The Brown Fox jumps", True),
        (Expectation(kind="contains", value="missing"), "present", False),
        (Expectation(kind="regex", value=r"id-\d{3}"), "ID-042", True),
        (Expectation(kind="regex", value=r"^yes$"), "no", False),
        (
            Expectation(
                kind="json_schema",
                value={
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
            ),
            '{"count":2}',
            True,
        ),
        (
            Expectation(kind="json_schema", value={"type": "array"}),
            '{"count":2}',
            False,
        ),
        (Expectation(kind="nonempty"), " useful ", True),
        (Expectation(kind="nonempty"), " \n\t ", False),
    ],
)
def test_text_scorers_pass_and_fail(expectation, response, passed):
    result = score_expectation(expectation, response)

    assert result.passed is passed


def test_tool_name_scorer_passes_and_fails():
    response = {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}],
            }
        }]
    }

    assert score_expectation(Expectation(kind="tool_name", value="GET_WEATHER"), response).passed
    assert not score_expectation(Expectation(kind="tool_name", value="send_email"), response).passed


def _suite(*, warmups: int = 1, repetitions: int = 2) -> BenchmarkSuite:
    return BenchmarkSuite(
        id="runner-test",
        version="1",
        description="runner behavior",
        kind="smoke",
        warmup_repetitions=warmups,
        repetitions=repetitions,
        cases=(
            BenchmarkCase(
                id="first",
                messages=(ChatMessage(role="user", content="first"),),
                request_overrides={"response_format": {"type": "text"}},
                expectations=(Expectation(kind="exact", value="ok"),),
            ),
            BenchmarkCase(
                id="second",
                messages=(ChatMessage(role="user", content="second"),),
                expectations=(Expectation(kind="nonempty"),),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_runner_executes_warmups_and_repetitions_with_transport():
    requests: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    runner = BenchmarkRunner(
        "https://model.invalid",
        "served-model",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await runner.run(
            _suite(),
            model={"id": "catalog-model", "capabilities": ["text"]},
            artifact="artifact-q4",
            deployment="local-llama-cpp",
            runtime={"backend": "llama.cpp"},
            hardware={"gpu": "RTX 4090"},
            sampling={"profile": "deterministic"},
            collect_telemetry=False,
            run_id="run-repeat",
        )
    finally:
        await runner.aclose()

    assert len(requests) == 6  # 2 cases * (1 warmup + 2 measured)
    assert len(result.samples) == 4
    assert {(sample.case_id, sample.repetition) for sample in result.samples} == {
        ("first", 0), ("first", 1), ("second", 0), ("second", 1)
    }
    assert all(sample.passed for sample in result.samples)
    assert result.summary["warmup_request_count"] == 2
    assert result.summary["sample_count"] == 4
    assert result.summary["usage"]["total_tokens"] == 16
    assert result.run["model"]["id"] == "catalog-model"
    assert result.run["artifact"]["id"] == "artifact-q4"
    assert result.run["deployment"]["id"] == "local-llama-cpp"
    assert len(result.run["suite"]["sha256"]) == 64
    assert result.run["suite"]["definition"]["cases"][0]["id"] == "first"
    assert len(result.run["request_contract"]["sha256"]) == 64
    assert result.run["request_contract"]["definition"]["requests"][0][
        "body"
    ]["model"] == "$MODEL_UNDER_TEST"
    assert requests[0]["response_format"] == {"type": "text"}


@pytest.mark.asyncio
async def test_effective_request_contract_canonically_includes_extra_body():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    transport = httpx.MockTransport(handler)
    suite = _suite(warmups=0, repetitions=1)
    runner_extension = BenchmarkRunner(
        "https://model.invalid",
        "served-model",
        transport=transport,
        extra_body={"metadata": {"b": 2, "a": 1}, "min_p": 0.1},
    )
    run_extension = BenchmarkRunner(
        "https://model.invalid", "served-model", transport=transport
    )
    different_extension = BenchmarkRunner(
        "https://model.invalid", "served-model", transport=transport
    )
    try:
        first = await runner_extension.run(suite, collect_telemetry=False)
        second = await run_extension.run(
            suite,
            collect_telemetry=False,
            extra_body={"min_p": 0.1, "metadata": {"a": 1, "b": 2}},
        )
        different = await different_extension.run(
            suite,
            collect_telemetry=False,
            extra_body={"min_p": 0.2, "metadata": {"a": 1, "b": 2}},
        )
    finally:
        await runner_extension.aclose()
        await run_extension.aclose()
        await different_extension.aclose()

    first_contract = first.run["request_contract"]
    assert first_contract["sha256"] == second.run["request_contract"]["sha256"]
    assert first_contract["sha256"] != different.run["request_contract"]["sha256"]
    assert first_contract["definition"]["requests"][0]["body"]["min_p"] == 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize("extension_source", ("runner", "run"))
async def test_extra_body_nested_mutation_cannot_change_snapshotted_workload(
    extension_source: str,
):
    extension = {"stop": ["initial"]}
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            extension["stop"].append("mutated-after-warmup")
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    runner = BenchmarkRunner(
        "https://model.invalid",
        "served-model",
        transport=httpx.MockTransport(handler),
        extra_body=extension if extension_source == "runner" else None,
    )
    try:
        result = await runner.run(
            _suite(warmups=1, repetitions=1),
            collect_telemetry=False,
            extra_body=extension if extension_source == "run" else None,
        )
    finally:
        await runner.aclose()

    assert extension["stop"] == ["initial", "mutated-after-warmup"]
    assert all(request["stop"] == ["initial"] for request in requests)
    contract_requests = result.run["request_contract"]["definition"]["requests"]
    assert all(item["body"]["stop"] == ["initial"] for item in contract_requests)


@pytest.mark.parametrize(
    "reserved",
    ("model", "messages", "temperature", "top_p", "max_tokens", "seed", "stream"),
)
def test_case_overrides_cannot_replace_request_identity_or_sampling(reserved: str):
    with pytest.raises(ValueError, match="cannot replace"):
        BenchmarkCase(
            id="unsafe",
            messages=(ChatMessage(role="user", content="original"),),
            request_overrides={reserved: "replacement"},
        )


@pytest.mark.asyncio
async def test_runner_rejects_response_for_a_different_model_identity():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "another-model",
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    runner = BenchmarkRunner(
        "https://model.invalid",
        "expected-model",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await runner.run(_suite(warmups=0, repetitions=1), collect_telemetry=False)
    finally:
        await runner.aclose()

    assert all(sample.error["type"] == "invalid_response" for sample in result.samples)
    assert all("model identity mismatch" in sample.error["message"] for sample in result.samples)


@pytest.mark.asyncio
async def test_runner_records_server_timings_and_client_observed_throughput(
    monkeypatch,
):
    ticks = iter((10.0, 12.0))
    monkeypatch.setattr("llm_lab.benchmark.perf_counter", lambda: next(ticks))

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "model",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
                "timings": {
                    "prompt_n": 10,
                    "prompt_ms": 100.0,
                    "prompt_per_token_ms": 10.0,
                    "prompt_per_second": 100.0,
                    "predicted_n": 4,
                    "predicted_ms": 500.0,
                    "predicted_per_token_ms": 125.0,
                    "predicted_per_second": 8.0,
                    "cache_n": 2,
                },
            },
        )

    suite = BenchmarkSuite(
        id="performance",
        version="1",
        description="performance fields",
        kind="performance",
        warmup_repetitions=0,
        repetitions=1,
        cases=(
            BenchmarkCase(
                id="generation",
                messages=(ChatMessage(role="user", content="hello"),),
                expectations=(Expectation(kind="nonempty"),),
            ),
        ),
    )
    runner = BenchmarkRunner(
        "https://model.invalid", "model", transport=httpx.MockTransport(handler)
    )
    try:
        result = await runner.run(suite, collect_telemetry=False)
    finally:
        await runner.aclose()

    sample = result.samples[0]
    assert sample.latency_ms == 2000.0
    assert sample.client_completion_tokens_per_second == 2.0
    assert sample.server_timings is not None
    assert sample.server_timings.prompt_tokens == 10
    assert sample.server_timings.prompt_ms == 100.0
    assert sample.server_timings.prompt_tokens_per_second == 100.0
    assert sample.server_timings.predicted_tokens == 4
    assert sample.server_timings.predicted_ms == 500.0
    assert sample.server_timings.predicted_tokens_per_second == 8.0
    assert sample.server_timings.raw["cache_n"] == 2

    performance = result.summary["performance"]
    assert performance["client_completion_tokens_per_second"] == {
        "count": 1,
        "mean": 2.0,
        "p50": 2.0,
        "p95": 2.0,
    }
    assert performance["server_predicted_tokens_per_second"] == {
        "count": 1,
        "mean": 8.0,
        "p50": 8.0,
        "p95": 8.0,
    }
    assert "ttft" not in json.dumps(result.summary).casefold()


@pytest.mark.asyncio
async def test_runner_records_api_errors_and_continues():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if prompt == "first":
            return httpx.Response(503, text="server is loading")
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    async with httpx.AsyncClient(
        base_url="https://model.invalid",
        transport=httpx.MockTransport(handler),
    ) as client:
        runner = BenchmarkRunner(model="served-model", client=client)
        result = await runner.run(
            _suite(warmups=0, repetitions=1),
            collect_telemetry=False,
            run_id="run-error",
        )

    assert len(result.samples) == 2
    failed, succeeded = result.samples
    assert failed.error["type"] == "http_status"
    assert failed.error["status_code"] == 503
    assert failed.passed is False
    assert succeeded.error is None
    assert succeeded.passed is True
    assert succeeded.server_timings is None
    assert succeeded.client_completion_tokens_per_second is None
    assert result.summary["error_count"] == 1
    assert result.summary["performance"][
        "client_completion_tokens_per_second"
    ] == {"count": 0, "mean": None, "p50": None, "p95": None}
    assert result.run["status"] == "completed_with_errors"


@pytest.mark.asyncio
async def test_any_warmup_error_marks_run_completed_with_errors():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        if call_count == 1:
            return httpx.Response(503, text="warmup failed")
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    runner = BenchmarkRunner(
        "https://model.invalid",
        "served-model",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await runner.run(
            _suite(warmups=1, repetitions=1),
            collect_telemetry=False,
        )
    finally:
        await runner.aclose()

    assert result.summary["warmup_request_count"] == 2
    assert result.summary["warmup_error_count"] == 1
    assert result.summary["error_count"] == 0
    assert result.run["status"] == "completed_with_errors"


@pytest.mark.asyncio
async def test_missing_capability_is_skipped_without_composite_score():
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "model": "model",
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    suite = BenchmarkSuite(
        id="capabilities",
        version="1",
        description="skip unsupported modalities",
        kind="capability",
        cases=(
            BenchmarkCase(
                id="text",
                messages=(ChatMessage(role="user", content="hello"),),
                required_capabilities=("text",),
                expectations=(Expectation(kind="nonempty"),),
            ),
            BenchmarkCase(
                id="vision",
                messages=(ChatMessage(role="user", content="inspect image"),),
                required_capabilities=("vision",),
                expectations=(Expectation(kind="nonempty"),),
            ),
        ),
    )
    runner = BenchmarkRunner(
        "https://model.invalid", "model", transport=httpx.MockTransport(handler)
    )
    try:
        result = await runner.run(
            suite,
            available_capabilities=("text",),
            collect_telemetry=False,
        )
    finally:
        await runner.aclose()

    assert calls == 1
    assert [sample.case_id for sample in result.samples] == ["text"]
    assert result.summary["pass_rate"] is None
    assert result.summary["composite_score"] is None
    assert result.summary["skipped_cases"][0]["case_id"] == "vision"
