"""Asynchronous OpenAI-compatible benchmark execution and scoring.

The runner intentionally speaks only the small, stable subset of the OpenAI
chat-completions protocol needed by the benchmark schemas.  ``httpx`` clients
and transports are injectable, which keeps the runner usable with real model
servers and deterministic in unit tests.
"""

from __future__ import annotations

import asyncio
import json
import math
import platform
import re
import statistics
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from jsonschema import Draft202012Validator

from .errors import BenchmarkError
from .hashing import canonical_sha256
from .schema import BenchmarkCase, BenchmarkSuite, Expectation, GenerationConfig


JsonObject = dict[str, Any]

_PROTECTED_REQUEST_KEYS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "stream",
}
_REQUEST_CONTRACT_MODEL = "$MODEL_UNDER_TEST"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    """Convert common immutable/schema values to plain JSON-compatible data."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return value


def _json_object_snapshot(value: Any, *, purpose: str) -> JsonObject:
    """Validate and deep-copy one mapping through the canonical JSON domain."""

    normalized = _jsonable(value)
    if not isinstance(normalized, Mapping):
        raise BenchmarkError(f"{purpose} must be a JSON object")
    try:
        encoded = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        snapshot = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"{purpose} must contain valid JSON values: {exc}") from exc
    if not isinstance(snapshot, dict):  # defensive: normalized was a mapping
        raise BenchmarkError(f"{purpose} must be a JSON object")
    return snapshot


def _metadata_ref(value: str | Mapping[str, Any] | None) -> JsonObject | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"id": value}
    return _jsonable(value)


def _metadata_ref_with_id(value: Any, identifier: str | None) -> JsonObject | None:
    metadata = _metadata_ref(value)
    if metadata is None:
        return {"id": identifier} if identifier is not None else None
    if identifier is not None:
        metadata = dict(metadata)
        metadata["id"] = identifier
    return metadata


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """The result of applying one expectation to one model response."""

    kind: str
    passed: bool
    expected: Any = None
    actual: Any = None
    message: str | None = None

    def to_dict(self) -> JsonObject:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Normalized OpenAI token accounting for a response."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: JsonObject | None = None

    def to_dict(self) -> JsonObject:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class ServerTimings:
    """Normalized llama.cpp timing fields, preserving the source object.

    These values are reported by the server.  They are deliberately kept
    separate from client-observed request latency, and none is a TTFT
    measurement because the benchmark runner uses non-streaming requests.
    """

    prompt_tokens: int | None = None
    prompt_ms: float | None = None
    prompt_per_token_ms: float | None = None
    prompt_tokens_per_second: float | None = None
    predicted_tokens: int | None = None
    predicted_ms: float | None = None
    predicted_per_token_ms: float | None = None
    predicted_tokens_per_second: float | None = None
    raw: JsonObject | None = None

    def to_dict(self) -> JsonObject:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SampleResult:
    """One measured benchmark request and its complete outcome."""

    run_id: str
    case_id: str
    repetition: int
    started_at: str
    latency_ms: float
    request: JsonObject
    response: JsonObject | None
    output_text: str | None
    tool_names: tuple[str, ...]
    usage: TokenUsage
    scores: tuple[ScoreResult, ...]
    passed: bool | None
    error: JsonObject | None = None
    server_timings: ServerTimings | None = None
    client_completion_tokens_per_second: float | None = None

    def to_dict(self) -> JsonObject:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """An immutable in-memory representation of a completed benchmark run."""

    run: JsonObject
    summary: JsonObject
    samples: tuple[SampleResult, ...]
    telemetry: tuple[Any, ...] = ()

    @property
    def run_id(self) -> str:
        return str(self.run["run_id"])

    def write_bundle(self, path: str | Path) -> Path:
        # Local import avoids a module cycle: results accepts BenchmarkRun-like
        # objects but does not need benchmark to be imported at module import.
        from .results import write_run_bundle

        return write_run_bundle(path, self)


def _assistant_parts(response: str | Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Extract assistant content and requested tool names from a response."""

    if isinstance(response, str):
        return response, ()

    message: Any = response
    choices = response.get("choices")
    if isinstance(choices, Sequence) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message", first)
    if not isinstance(message, Mapping):
        return "", ()

    content = message.get("content")
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        # Multimodal-compatible servers may return structured content blocks.
        text = json.dumps(content, ensure_ascii=False, sort_keys=True)

    names: list[str] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)):
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if isinstance(function, Mapping) and function.get("name") is not None:
                names.append(str(function["name"]))
            elif call.get("name") is not None:
                names.append(str(call["name"]))
    legacy_call = message.get("function_call")
    if isinstance(legacy_call, Mapping) and legacy_call.get("name") is not None:
        names.append(str(legacy_call["name"]))
    return text, tuple(names)


def _expected_values(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return (value,)


def score_expectation(
    expectation: Expectation | Mapping[str, Any],
    response: str | Mapping[str, Any],
) -> ScoreResult:
    """Score one response using one schema expectation.

    String matchers default to case-insensitive behavior, as specified by
    :class:`~llm_lab.schema.Expectation`.  Sequence values for ``contains`` and
    ``tool_name`` mean that every listed value is required.
    """

    if not isinstance(expectation, Expectation):
        expectation = Expectation.model_validate(expectation)
    text, tool_names = _assistant_parts(response)
    expected = expectation.value

    def normalize(value: Any) -> str:
        rendered = value if isinstance(value, str) else str(value)
        return rendered if expectation.case_sensitive else rendered.casefold()

    if expectation.kind == "exact":
        passed = normalize(text) == normalize(expected)
        return ScoreResult("exact", passed, expected, text)

    if expectation.kind == "contains":
        values = _expected_values(expected)
        passed = all(normalize(value) in normalize(text) for value in values)
        return ScoreResult("contains", passed, expected, text)

    if expectation.kind == "regex":
        flags = 0 if expectation.case_sensitive else re.IGNORECASE
        try:
            passed = re.search(str(expected), text, flags=flags) is not None
            return ScoreResult("regex", passed, expected, text)
        except re.error as exc:
            return ScoreResult("regex", False, expected, text, f"invalid regex: {exc}")

    if expectation.kind == "json_schema":
        schema = expected.get("schema", expected) if isinstance(expected, Mapping) else expected
        if not isinstance(schema, Mapping):
            return ScoreResult(
                "json_schema", False, expected, text, "expectation value is not a JSON schema"
            )
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            return ScoreResult("json_schema", False, expected, text, f"invalid JSON: {exc}")
        try:
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(parsed)
        except Exception as exc:  # jsonschema exposes several validation subclasses
            return ScoreResult("json_schema", False, expected, parsed, str(exc))
        return ScoreResult("json_schema", True, expected, parsed)

    if expectation.kind == "tool_name":
        values = tuple(str(value) for value in _expected_values(expected))
        actual = tool_names
        if expectation.case_sensitive:
            passed = all(value in actual for value in values)
        else:
            folded = {name.casefold() for name in actual}
            passed = all(value.casefold() in folded for value in values)
        return ScoreResult("tool_name", passed, expected, list(actual))

    if expectation.kind == "nonempty":
        passed = bool(text.strip())
        return ScoreResult("nonempty", passed, expected, text)

    # Pydantic prevents this for normal callers, but retaining a defensive
    # branch makes the scorer safe when called with objects from older schemas.
    return ScoreResult(str(expectation.kind), False, expected, text, "unknown scorer")


def score_response(
    expectations: Sequence[Expectation],
    response: str | Mapping[str, Any],
) -> tuple[tuple[ScoreResult, ...], bool | None]:
    """Apply all expectations, returning details and an all-of verdict."""

    if not expectations:
        return (), None
    scores = tuple(score_expectation(expectation, response) for expectation in expectations)
    return scores, all(score.passed for score in scores)


score_output = score_expectation


def _usage_from_response(response: Mapping[str, Any]) -> TokenUsage:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return TokenUsage()

    def integer(name: str) -> int | None:
        value = usage.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return TokenUsage(
        prompt_tokens=integer("prompt_tokens"),
        completion_tokens=integer("completion_tokens"),
        total_tokens=integer("total_tokens"),
        raw=_jsonable(usage),
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rendered = float(value)
    return rendered if math.isfinite(rendered) and rendered >= 0 else None


def _finite_integer(value: Any) -> int | None:
    rendered = _finite_number(value)
    if rendered is None or not rendered.is_integer():
        return None
    return int(rendered)


def _server_timings_from_response(
    response: Mapping[str, Any],
) -> ServerTimings | None:
    timings = response.get("timings")
    if not isinstance(timings, Mapping):
        return None
    return ServerTimings(
        prompt_tokens=_finite_integer(timings.get("prompt_n")),
        prompt_ms=_finite_number(timings.get("prompt_ms")),
        prompt_per_token_ms=_finite_number(timings.get("prompt_per_token_ms")),
        prompt_tokens_per_second=_finite_number(timings.get("prompt_per_second")),
        predicted_tokens=_finite_integer(timings.get("predicted_n")),
        predicted_ms=_finite_number(timings.get("predicted_ms")),
        predicted_per_token_ms=_finite_number(
            timings.get("predicted_per_token_ms")
        ),
        predicted_tokens_per_second=_finite_number(
            timings.get("predicted_per_second")
        ),
        raw=_jsonable(timings),
    )


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _latency_summary(samples: Sequence[SampleResult]) -> JsonObject:
    values = [sample.latency_ms for sample in samples]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None,
                "p50": None, "p95": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
    }


def _metric_summary(values: Sequence[float | None]) -> JsonObject:
    known = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not known:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(known),
        "mean": statistics.fmean(known),
        "p50": _percentile(known, 0.50),
        "p95": _percentile(known, 0.95),
    }


def _performance_summary(samples: Sequence[SampleResult]) -> JsonObject:
    def server_value(name: str) -> list[float | None]:
        return [
            None
            if sample.server_timings is None
            else getattr(sample.server_timings, name)
            for sample in samples
        ]

    return {
        "client_completion_tokens_per_second": _metric_summary([
            sample.client_completion_tokens_per_second for sample in samples
        ]),
        "server_prompt_ms": _metric_summary(server_value("prompt_ms")),
        "server_prompt_per_token_ms": _metric_summary(
            server_value("prompt_per_token_ms")
        ),
        "server_prompt_tokens_per_second": _metric_summary(
            server_value("prompt_tokens_per_second")
        ),
        "server_predicted_ms": _metric_summary(server_value("predicted_ms")),
        "server_predicted_per_token_ms": _metric_summary(
            server_value("predicted_per_token_ms")
        ),
        "server_predicted_tokens_per_second": _metric_summary(
            server_value("predicted_tokens_per_second")
        ),
    }


def summarize_samples(
    samples: Sequence[SampleResult],
    *,
    skipped_cases: Sequence[Mapping[str, Any]] = (),
    warmup_request_count: int = 0,
    warmup_error_count: int = 0,
) -> JsonObject:
    """Build execution and per-case summaries without inventing missing scores."""

    errors = [sample for sample in samples if sample.error is not None]
    scoreable = [sample for sample in samples if sample.passed is not None]
    passed = [sample for sample in scoreable if sample.passed]

    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage: JsonObject = {}
    for field in usage_fields:
        values = [getattr(sample.usage, field) for sample in samples]
        known = [value for value in values if value is not None]
        usage[field] = sum(known) if known else None

    per_case: JsonObject = {}
    for case_id in sorted({sample.case_id for sample in samples}):
        case_samples = [sample for sample in samples if sample.case_id == case_id]
        case_scoreable = [sample for sample in case_samples if sample.passed is not None]
        case_passed = sum(sample.passed is True for sample in case_scoreable)
        per_case[case_id] = {
            "sample_count": len(case_samples),
            "error_count": sum(sample.error is not None for sample in case_samples),
            "scoreable_sample_count": len(case_scoreable),
            "passed_sample_count": case_passed,
            "failed_sample_count": len(case_scoreable) - case_passed,
            "pass_rate": case_passed / len(case_scoreable) if case_scoreable else None,
            "latency_ms": _latency_summary(case_samples),
            "performance": _performance_summary(case_samples),
        }

    # This is deliberately not a cross-task composite.  When a suite omitted
    # cases due to unavailable capabilities, no overall quality score is
    # emitted; consumers must use the explicit per-case measurements.
    complete_task_set = not skipped_cases
    pass_rate = len(passed) / len(scoreable) if scoreable and complete_task_set else None
    return {
        "sample_count": len(samples),
        "successful_request_count": len(samples) - len(errors),
        "error_count": len(errors),
        "scoreable_sample_count": len(scoreable),
        "passed_sample_count": len(passed),
        "failed_sample_count": len(scoreable) - len(passed),
        "pass_rate": pass_rate,
        "composite_score": None,
        "latency_ms": _latency_summary(samples),
        "performance": _performance_summary(samples),
        "usage": usage,
        "warmup_request_count": warmup_request_count,
        "warmup_error_count": warmup_error_count,
        "skipped_cases": [_jsonable(item) for item in skipped_cases],
        "per_case": per_case,
    }


def _runtime_metadata() -> JsonObject:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "httpx": _package_version("httpx"),
        "jsonschema": _package_version("jsonschema"),
        "llm_lab": _package_version("llm-lab"),
    }


def _hardware_metadata(telemetry: Sequence[Any]) -> JsonObject:
    metadata: JsonObject = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_pointer_bits": 64 if sys.maxsize > 2**32 else 32,
    }
    try:
        import psutil

        metadata["cpu_logical_count"] = psutil.cpu_count(logical=True)
        metadata["memory_bytes"] = psutil.virtual_memory().total
    except Exception:
        metadata["cpu_logical_count"] = None
        metadata["memory_bytes"] = None

    seen: set[tuple[Any, Any]] = set()
    gpus: list[JsonObject] = []
    for value in telemetry:
        item = value.to_dict() if hasattr(value, "to_dict") else _jsonable(value)
        if not isinstance(item, Mapping) or not item.get("available"):
            continue
        key = (item.get("index"), item.get("uuid"))
        if key in seen:
            continue
        seen.add(key)
        gpus.append({
            "index": item.get("index"),
            "uuid": item.get("uuid"),
            "name": item.get("name"),
            "memory_total_mib": item.get("memory_total_mib"),
        })
    metadata["gpus"] = gpus
    return metadata


class BenchmarkRunner:
    """Run a :class:`BenchmarkSuite` against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str | None = None,
        *,
        model_id: str | None = None,
        served_model: str | None = None,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float | httpx.Timeout = 120.0,
        endpoint: str = "/v1/chat/completions",
        extra_body: Mapping[str, Any] | None = None,
        telemetry_interval_seconds: float = 1.0,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("provide either client or transport, not both")
        if model is not None and served_model is not None and model != served_model:
            raise ValueError("model and served_model disagree")
        self.base_url = base_url.rstrip("/")
        self.model = served_model or model or model_id
        self.endpoint = endpoint
        self.extra_body = dict(extra_body or {})
        self.telemetry_interval_seconds = telemetry_interval_seconds
        self._owns_client = client is None
        if client is None:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=timeout,
                transport=transport,
            )
        else:
            self._client = client

    async def __aenter__(self) -> "BenchmarkRunner":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    @staticmethod
    def _generation(case: BenchmarkCase, suite: BenchmarkSuite) -> GenerationConfig:
        return case.generation or suite.defaults

    def _request_body(
        self,
        case: BenchmarkCase,
        suite: BenchmarkSuite,
        served_model: str,
        extra_body: Mapping[str, Any] | None,
        *,
        runner_extra_body: Mapping[str, Any] | None = None,
    ) -> JsonObject:
        generation = self._generation(case, suite)
        body: JsonObject = {
            "model": served_model,
            "messages": [message.model_dump(mode="json") for message in case.messages],
            **generation.model_dump(mode="json"),
        }
        for source, extensions in (
            ("case request_overrides", case.request_overrides),
            (
                "runner extra_body",
                self.extra_body
                if runner_extra_body is None
                else runner_extra_body,
            ),
            ("run extra_body", extra_body or {}),
        ):
            conflicts = sorted(_PROTECTED_REQUEST_KEYS.intersection(extensions))
            if conflicts:
                raise BenchmarkError(
                    f"{source} cannot override protected request fields: {conflicts}"
                )
            body.update(_jsonable(extensions))
        return body

    def _request_contract(
        self,
        suite: BenchmarkSuite,
        requests_by_case: Mapping[str, Mapping[str, Any]],
    ) -> JsonObject:
        """Describe and hash the exact model-independent request workload."""

        definition: JsonObject = {
            "method": "POST",
            "endpoint": self.endpoint,
            "warmup_repetitions": suite.warmup_repetitions,
            "repetitions": suite.repetitions,
            "requests": [
                {
                    "case_id": case.id,
                    "body": {
                        **_jsonable(requests_by_case[case.id]),
                        "model": _REQUEST_CONTRACT_MODEL,
                    },
                }
                for case in suite.cases
            ],
        }
        return {
            "schema_version": 1,
            "sha256": canonical_sha256(definition),
            "definition": definition,
        }

    async def _request(
        self,
        *,
        run_id: str,
        case: BenchmarkCase,
        repetition: int,
        request: JsonObject,
    ) -> SampleResult:
        started_at = _utc_now()
        started = perf_counter()
        parsed: JsonObject | None = None
        error: JsonObject | None = None
        output_text: str | None = None
        tool_names: tuple[str, ...] = ()
        usage = TokenUsage()
        server_timings: ServerTimings | None = None
        scores: tuple[ScoreResult, ...] = ()
        passed: bool | None = None
        try:
            response = await self._client.post(self.endpoint, json=request)
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, Mapping):
                raise ValueError("response JSON is not an object")
            expected_model = request.get("model")
            reported_model = value.get("model")
            if reported_model != expected_model:
                raise ValueError(
                    "response model identity mismatch: "
                    f"expected {expected_model!r}, got {reported_model!r}"
                )
            choices = value.get("choices")
            if (
                not isinstance(choices, Sequence)
                or isinstance(choices, (str, bytes, bytearray))
                or not choices
                or not isinstance(choices[0], Mapping)
                or not isinstance(choices[0].get("message"), Mapping)
            ):
                raise ValueError("response has no valid choices[0].message")
            parsed = _jsonable(value)
            output_text, tool_names = _assistant_parts(parsed)
            usage = _usage_from_response(parsed)
            server_timings = _server_timings_from_response(parsed)
            scores, passed = score_response(case.expectations, parsed)
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:4096]
            error = {
                "type": "http_status",
                "message": str(exc),
                "status_code": exc.response.status_code,
                "response_body": body,
            }
            passed = False if case.expectations else None
        except httpx.RequestError as exc:
            error = {"type": "request", "message": str(exc)}
            passed = False if case.expectations else None
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            error = {"type": "invalid_response", "message": str(exc)}
            passed = False if case.expectations else None
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # preserve suite progress on server/protocol surprises
            error = {"type": type(exc).__name__, "message": str(exc)}
            passed = False if case.expectations else None
        latency_ms = (perf_counter() - started) * 1000
        client_completion_tokens_per_second: float | None = None
        if (
            usage.completion_tokens is not None
            and usage.completion_tokens >= 0
            and latency_ms > 0
        ):
            # This is end-to-end throughput observed by the non-streaming
            # client, not server decode throughput and not TTFT.
            client_completion_tokens_per_second = (
                usage.completion_tokens / (latency_ms / 1000)
            )
        return SampleResult(
            run_id=run_id,
            case_id=case.id,
            repetition=repetition,
            started_at=started_at,
            latency_ms=latency_ms,
            request=request,
            response=parsed,
            output_text=output_text,
            tool_names=tool_names,
            usage=usage,
            scores=scores,
            passed=passed,
            error=error,
            server_timings=server_timings,
            client_completion_tokens_per_second=(
                client_completion_tokens_per_second
            ),
        )

    async def run(
        self,
        suite: BenchmarkSuite,
        *,
        model: str | Mapping[str, Any] | None = None,
        model_id: str | None = None,
        artifact: str | Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        deployment: str | Mapping[str, Any] | None = None,
        deployment_id: str | None = None,
        runtime: Mapping[str, Any] | None = None,
        hardware: Mapping[str, Any] | None = None,
        sampling: Mapping[str, Any] | None = None,
        available_capabilities: Sequence[str] | None = None,
        run_id: str | None = None,
        telemetry_sampler: Any | None = None,
        collect_telemetry: bool = True,
        extra_body: Mapping[str, Any] | None = None,
        bundle_dir: str | Path | None = None,
    ) -> BenchmarkRun:
        """Execute warmups then measured repetitions for every eligible case.

        API failures are represented in :class:`SampleResult` and do not abort
        subsequent cases.  Warmup requests count toward call volume but are not
        included in score or latency summaries.
        """

        if not isinstance(suite, BenchmarkSuite):
            suite = BenchmarkSuite.model_validate(suite)
        resolved_run_id = run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
        model_meta = _metadata_ref_with_id(model if model is not None else self.model, model_id)
        artifact_meta = _metadata_ref_with_id(artifact, artifact_id)
        deployment_meta = _metadata_ref_with_id(deployment, deployment_id)

        served_model: str | None = self.model or model_id
        if served_model is None and isinstance(model, str):
            served_model = model
        if served_model is None and isinstance(deployment_meta, Mapping):
            alias = deployment_meta.get("public_alias") or deployment_meta.get("served_model")
            served_model = str(alias) if alias is not None else None
        if served_model is None and isinstance(model_meta, Mapping):
            candidate = model_meta.get("id")
            served_model = str(candidate) if candidate is not None else None
        if served_model is None:
            raise BenchmarkError("a served model name is required")
        runner_extra_body = _json_object_snapshot(
            self.extra_body,
            purpose="runner extra_body",
        )
        run_extra_body = _json_object_snapshot(
            extra_body or {},
            purpose="run extra_body",
        )
        requests_by_case = {
            case.id: self._request_body(
                case,
                suite,
                served_model,
                run_extra_body,
                runner_extra_body=runner_extra_body,
            )
            for case in suite.cases
        }
        request_contract = self._request_contract(suite, requests_by_case)

        capabilities: set[str] | None
        if available_capabilities is not None:
            capabilities = set(available_capabilities)
        elif isinstance(model_meta, Mapping) and "capabilities" in model_meta:
            capabilities = set(model_meta.get("capabilities") or ())
        else:
            capabilities = None

        skipped_cases: list[JsonObject] = []
        eligible: list[BenchmarkCase] = []
        for case in suite.cases:
            missing = sorted(set(case.required_capabilities) - capabilities) if capabilities is not None else []
            if missing:
                skipped_cases.append({"case_id": case.id, "reason": "missing_capabilities",
                                      "missing_capabilities": missing})
            else:
                eligible.append(case)

        sampler = telemetry_sampler
        if collect_telemetry and sampler is None:
            from .telemetry import NvidiaTelemetrySampler

            sampler = NvidiaTelemetrySampler(interval_seconds=self.telemetry_interval_seconds)

        started_at = _utc_now()
        telemetry_start_index = len(getattr(sampler, "samples", ())) if sampler is not None else 0
        if sampler is not None and collect_telemetry:
            await sampler.start()

        measured: list[SampleResult] = []
        warmup_count = 0
        warmup_errors = 0
        try:
            for repetition in range(suite.warmup_repetitions):
                for case in eligible:
                    request = _json_object_snapshot(
                        requests_by_case[case.id],
                        purpose="effective benchmark request",
                    )
                    warmup = await self._request(
                        run_id=resolved_run_id,
                        case=case,
                        repetition=-(repetition + 1),
                        request=request,
                    )
                    warmup_count += 1
                    warmup_errors += warmup.error is not None

            for repetition in range(suite.repetitions):
                for case in eligible:
                    request = _json_object_snapshot(
                        requests_by_case[case.id],
                        purpose="effective benchmark request",
                    )
                    measured.append(await self._request(
                        run_id=resolved_run_id,
                        case=case,
                        repetition=repetition,
                        request=request,
                    ))
        finally:
            if sampler is not None and collect_telemetry:
                await sampler.stop()

        telemetry = (
            tuple(getattr(sampler, "samples", ())[telemetry_start_index:])
            if sampler is not None else ()
        )
        finished_at = _utc_now()
        summary = summarize_samples(
            measured,
            skipped_cases=skipped_cases,
            warmup_request_count=warmup_count,
            warmup_error_count=warmup_errors,
        )
        status = (
            "completed_with_errors"
            if summary["error_count"] or summary["warmup_error_count"]
            else "completed"
        )
        if not measured:
            status = "no_eligible_cases"

        runtime_meta = _runtime_metadata()
        if runtime:
            runtime_meta.update(_jsonable(runtime))
        hardware_meta = _hardware_metadata(telemetry)
        if hardware:
            hardware_meta.update(_jsonable(hardware))
        sampling_meta: JsonObject = {
            "warmup_repetitions": suite.warmup_repetitions,
            "repetitions": suite.repetitions,
            "defaults": suite.defaults.model_dump(mode="json"),
        }
        if sampling:
            sampling_meta.update(_jsonable(sampling))
        if deployment_meta is None:
            deployment_meta = {"id": None, "base_url": self.base_url, "served_model": served_model}

        suite_definition = suite.model_dump(mode="json")
        run_document: JsonObject = {
            "schema_version": 1,
            "run_id": resolved_run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": status,
            "model": model_meta or {"id": served_model},
            "artifact": artifact_meta,
            "deployment": deployment_meta,
            "suite": {
                "id": suite.id,
                "version": suite.version,
                "kind": suite.kind,
                "description": suite.description,
                "sha256": canonical_sha256(suite_definition),
                "definition": suite_definition,
            },
            "request_contract": request_contract,
            "runtime": runtime_meta,
            "hardware": hardware_meta,
            "sampling": sampling_meta,
        }
        completed = BenchmarkRun(run_document, summary, tuple(measured), telemetry)
        if bundle_dir is not None:
            completed.write_bundle(bundle_dir)
        return completed


async def run_benchmark(
    suite: BenchmarkSuite,
    *,
    base_url: str = "http://127.0.0.1:8080",
    served_model: str,
    transport: httpx.AsyncBaseTransport | None = None,
    client: httpx.AsyncClient | None = None,
    **kwargs: Any,
) -> BenchmarkRun:
    """Convenience wrapper for one benchmark run."""

    runner = BenchmarkRunner(base_url, served_model, transport=transport, client=client)
    try:
        return await runner.run(suite, **kwargs)
    finally:
        await runner.aclose()
