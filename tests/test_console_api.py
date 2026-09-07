from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request

from llm_lab.console_api import ConsoleController, install_console
from llm_lab.control_service import (
    ActivationResult,
    BenchmarkLaunchInput,
    ControlServiceError,
)
from llm_lab.operations import EventBroker, OperationStatus, OperationStore
from llm_lab.paths import LabPaths
from llm_lab.registry import Registry
from llm_lab.telemetry import unavailable_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUSE_DEPLOYMENT = "muse-glimmer-30b-4090-8k"


def _paths(tmp_path: Path) -> LabPaths:
    """Give every API test an isolated catalog and durable state directory."""

    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(PROJECT_ROOT / "catalog", repo / "catalog")
    paths = LabPaths.discover(repo_root=repo, data_root=tmp_path / "data")
    paths.initialize()
    return paths


class FakeControlService:
    def __init__(
        self,
        *,
        error: BaseException | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.error = error
        self.release = release
        self.entered = threading.Event()
        self.calls: list[tuple[str, str, str]] = []

    def activate(
        self,
        deployment_id: str,
        *,
        expected_runtime_revision: str,
        expected_catalog_revision: str,
    ) -> ActivationResult:
        self.calls.append(
            (
                deployment_id,
                expected_runtime_revision,
                expected_catalog_revision,
            )
        )
        self.entered.set()
        if self.release is not None and not self.release.wait(timeout=5):
            raise RuntimeError("test did not release the activation")
        if self.error is not None:
            raise self.error
        return ActivationResult(
            deployment_id=deployment_id,
            public_alias="local-agent",
            artifact_id="muse-glimmer-30b-kquant",
            changed=True,
            idempotent=False,
            ready=True,
            phase="ready",
            runtime_revision="rt1-after-activation",
            etag='"rt1-after-activation"',
        )


class FakeBenchmarkControlService(FakeControlService):
    def __init__(self) -> None:
        super().__init__()
        self.benchmark_context_active = False
        self.benchmark_calls: list[dict[str, Any]] = []

    @contextmanager
    def benchmark_launch(
        self,
        suite_id: str,
        deployment_id: str,
        *,
        expected_runtime_revision: str | None = None,
        if_match: str | None = None,
        expected_catalog_revision: str | None = None,
        collect_telemetry: bool = True,
    ) -> Iterator[BenchmarkLaunchInput]:
        self.benchmark_calls.append(
            {
                "suite_id": suite_id,
                "deployment_id": deployment_id,
                "expected_runtime_revision": expected_runtime_revision,
                "if_match": if_match,
                "expected_catalog_revision": expected_catalog_revision,
                "collect_telemetry": collect_telemetry,
            }
        )
        self.benchmark_context_active = True
        try:
            yield BenchmarkLaunchInput(
                suite_id=suite_id,
                suite_version="1",
                deployment_id=deployment_id,
                public_alias="local-agent",
                model_id="muse-glimmer-30b",
                artifact_id="muse-glimmer-30b-kquant",
                collect_telemetry=collect_telemetry,
                runtime_revision=expected_runtime_revision or "rt1-current",
                runtime_etag=f'"{expected_runtime_revision or "rt1-current"}"',
                catalog_revision="cat1-current",
                launch_revision="bench1-reviewed",
            )
        finally:
            self.benchmark_context_active = False


def _app(
    paths: LabPaths,
    service: FakeControlService,
    *,
    static_directory: Path | None = None,
) -> tuple[FastAPI, ConsoleController]:
    controller = ConsoleController(paths, control_service=service)  # type: ignore[arg-type]
    app = FastAPI()
    install_console(
        app,
        paths,
        controller=controller,
        static_directory=static_directory,
    )
    return app, controller


async def _wait_for_operation(
    client: httpx.AsyncClient,
    location: str,
    *,
    terminal_state: str,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    last: dict[str, Any] | None = None
    while loop.time() < deadline:
        response = await client.get(location)
        assert response.status_code == 200
        last = response.json()
        if last["state"] == terminal_state:
            return last
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"operation did not reach {terminal_state!r}; last document was {last!r}"
    )


@pytest.mark.asyncio
async def test_catalog_to_muse_activation_succeeds_and_can_be_polled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    service = FakeControlService()
    app, controller = _app(paths, service)
    monkeypatch.setattr(
        "llm_lab.console_api.NvidiaTelemetrySampler.sample_once",
        lambda self: (unavailable_sample("not sampled in API test"),),
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            portfolio_response = await client.get("/api/v1/models")
            runtime_response = await client.get("/api/v1/runtime")

            assert portfolio_response.status_code == 200
            portfolio = portfolio_response.json()
            muse = next(
                model for model in portfolio["models"] if model["id"] == "muse-glimmer-30b"
            )
            deployment = next(
                item for item in muse["deployments"] if item["id"] == MUSE_DEPLOYMENT
            )
            assert muse["display_name"] == "Muse Glimmer 30B"
            assert deployment["public_alias"] == "local-agent"
            assert deployment["artifact"]["id"] == "muse-glimmer-30b-kquant"
            assert portfolio["catalog_revision"].startswith("cat1-")

            assert runtime_response.status_code == 200
            runtime = runtime_response.json()
            assert runtime["revision"].startswith("rt1-")
            assert runtime_response.headers["etag"] == f'"{runtime["revision"]}"'

            accepted = await client.post(
                "/api/v1/runtime/activations",
                headers={
                    "If-Match": runtime_response.headers["etag"],
                    "Idempotency-Key": "activate-muse-e2e-001",
                },
                json={
                    "deployment_id": deployment["id"],
                    "catalog_revision": portfolio["catalog_revision"],
                },
            )

            assert accepted.status_code == 202
            assert accepted.json()["kind"] == "activate"
            assert accepted.json()["state"] == "queued"
            assert accepted.json()["requested_deployment_id"] == MUSE_DEPLOYMENT
            location = accepted.headers["location"]
            assert location == f'/api/v1/operations/{accepted.json()["id"]}'

            completed = await _wait_for_operation(
                client, location, terminal_state="succeeded"
            )

        assert completed["result"] == {
            "artifact_id": "muse-glimmer-30b-kquant",
            "changed": True,
            "deployment_id": MUSE_DEPLOYMENT,
            "etag": '"rt1-after-activation"',
            "idempotent": False,
            "phase": "ready",
            "public_alias": "local-agent",
            "ready": True,
            "runtime_revision": "rt1-after-activation",
        }
        assert service.calls == [
            (
                MUSE_DEPLOYMENT,
                runtime["revision"],
                portfolio["catalog_revision"],
            )
        ]
        with Registry(paths.registry_path) as registry:
            history = registry.list_history(
                entity_type="deployment", entity_id=MUSE_DEPLOYMENT
            )
        assert len(history) == 1
        assert history[0].event_type == "deployment.activated"
        assert history[0].payload == completed["result"]
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_activation_requires_runtime_and_idempotency_preconditions(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    service = FakeControlService()
    app, controller = _app(paths, service)
    body = {
        "deployment_id": MUSE_DEPLOYMENT,
        "catalog_revision": "cat1-current",
    }

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            missing_runtime = await client.post(
                "/api/v1/runtime/activations",
                headers={"Idempotency-Key": "activation-001"},
                json=body,
            )
            missing_idempotency = await client.post(
                "/api/v1/runtime/activations",
                headers={"If-Match": "rt1-current"},
                json=body,
            )

        assert missing_runtime.status_code == 428
        assert missing_runtime.json() == {
            "error": {
                "code": "precondition_required",
                "message": "A current runtime If-Match value is required.",
                "retryable": False,
                "details": {},
            }
        }
        assert missing_idempotency.status_code == 428
        assert missing_idempotency.json()["error"]["code"] == "idempotency_required"
        assert service.calls == []
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_activation_rejects_malformed_idempotency_key_with_structured_error(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    service = FakeControlService()
    app, controller = _app(paths, service)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            response = await client.post(
                "/api/v1/runtime/activations",
                headers={
                    "If-Match": '"rt1-current"',
                    "Idempotency-Key": "contains an invalid space",
                },
                json={
                    "deployment_id": MUSE_DEPLOYMENT,
                    "catalog_revision": "cat1-current",
                },
            )

        assert response.status_code == 400
        assert response.json() == {
            "error": {
                "code": "invalid_idempotency_key",
                "message": "Idempotency-Key must be a valid printable identifier.",
                "retryable": False,
                "details": {},
            }
        }
        assert service.calls == []
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_activation_idempotency_replays_and_rejects_key_rebinding(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    service = FakeControlService()
    app, controller = _app(paths, service)
    headers = {
        "If-Match": "rt1-current",
        "Idempotency-Key": "stable-browser-request-id",
    }
    body = {
        "deployment_id": MUSE_DEPLOYMENT,
        "catalog_revision": "cat1-current",
    }

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            accepted = await client.post(
                "/api/v1/runtime/activations", headers=headers, json=body
            )
            assert accepted.status_code == 202
            completed = await _wait_for_operation(
                client, accepted.headers["location"], terminal_state="succeeded"
            )

            replayed = await client.post(
                "/api/v1/runtime/activations", headers=headers, json=body
            )
            conflict = await client.post(
                "/api/v1/runtime/activations",
                headers=headers,
                json={**body, "deployment_id": "qwen3.8-27b-4090-8k"},
            )

        assert replayed.status_code == 200
        assert replayed.headers["location"] == accepted.headers["location"]
        assert replayed.json() == completed
        assert conflict.status_code == 409
        assert conflict.json()["error"] == {
            "code": "idempotency_conflict",
            "message": "That Idempotency-Key belongs to a different request.",
            "retryable": False,
            "details": {},
        }
        assert len(service.calls) == 1
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_activation_reports_busy_operation_without_starting_second_request(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    release = threading.Event()
    service = FakeControlService(release=release)
    app, controller = _app(paths, service)
    body = {
        "deployment_id": MUSE_DEPLOYMENT,
        "catalog_revision": "cat1-current",
    }

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            first = await client.post(
                "/api/v1/runtime/activations",
                headers={
                    "If-Match": "rt1-current",
                    "Idempotency-Key": "first-activation",
                },
                json=body,
            )
            assert first.status_code == 202
            assert await asyncio.to_thread(service.entered.wait, 2)

            busy = await client.post(
                "/api/v1/runtime/activations",
                headers={
                    "If-Match": "rt1-current",
                    "Idempotency-Key": "second-activation",
                },
                json={**body, "deployment_id": "qwen3.8-27b-4090-8k"},
            )
            assert busy.status_code == 409
            assert busy.json()["error"] == {
                "code": "operation_busy",
                "message": "Another lifecycle operation is already running.",
                "retryable": True,
                "details": {"active_operation_id": first.json()["id"]},
            }

            release.set()
            await _wait_for_operation(
                client, first.headers["location"], terminal_state="succeeded"
            )

        assert len(service.calls) == 1
    finally:
        release.set()
        controller.close()


@pytest.mark.asyncio
async def test_benchmark_worker_holds_launch_context_and_forwards_runtime_revision(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    service = FakeBenchmarkControlService()
    executor_calls: list[tuple[str, str, bool]] = []

    def execute_benchmark(
        suite_id: str, deployment_id: str, telemetry: bool
    ) -> Mapping[str, Any]:
        assert service.benchmark_context_active, (
            "benchmark executor ran after the validated runtime lease was released"
        )
        executor_calls.append((suite_id, deployment_id, telemetry))
        return {"run_id": "run-web-001", "status": "completed", "pass_rate": 1.0}

    controller = ConsoleController(
        paths,
        control_service=service,  # type: ignore[arg-type]
        benchmark_executor=execute_benchmark,
    )
    app = FastAPI()
    install_console(app, paths, controller=controller)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            accepted = await client.post(
                "/api/v1/benchmarks",
                headers={
                    "If-Match": '"rt1-reviewed-browser-state"',
                    "Idempotency-Key": "benchmark-from-web-001",
                },
                json={
                    "deployment_id": MUSE_DEPLOYMENT,
                    "suite_id": "smoke",
                    "telemetry": False,
                },
            )
            assert accepted.status_code == 202
            completed = await _wait_for_operation(
                client, accepted.headers["location"], terminal_state="succeeded"
            )

        assert service.benchmark_calls == [
            {
                "suite_id": "smoke",
                "deployment_id": MUSE_DEPLOYMENT,
                "expected_runtime_revision": "rt1-reviewed-browser-state",
                "if_match": None,
                "expected_catalog_revision": None,
                "collect_telemetry": False,
            }
        ]
        assert executor_calls == [("smoke", MUSE_DEPLOYMENT, False)]
        assert service.benchmark_context_active is False
        assert completed["result"] == {
            "run_id": "run-web-001",
            "status": "completed",
            "pass_rate": 1.0,
        }
    finally:
        controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_code", "expected_retryable"),
    (
        (
            RuntimeError(
                "Bearer raw-token password=hunter2 failed at /home/alice/private/model.gguf"
            ),
            "operation_failed",
            True,
        ),
        (
            ControlServiceError(
                "runtime_unavailable",
                "Bearer raw-token failed at /home/alice/private/model.gguf; token=hunter2",
                retryable=True,
            ),
            "runtime_unavailable",
            True,
        ),
    ),
)
async def test_activation_failure_documents_are_redacted(
    tmp_path: Path,
    error: BaseException,
    expected_code: str,
    expected_retryable: bool,
) -> None:
    paths = _paths(tmp_path)
    service = FakeControlService(error=error)
    app, controller = _app(paths, service)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            accepted = await client.post(
                "/api/v1/runtime/activations",
                headers={
                    "If-Match": "rt1-sensitive-revision",
                    "Idempotency-Key": "secret-browser-key",
                },
                json={
                    "deployment_id": MUSE_DEPLOYMENT,
                    "catalog_revision": "cat1-sensitive-revision",
                },
            )
            failed = await _wait_for_operation(
                client, accepted.headers["location"], terminal_state="failed"
            )

        assert failed["error"]["code"] == expected_code
        assert failed["error"]["retryable"] is expected_retryable
        rendered = json.dumps(failed)
        for private_value in (
            "raw-token",
            "hunter2",
            "/home/alice",
            "secret-browser-key",
            "rt1-sensitive-revision",
            "cat1-sensitive-revision",
        ):
            assert private_value not in rendered
        assert "idempotency_key" not in rendered
        assert "runtime_revision" not in rendered
        assert "catalog_revision" not in rendered
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_controller_close_waits_for_in_flight_activation_before_closing_store(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    release = threading.Event()
    service = FakeControlService(release=release)
    app, controller = _app(paths, service)
    close_task: asyncio.Task[None] | None = None

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            accepted = await client.post(
                "/api/v1/runtime/activations",
                headers={
                    "If-Match": "rt1-current",
                    "Idempotency-Key": "activation-during-shutdown",
                },
                json={
                    "deployment_id": MUSE_DEPLOYMENT,
                    "catalog_revision": "cat1-current",
                },
            )
            assert accepted.status_code == 202
            assert await asyncio.to_thread(service.entered.wait, 2)

            close_task = asyncio.create_task(asyncio.to_thread(controller.close))
            await asyncio.sleep(0.05)
            close_returned_before_worker = close_task.done()
            release.set()
            await asyncio.wait_for(close_task, timeout=2)

        # Join explicitly as test cleanup too. On the buggy implementation,
        # close() returns early and the worker is otherwise still unwinding.
        await asyncio.to_thread(controller._executor.shutdown, wait=True)
        with OperationStore(paths.data_root / "state/operations.sqlite") as store:
            persisted = store.get(accepted.json()["id"])

        assert not close_returned_before_worker, (
            "ConsoleController.close() returned while activation still owned its store"
        )
        assert persisted.status is OperationStatus.SUCCEEDED
        assert persisted.result is not None
        assert persisted.result["deployment_id"] == MUSE_DEPLOYMENT
    finally:
        release.set()
        if close_task is not None and not close_task.done():
            await asyncio.wait_for(close_task, timeout=2)
        await asyncio.to_thread(controller._executor.shutdown, wait=True)
        controller.close()


@pytest.mark.asyncio
async def test_unexpected_activation_exception_is_logged_but_publicly_redacted(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    paths = _paths(tmp_path)
    secret = "worker-secret-that-must-not-cross-the-api"
    service = FakeControlService(
        error=RuntimeError(f"backend exploded at /home/alice/model/private: token={secret}")
    )
    app, controller = _app(paths, service)
    caplog.set_level(logging.ERROR)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            accepted = await client.post(
                "/api/v1/runtime/activations",
                headers={
                    "If-Match": "rt1-current",
                    "Idempotency-Key": "activation-worker-error",
                },
                json={
                    "deployment_id": MUSE_DEPLOYMENT,
                    "catalog_revision": "cat1-current",
                },
            )
            failed = await _wait_for_operation(
                client, accepted.headers["location"], terminal_state="failed"
            )

        assert failed["error"] == {
            "code": "operation_failed",
            "message": (
                "The operation could not be completed. "
                "Inspect operator logs for details."
            ),
            "retryable": True,
        }
        assert secret not in json.dumps(failed)
        assert "/home/alice" not in json.dumps(failed)
        assert any(
            record.levelno >= logging.ERROR
            and record.exc_info is not None
            and record.exc_info[0] is RuntimeError
            for record in caplog.records
        ), "unexpected worker exception was not logged with traceback context"
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_operation_lookup_hides_invalid_and_unknown_identifiers(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    service = FakeControlService()
    app, controller = _app(paths, service)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            unknown = await client.get("/api/v1/operations/op_" + "0" * 32)
            invalid = await client.get("/api/v1/operations/not-an-operation")

        assert unknown.status_code == invalid.status_code == 404
        assert unknown.json() == invalid.json() == {
            "detail": "Operation was not found."
        }
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_sse_starts_with_healthy_snapshot_at_one_stable_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    broker = EventBroker(stream_id="a" * 32)
    broker.publish("runtime.changed", data={"reason": "seed cursor"})
    controller = ConsoleController(
        paths,
        control_service=FakeControlService(),  # type: ignore[arg-type]
        broker=broker,
    )
    app = FastAPI()
    install_console(app, paths, controller=controller)
    runtime_checks: list[bool] = []

    def runtime_snapshot(*, check_health: bool = True) -> dict[str, Any]:
        runtime_checks.append(check_health)
        return {
            "schema_version": 1,
            "revision": "rt1-sse-snapshot",
            "active": True,
            "ready": True,
            "running": True,
            "healthy": True,
            "phase": "ready",
            "deployment_id": MUSE_DEPLOYMENT,
        }

    def fail_if_waited(*args: object, **kwargs: object) -> None:
        raise AssertionError("SSE startup must not call blocking EventBroker.wait()")

    monkeypatch.setattr(controller, "runtime", runtime_snapshot)
    monkeypatch.setattr(broker, "wait", fail_if_waited)
    expected_cursor = f'{"a" * 32}:1'

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    routes = list(app.routes)
    for route in tuple(routes):
        included_router = getattr(route, "original_router", None)
        routes.extend(getattr(included_router, "routes", ()))
    event_route = next(
        route for route in routes if getattr(route, "path", None) == "/api/v1/events"
    )
    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/v1/events",
            "raw_path": b"/api/v1/events",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("console.test", 80),
        },
        receive,
    )

    try:
        response = await event_route.endpoint(request)
        first_frame = await anext(response.body_iterator)
        if isinstance(first_frame, bytes):
            first_frame = first_frame.decode("utf-8")
        await response.body_iterator.aclose()

        lines = first_frame.splitlines()
        payload = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        assert lines[:2] == [f"id: {expected_cursor}", "event: snapshot"]
        assert payload["id"] == expected_cursor == broker.cursor
        assert payload["type"] == "snapshot"
        assert payload["runtime"]["healthy"] is True
        assert runtime_checks == [True]
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_static_console_serves_assets_and_spa_fallback(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><title>LLM Lab test console</title>", encoding="utf-8"
    )
    (assets / "app.js").write_text("globalThis.CONSOLE_TEST = true;", encoding="utf-8")
    service = FakeControlService()
    app, controller = _app(paths, service, static_directory=dist)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
            follow_redirects=False,
        ) as client:
            root = await client.get("/")
            no_slash = await client.get("/ui")
            deep_link = await client.get("/ui/models")
            asset = await client.get("/ui/assets/app.js")
            missing_asset = await client.get("/ui/assets/missing.js")

        assert root.status_code == no_slash.status_code == 307
        assert root.headers["location"] == no_slash.headers["location"] == "/ui/"
        assert deep_link.status_code == 200
        assert "LLM Lab test console" in deep_link.text
        assert deep_link.headers["cache-control"] == "no-cache"
        assert asset.status_code == 200
        assert asset.text == "globalThis.CONSOLE_TEST = true;"
        assert asset.headers["cache-control"] == (
            "public, max-age=31536000, immutable"
        )
        assert missing_asset.status_code == 404
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_static_console_has_structured_error_when_not_built(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    service = FakeControlService()
    app, controller = _app(
        paths, service, static_directory=tmp_path / "does-not-exist"
    )

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://console.test",
        ) as client:
            response = await client.get("/ui/models")

        assert response.status_code == 503
        assert response.json()["error"] == {
            "code": "console_not_built",
            "message": "The web console has not been built for this installation.",
            "retryable": False,
            "details": {},
        }
    finally:
        controller.close()
