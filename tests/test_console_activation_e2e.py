from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from llm_lab.catalog import Catalog
from llm_lab.console_api import ConsoleController
from llm_lab.control_service import ControlService
from llm_lab.gateway import create_app
from llm_lab.mock_backend import create_mock_app
from llm_lab.operations import EventBroker, OperationStatus, OperationStore
from llm_lab.paths import LabPaths
from llm_lab.runtime import LaunchPlan, LaunchRecord, RuntimeManager
from llm_lab.storage import ArtifactStore


MUSE_DEPLOYMENT_ID = "muse-glimmer-30b-4090-8k"
MUSE_ARTIFACT_ID = "muse-glimmer-30b-kquant"
MUSE_MODEL_ID = "muse-glimmer-30b"
MUSE_ALIAS = "local-agent"
ARTIFACT_PAYLOAD = b"tiny reviewed mock artifact\n"
PRIVATE_VALUE = "do-not-disclose"


class RecordingLauncher:
    """Implement RuntimeManager's launcher seam without starting a process."""

    def __init__(self) -> None:
        self.started: list[LaunchPlan] = []
        self.stopped: list[LaunchRecord] = []
        self.running: set[int] = set()
        self._next_pid = 40_000

    def start(self, plan: LaunchPlan) -> LaunchRecord:
        assert plan.kind == "process"
        self._next_pid += 1
        self.started.append(plan)
        self.running.add(self._next_pid)
        return LaunchRecord(
            kind="process",
            command=plan.command,
            started_at="2026-09-06T12:00:00+00:00",
            pid=self._next_pid,
            process_create_time=float(self._next_pid),
            resolved_executable=plan.command[0],
        )

    def is_running(self, record: LaunchRecord) -> bool:
        return record.pid in self.running

    def stop(self, record: LaunchRecord, timeout_seconds: float = 10.0) -> None:
        self.stopped.append(record)
        if record.pid is not None:
            self.running.discard(record.pid)


def _install_test_catalog_and_artifact(paths: LabPaths, source: Path) -> Catalog:
    source.mkdir(parents=True)
    (source / "model.gguf").write_bytes(ARTIFACT_PAYLOAD)
    paths.catalog_root.mkdir(parents=True)
    document = {
        "models": [
            {
                "schema_version": 1,
                "id": MUSE_MODEL_ID,
                "display_name": "Muse Glimmer 30B",
                "family": "muse-glimmer",
                "description": "A tiny local fixture exercising the Muse profile.",
                "total_params_b": 29.6,
                "active_params_b": 29.6,
                "native_context": 131_072,
                "modalities": ["text", "image"],
                "capabilities": ["chat", "reasoning", "tools", "vision"],
                "license": {
                    "name": "Apache-2.0",
                    "commercial_use": True,
                    "osi_approved": True,
                    "acceptance_required": False,
                },
                "upstream": {
                    "provider": "local",
                    "local_path": str(source),
                    "revision": "fixture-v1",
                },
            }
        ],
        "artifacts": [
            {
                "schema_version": 1,
                "id": MUSE_ARTIFACT_ID,
                "model_id": MUSE_MODEL_ID,
                "source": {
                    "provider": "local",
                    "local_path": str(source),
                    "revision": "fixture-v1",
                },
                "format": "gguf",
                "quantization": "test-only",
                "expected_size_bytes": len(ARTIFACT_PAYLOAD),
                "files": [{"pattern": "model.gguf", "role": "weights"}],
                "requires_remote_code": False,
            }
        ],
        "deployments": [
            {
                "schema_version": 1,
                "id": MUSE_DEPLOYMENT_ID,
                "artifact_id": MUSE_ARTIFACT_ID,
                "public_alias": MUSE_ALIAS,
                "backend": "mock",
                "host": "127.0.0.1",
                "port": 18_081,
                "context_size": 8_192,
                "parallel": 1,
                "reasoning_mode": "auto",
                "startup_timeout_seconds": 1,
                "environment": {"PRIVATE_VALUE": PRIVATE_VALUE},
            }
        ],
    }
    (paths.catalog_root / "catalog.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    paths.initialize()

    catalog = Catalog.load(paths.catalog_root)
    with ArtifactStore(paths) as artifact_store:
        promoted = artifact_store.promote(
            catalog.get_artifact(MUSE_ARTIFACT_ID),
            source,
            resolved_revision="fixture-v1",
        )
        assert promoted.manifest.total_logical_bytes == len(ARTIFACT_PAYLOAD)
        assert artifact_store.verify(MUSE_ARTIFACT_ID).view_verified
    return catalog


async def _poll_terminal_operation(
    client: httpx.AsyncClient, location: str
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + 3
    while True:
        response = await client.get(location)
        assert response.status_code == 200
        document = response.json()
        if document["state"] in {"succeeded", "failed", "cancelled"}:
            return document
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(f"activation did not finish; last response: {document!r}")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_muse_activation_succeeds_through_the_real_control_stack(
    tmp_path: Path,
) -> None:
    paths = LabPaths.discover(
        repo_root=tmp_path / "repo", data_root=tmp_path / "data"
    )
    catalog = _install_test_catalog_and_artifact(paths, tmp_path / "source")
    launcher = RecordingLauncher()
    readiness_probes: list[tuple[str, float]] = []

    def report_ready(url: str, timeout: float) -> None:
        readiness_probes.append((url, timeout))

    runtime_manager = RuntimeManager(
        paths,
        process_launcher=launcher,
        health_checker=report_ready,
    )
    control_service = ControlService(paths, runtime_manager=runtime_manager)
    broker = EventBroker(stream_id="e" * 32)
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_mock_app(MUSE_ALIAS)),
        base_url="http://backend.invalid",
    )

    with OperationStore(
        paths.data_root / "state/operations.sqlite", broker=broker
    ) as store:
        controller = ConsoleController(
            paths,
            runtime_manager=runtime_manager,
            control_service=control_service,
            operation_store=store,
            broker=broker,
        )
        app = create_app(
            paths,
            client=upstream,
            runtime_manager=runtime_manager,
            console_controller=controller,
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://gateway.invalid",
            ) as client:
                models_before = await client.get("/api/v1/models")
                runtime_before = await client.get("/api/v1/runtime")

                assert models_before.status_code == 200
                assert runtime_before.status_code == 200
                assert runtime_before.json()["active"] is False
                assert runtime_before.headers["etag"] == (
                    f'"{runtime_before.json()["revision"]}"'
                )
                muse_before = next(
                    model
                    for model in models_before.json()["models"]
                    if model["id"] == MUSE_MODEL_ID
                )
                deployment_before = muse_before["deployments"][0]
                assert deployment_before["id"] == MUSE_DEPLOYMENT_ID
                assert deployment_before["artifact"]["registered"] is True
                assert deployment_before["active"] is False
                assert deployment_before["ready"] is False

                activation_payload = {
                    "deployment_id": MUSE_DEPLOYMENT_ID,
                    "catalog_revision": models_before.json()["catalog_revision"],
                }
                activation_headers = {
                    "If-Match": runtime_before.headers["etag"],
                    "Idempotency-Key": "muse-glimmer-activation-e2e-001",
                }
                accepted = await client.post(
                    "/api/v1/runtime/activations",
                    json=activation_payload,
                    headers=activation_headers,
                )

                assert accepted.status_code == 202
                accepted_document = accepted.json()
                assert accepted.headers["location"] == (
                    f'/api/v1/operations/{accepted_document["id"]}'
                )
                assert accepted_document["kind"] == "activate"
                assert accepted_document["state"] == "queued"
                assert accepted_document["requested_deployment_id"] == (
                    MUSE_DEPLOYMENT_ID
                )
                assert accepted_document["result"] is None
                assert accepted_document["error"] is None

                operation = await _poll_terminal_operation(
                    client, accepted.headers["location"]
                )
                assert operation["state"] == "succeeded"
                assert operation["error"] is None
                assert operation["requested_deployment_id"] == MUSE_DEPLOYMENT_ID
                assert operation["progress"] == 35
                assert operation["message"] == "Switching deployment"
                assert operation["result"] == {
                    "deployment_id": MUSE_DEPLOYMENT_ID,
                    "public_alias": MUSE_ALIAS,
                    "artifact_id": MUSE_ARTIFACT_ID,
                    "changed": True,
                    "idempotent": False,
                    "ready": True,
                    "phase": "ready",
                    "runtime_revision": operation["result"]["runtime_revision"],
                    "etag": operation["result"]["etag"],
                }

                runtime_after = await client.get("/api/v1/runtime")
                models_after = await client.get("/api/v1/models")
                health_after = await client.get("/health")
                served_models = await client.get("/v1/models")
                chat = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": MUSE_ALIAS,
                        "messages": [
                            {"role": "user", "content": "Muse is active"}
                        ],
                    },
                )

                assert runtime_after.status_code == 200
                runtime_document = runtime_after.json()
                assert runtime_after.headers["etag"] == (
                    f'"{runtime_document["revision"]}"'
                )
                assert operation["result"]["runtime_revision"] == (
                    runtime_document["revision"]
                )
                assert operation["result"]["etag"] == runtime_after.headers["etag"]
                assert {
                    "active": runtime_document["active"],
                    "ready": runtime_document["ready"],
                    "running": runtime_document["running"],
                    "healthy": runtime_document["healthy"],
                    "phase": runtime_document["phase"],
                    "deployment_id": runtime_document["deployment_id"],
                    "public_alias": runtime_document["public_alias"],
                    "model_name": runtime_document["model_name"],
                    "operation": runtime_document["operation"],
                } == {
                    "active": True,
                    "ready": True,
                    "running": True,
                    "healthy": True,
                    "phase": "ready",
                    "deployment_id": MUSE_DEPLOYMENT_ID,
                    "public_alias": MUSE_ALIAS,
                    "model_name": "Muse Glimmer 30B",
                    "operation": None,
                }

                assert models_after.status_code == 200
                assert models_after.json()["catalog_revision"] == (
                    models_before.json()["catalog_revision"]
                )
                muse_after = next(
                    model
                    for model in models_after.json()["models"]
                    if model["id"] == MUSE_MODEL_ID
                )
                deployment_after = muse_after["deployments"][0]
                assert deployment_after["id"] == MUSE_DEPLOYMENT_ID
                assert deployment_after["artifact"]["registered"] is True
                assert deployment_after["active"] is True
                assert deployment_after["ready"] is True
                assert deployment_after["phase"] == "ready"

                assert health_after.status_code == 200
                assert health_after.json()["status"] == "ok"
                assert health_after.json()["active"] is True
                assert health_after.json()["ready"] is True
                assert health_after.json()["deployment"] == MUSE_DEPLOYMENT_ID
                assert health_after.json()["model"] == MUSE_ALIAS
                assert served_models.status_code == 200
                assert served_models.json()["data"][0]["id"] == MUSE_ALIAS
                assert chat.status_code == 200
                assert chat.headers["x-llm-lab-mock"] == "true"
                assert chat.json()["model"] == MUSE_ALIAS
                assert chat.json()["choices"][0]["message"]["content"] == (
                    "mock response: Muse is active"
                )

                replayed = await client.post(
                    "/api/v1/runtime/activations",
                    json=activation_payload,
                    headers=activation_headers,
                )
                assert replayed.status_code == 200
                assert replayed.headers["location"] == accepted.headers["location"]
                assert replayed.json() == operation
        finally:
            controller.close()
            await upstream.aclose()

        persisted = store.get(accepted_document["id"])
        assert persisted.status is OperationStatus.SUCCEEDED
        assert persisted.result is not None
        assert persisted.result["deployment_id"] == MUSE_DEPLOYMENT_ID

        assert len(launcher.started) == 1
        launch_plan = launcher.started[0]
        assert launch_plan.deployment_id == MUSE_DEPLOYMENT_ID
        assert launch_plan.command[:3] == (
            sys.executable,
            "-m",
            "llm_lab.mock_backend",
        )
        assert launch_plan.environment["PRIVATE_VALUE"] == PRIVATE_VALUE
        assert launcher.stopped == []
        assert readiness_probes
        assert all(url == "http://127.0.0.1:18081/health" for url, _ in readiness_probes)

        events = broker.replay().events
        assert [event.event_type for event in events] == [
            "operation.queued",
            "operation.running",
            "operation.progress",
            "operation.progress",
            "operation.succeeded",
            "runtime.changed",
        ]
        assert all(event.operation_id == accepted_document["id"] for event in events)
        assert events[-1].data == {"operation_id": accepted_document["id"]}
        public_documents = [
            operation,
            runtime_document,
            models_after.json(),
            *[event.to_dict() for event in events],
        ]
        assert PRIVATE_VALUE not in str(public_documents)
        assert str(paths.data_root) not in str(public_documents)
        assert catalog.get_deployment(MUSE_DEPLOYMENT_ID).backend.value == "mock"
