"""Local web-console API and static application adapter.

The console is deliberately a projection over the existing catalog, storage,
runtime, and benchmark services.  Browser clients can select catalog IDs, but
they can never submit executables, paths, environment variables, backend URLs,
or arbitrary command-line arguments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .catalog import Catalog
from .console_queries import ConsoleQueryService
from .control_service import ControlService, ControlServiceError
from .hashing import canonical_sha256
from .operations import (
    EventBroker,
    IdempotencyConflictError,
    OperationBusyError,
    OperationNotFoundError,
    OperationRecord,
    OperationStatus,
    OperationStore,
    SafeOperationError,
)
from .paths import LabPaths
from .registry import Registry
from .results import ResultsStore
from .runtime import RuntimeManager, RuntimeState, RuntimeStatus, read_active_state
from .telemetry import NvidiaTelemetrySampler


CONSOLE_SCHEMA_VERSION = 1
DEFAULT_STORAGE_RESERVE_BYTES = 540_000_000_000
LOGGER = logging.getLogger(__name__)
_TERMINAL_OPERATION_STATES = {
    OperationStatus.SUCCEEDED,
    OperationStatus.FAILED,
    OperationStatus.CANCELLED,
    OperationStatus.INTERRUPTED,
}


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActivationRequest(_RequestModel):
    deployment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    catalog_revision: str = Field(min_length=8, max_length=96)


class BenchmarkRequest(_RequestModel):
    deployment_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    suite_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    telemetry: bool = True


BenchmarkExecutor = Callable[[str, str, bool], Mapping[str, Any]]


class _SnapshotRuntimeReader:
    """Non-blocking runtime reader for console projections.

    Lifecycle mutations own the GPU lock.  Painting a catalog card must not
    queue behind a multi-minute model load, so reads inspect one safely loaded
    state snapshot without acquiring that mutation lock.
    """

    def __init__(self, manager: RuntimeManager) -> None:
        self.manager = manager
        self.paths = manager.paths

    def status(self, *, check_health: bool = True) -> RuntimeStatus:
        state = read_active_state(self.paths)
        if state is None:
            return RuntimeStatus(
                active=False,
                ready=False,
                running=False,
                healthy=None,
                state=None,
            )
        return self.manager.inspect_state(state, check_health=check_health)


def _public_error(
    status: int,
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "details": dict(details or {}),
            }
        },
    )


def _status_for_control_error(error: ControlServiceError) -> int:
    if error.code in {"unknown_deployment", "unknown_artifact", "unknown_suite"}:
        return 404
    if error.code in {"precondition_required"}:
        return 428
    if error.code.startswith("stale_"):
        return 412
    if error.code in {
        "artifact_not_installed",
        "artifact_catalog_mismatch",
        "artifact_verification_failed",
        "runtime_verification_failed",
        "deployment_not_active",
        "stale_active_deployment",
        "runtime_identity_mismatch",
    }:
        return 422
    if error.code in {"runtime_not_ready", "runtime_unavailable", "control_unavailable"}:
        return 503
    return 409


def _catalog_revision(catalog: Catalog) -> str:
    payload = {
        name: [entries[key].model_dump(mode="json") for key in sorted(entries)]
        for name, entries in (
            ("models", catalog.models),
            ("artifacts", catalog.artifacts),
            ("deployments", catalog.deployments),
            ("suites", catalog.suites),
            ("runtime_locks", catalog.runtime_locks),
        )
    }
    return "cat1-" + canonical_sha256(
        {"schema_version": 1, "catalog": payload}
    )


def _runtime_revision(state: RuntimeState | None) -> str:
    if state is None:
        payload: dict[str, Any] = {"schema_version": 1, "active": False}
    else:
        payload = {
            "schema_version": 1,
            "active": True,
            "phase": state.phase,
            "deployment_id": state.deployment.id,
            "artifact_id": state.deployment.artifact_id,
            "public_alias": state.deployment.public_alias,
            "backend": state.deployment.backend.value,
            "activated_at": state.activated_at,
            "launch": {
                "kind": state.launch.kind,
                "started_at": state.launch.started_at,
                "pid": state.launch.pid,
                "process_create_time": state.launch.process_create_time,
                "container_id": state.launch.container_id,
                "executable_sha256": state.launch.executable_sha256,
                "resolved_image": state.launch.resolved_image,
            },
            "has_error": state.error is not None,
        }
    return "rt1-" + canonical_sha256(payload)


def _datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


def _operation_document(record: OperationRecord) -> dict[str, Any]:
    request = record.request
    progress = record.progress
    result = record.result
    error = record.error
    requested_deployment = request.get("deployment_id")
    previous_deployment = progress.get("previous_deployment_id")
    public_state = (
        "failed" if record.status == OperationStatus.INTERRUPTED else record.status.value
    )
    return {
        "id": record.operation_id,
        "kind": record.kind,
        "state": public_state,
        "requested_deployment_id": (
            requested_deployment if isinstance(requested_deployment, str) else None
        ),
        "previous_deployment_id": (
            previous_deployment if isinstance(previous_deployment, str) else None
        ),
        "created_at": _datetime(record.created_at),
        "updated_at": _datetime(record.updated_at),
        "progress": (
            progress.get("percent")
            if isinstance(progress.get("percent"), (int, float))
            else None
        ),
        "message": progress.get("message") if isinstance(progress.get("message"), str) else None,
        "error": None if error is None else error.to_dict(),
        "result": None if result is None else dict(result),
        "revision": record.revision,
    }


class ConsoleController:
    """Own lazy console resources and execute admitted operations serially."""

    def __init__(
        self,
        paths: LabPaths,
        *,
        runtime_manager: RuntimeManager | None = None,
        control_service: ControlService | None = None,
        operation_store: OperationStore | None = None,
        broker: EventBroker | None = None,
        benchmark_executor: BenchmarkExecutor | None = None,
    ) -> None:
        self.paths = paths
        self.runtime_manager = runtime_manager or RuntimeManager(paths)
        self.control_service = control_service or ControlService(
            paths, runtime_manager=self.runtime_manager
        )
        self._provided_store = operation_store
        self._store: OperationStore | None = operation_store
        self._registry: Registry | None = None
        self.broker = broker or (operation_store.broker if operation_store else EventBroker())
        self._benchmark_executor = benchmark_executor or self._execute_benchmark
        self._resource_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-console")
        self._closing = False
        self._closed = False
        if self._store is not None:
            self._store.reconcile_incomplete()

    @property
    def store(self) -> OperationStore:
        with self._resource_lock:
            if self._closed:
                raise RuntimeError("console controller is closed")
            if self._store is None:
                self.paths.initialize()
                self._store = OperationStore(
                    self.paths.data_root / "state/operations.sqlite",
                    broker=self.broker,
                )
                self._store.reconcile_incomplete()
            return self._store

    def close(self) -> None:
        with self._resource_lock:
            if self._closed or self._closing:
                return
            self._closing = True

        # Do not close durable resources underneath an admitted operation.  A
        # graceful gateway shutdown waits for the serialized worker to record
        # its terminal state before releasing SQLite and registry handles.
        self._executor.shutdown(wait=True, cancel_futures=False)

        with self._resource_lock:
            if self._store is not None and self._provided_store is None:
                self._store.close()
                self._store = None
            if self._registry is not None:
                self._registry.close()
                self._registry = None
            self._closed = True
            self._closing = False

    def _catalog(self) -> Catalog:
        return Catalog.load(self.paths.catalog_root)

    def _query(
        self, *, include_results: bool = False
    ) -> tuple[ConsoleQueryService, ResultsStore | None]:
        self.paths.initialize()
        with self._resource_lock:
            if self._registry is None:
                self._registry = Registry(self.paths.registry_path)
            registry = self._registry
        results: ResultsStore | None = None
        if include_results:
            try:
                results = ResultsStore(self.paths.results_db_path)
            except BaseException:
                raise
        return (
            ConsoleQueryService(
                # Results are not touched by model/runtime/storage projections.
                # Avoid opening DuckDB on those latency-sensitive read paths.
                self._catalog(),
                registry,
                _SnapshotRuntimeReader(self.runtime_manager),  # type: ignore[arg-type]
                results,  # type: ignore[arg-type]
            ),
            results,
        )

    def portfolio(self) -> dict[str, Any]:
        service, results = self._query()
        try:
            response = service.models()
            catalog = service.catalog
            runtime = service.runtime_status(check_health=False)
            artifacts = {
                item.id: item
                for model in response.models
                for item in model.artifacts
            }
            models: list[dict[str, Any]] = []
            for model in response.models:
                deployments: list[dict[str, Any]] = []
                for deployment in model.deployments:
                    artifact = artifacts[deployment.artifact_id]
                    spec = catalog.deployments[deployment.id]
                    deployments.append(
                        {
                            "id": deployment.id,
                            "public_alias": deployment.public_alias,
                            "backend": deployment.backend,
                            "context_size": deployment.context_size,
                            "reasoning_mode": deployment.reasoning_mode,
                            "parallel": deployment.parallel,
                            "kv_cache": f"{spec.kv_cache_type_k}/{spec.kv_cache_type_v}",
                            "active": deployment.active,
                            "ready": deployment.ready,
                            "phase": (
                                runtime.lifecycle_phase
                                if deployment.active and runtime.lifecycle_phase
                                else "inactive"
                            ),
                            "artifact": {
                                "id": artifact.id,
                                "format": artifact.format,
                                "quantization": artifact.quantization,
                                "size_bytes": artifact.installed_size_bytes
                                or artifact.expected_size_bytes
                                or 0,
                                "registered": artifact.installation.value == "installed",
                                "verified": None,
                                "manifest_sha256": artifact.manifest_sha256,
                            },
                        }
                    )
                source_model = catalog.models[model.id]
                models.append(
                    {
                        "id": model.id,
                        "display_name": model.display_name,
                        "family": model.family,
                        "description": model.description,
                        "total_params_b": model.total_params_b,
                        "active_params_b": model.active_params_b,
                        "native_context": model.native_context,
                        "modalities": list(model.modalities),
                        "capabilities": list(model.capabilities),
                        "license": {
                            **model.license.model_dump(mode="json"),
                            "url": source_model.license.url,
                        },
                        "deployments": deployments,
                    }
                )
            return {
                "schema_version": CONSOLE_SCHEMA_VERSION,
                "catalog_revision": _catalog_revision(catalog),
                "models": models,
            }
        finally:
            if results is not None:
                results.close()

    def runtime(self, *, check_health: bool = True) -> dict[str, Any]:
        self.paths.initialize()
        try:
            state = read_active_state(self.paths)
            status = (
                None
                if state is None
                else self.runtime_manager.inspect_state(state, check_health=check_health)
            )
        except Exception:
            state = None
            status = None
        sample = NvidiaTelemetrySampler().sample_once()[0]
        catalog: Catalog | None
        try:
            catalog = self._catalog()
        except Exception:
            catalog = None
        model_name = None
        if state is not None and catalog is not None:
            artifact = catalog.artifacts.get(state.deployment.artifact_id)
            model = None if artifact is None else catalog.models.get(artifact.model_id)
            model_name = None if model is None else model.display_name
        # Status is a read path and must stay responsive while the durable
        # operation database is being opened or recovered.  Do not create that
        # database merely to paint the runtime strip.
        with self._resource_lock:
            opened_store = self._store
        active_operation = (
            None if opened_store is None else opened_store.active_mutation()
        )
        return {
            "schema_version": CONSOLE_SCHEMA_VERSION,
            "revision": _runtime_revision(state),
            "active": state is not None,
            "ready": bool(status and status.ready),
            "running": bool(status and status.running),
            "healthy": bool(status and status.healthy),
            "phase": "inactive" if state is None else state.phase,
            "deployment_id": None if state is None else state.deployment.id,
            "public_alias": None if state is None else state.deployment.public_alias,
            "model_name": model_name,
            "activated_at": None if state is None else state.activated_at,
            "context_size": None if state is None else state.deployment.context_size,
            "memory_used_mib": sample.memory_used_mib if sample.available else None,
            "memory_total_mib": sample.memory_total_mib if sample.available else None,
            "gpu_utilization_percent": (
                sample.gpu_utilization_percent if sample.available else None
            ),
            "operation": (
                None if active_operation is None else _operation_document(active_operation)
            ),
        }

    def storage(self) -> dict[str, Any]:
        service, results = self._query()
        try:
            report = service.storage_report()
            portfolio = service.models()
            active_id = portfolio.active_deployment_id
            active_artifact = (
                None
                if active_id is None
                else service.catalog.deployments[active_id].artifact_id
            )
            artifacts = [
                {
                    "id": artifact.id,
                    "model_id": artifact.model_id,
                    "format": artifact.format,
                    "quantization": artifact.quantization,
                    "logical_bytes": artifact.installed_size_bytes
                    or artifact.expected_size_bytes
                    or 0,
                    "registered": artifact.installation.value == "installed",
                    "active": artifact.id == active_artifact,
                    "manifest_sha256": artifact.manifest_sha256,
                }
                for model in portfolio.models
                for artifact in model.artifacts
                if artifact.installation.value != "missing"
            ]
            reserve = min(DEFAULT_STORAGE_RESERVE_BYTES, report.filesystem.total_bytes)
            return {
                "schema_version": CONSOLE_SCHEMA_VERSION,
                "total_bytes": report.filesystem.total_bytes,
                "free_bytes": report.filesystem.free_bytes,
                "used_bytes": report.filesystem.used_bytes,
                "reserve_bytes": reserve,
                "logical_bytes": report.registered.logical_bytes,
                "unique_blob_bytes": report.registered.unique_blob_bytes,
                "artifact_count": report.registered.artifact_count,
                "artifacts": artifacts,
            }
        finally:
            if results is not None:
                results.close()

    def runs(self) -> dict[str, Any]:
        service, results = self._query(include_results=True)
        try:
            rows = service.benchmark_runs().runs
            return {
                "schema_version": CONSOLE_SCHEMA_VERSION,
                "runs": [self._run_summary(row) for row in rows],
            }
        finally:
            assert results is not None
            results.close()

    def run(self, run_id: str) -> dict[str, Any]:
        service, results = self._query(include_results=True)
        try:
            detail = service.benchmark_run(run_id)
            return {
                **self._run_summary(detail),
                "suite_sha256": detail.suite_sha256,
                "request_contract_sha256": detail.request_contract_sha256,
                "cases": [
                    {
                        "case_id": item.case_id,
                        "pass_rate": item.pass_rate,
                        "sample_count": item.sample_count,
                        "error_count": item.error_count,
                        "mean_latency_ms": item.metrics.latency_ms.mean,
                        "mean_tokens_per_second": (
                            item.metrics.server_predicted_tokens_per_second.mean
                            or item.metrics.client_completion_tokens_per_second.mean
                        ),
                    }
                    for item in detail.cases
                ],
                # Only aggregate telemetry crosses the disclosure boundary.
                "telemetry": [],
                "telemetry_summary": detail.telemetry.model_dump(mode="json"),
            }
        finally:
            assert results is not None
            results.close()

    @staticmethod
    def _run_summary(row: Any) -> dict[str, Any]:
        return {
            "run_id": row.run_id,
            "suite_id": row.suite_id or "unknown",
            "suite_version": row.suite_version or "unknown",
            "deployment_id": row.deployment_id or "unknown",
            "model_id": row.model_id,
            "status": row.status,
            "started_at": _datetime(row.started_at),
            "finished_at": _datetime(row.finished_at),
            "pass_rate": row.pass_rate,
            "sample_count": row.sample_count,
            "error_count": row.error_count,
            "warmup_error_count": row.warmup_error_count,
            "mean_latency_ms": row.metrics.latency_ms.mean,
            "mean_tokens_per_second": (
                row.metrics.server_predicted_tokens_per_second.mean
                or row.metrics.client_completion_tokens_per_second.mean
            ),
            "peak_memory_mib": None,
        }

    def system(self) -> dict[str, Any]:
        self.paths.initialize()
        catalog = self._catalog()
        try:
            state = read_active_state(self.paths)
        except Exception:
            state = None
        lock = None
        if state is not None and state.deployment.runtime_lock_id:
            lock = catalog.runtime_locks.get(state.deployment.runtime_lock_id)
        sample = NvidiaTelemetrySampler().sample_once()[0]
        try:
            package_version = version("llm-lab")
        except PackageNotFoundError:
            package_version = "development"
        return {
            "schema_version": CONSOLE_SCHEMA_VERSION,
            "gateway_version": package_version,
            "catalog_valid": True,
            "catalog_revision": _catalog_revision(catalog),
            "runtime_lock_id": None if lock is None else lock.id,
            "runtime_sha256": (
                state.launch.executable_sha256
                if state is not None and state.launch.executable_sha256
                else (None if lock is None else lock.binary_sha256)
            ),
            "runtime_version": (
                None if state is None else state.launch.executable_version
            ),
            "gpu_name": sample.name if sample.available else None,
            "driver_version": _nvidia_driver_version(),
            "data_root_label": "LLM Lab managed local storage",
        }

    def operation(self, operation_id: str) -> dict[str, Any]:
        return _operation_document(self.store.get(operation_id))

    def admit(
        self,
        kind: Literal["activate", "stop", "benchmark"],
        request: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        with self._resource_lock:
            if self._closed or self._closing:
                raise RuntimeError("console controller is closing")
            admission = self.store.admit(
                kind,
                request=request,
                idempotency_key=idempotency_key,
                mutating=True,
            )
            if not admission.replayed:
                self._executor.submit(
                    self._execute_operation, admission.operation.operation_id
                )
        return _operation_document(admission.operation), admission.replayed

    def _execute_operation(self, operation_id: str) -> None:
        try:
            record = self.store.start(operation_id)
            request = record.request
            self.store.set_progress(
                operation_id, {"percent": 10, "message": "Validating reviewed inputs"}
            )
            if record.kind == "activate":
                self.store.set_progress(
                    operation_id, {"percent": 35, "message": "Switching deployment"}
                )
                activation = self.control_service.activate(
                    str(request["deployment_id"]),
                    expected_runtime_revision=str(request["runtime_revision"]),
                    expected_catalog_revision=str(request["catalog_revision"]),
                )
                result = activation.to_dict()
                if activation.changed:
                    try:
                        with Registry(self.paths.registry_path) as registry:
                            registry.record_history(
                                "deployment.activated",
                                "deployment",
                                activation.deployment_id,
                                result,
                            )
                    except Exception:
                        # The runtime is already ready and must not be reported
                        # as failed solely because secondary audit persistence
                        # is unavailable. Preserve the operator evidence.
                        LOGGER.exception(
                            "Could not record activation history for %s",
                            activation.deployment_id,
                        )
            elif record.kind == "stop":
                self.store.set_progress(
                    operation_id, {"percent": 45, "message": "Stopping active deployment"}
                )
                result = self.control_service.stop(
                    expected_runtime_revision=str(request["runtime_revision"])
                ).to_dict()
            elif record.kind == "benchmark":
                self.store.set_progress(
                    operation_id, {"percent": 20, "message": "Running benchmark suite"}
                )
                with self.control_service.benchmark_launch(
                    str(request["suite_id"]),
                    str(request["deployment_id"]),
                    expected_runtime_revision=str(request["runtime_revision"]),
                    collect_telemetry=bool(request["telemetry"]),
                ) as launch:
                    result = dict(
                        self._benchmark_executor(
                            launch.suite_id,
                            launch.deployment_id,
                            launch.collect_telemetry,
                        )
                    )
            else:  # OperationStore accepts generic kinds, but this owner does not.
                raise SafeOperationError("unsupported_operation", "Unsupported operation.")
            self.store.succeed(operation_id, result=result)
            self.broker.publish(
                "runtime.changed",
                operation_id=operation_id,
                data={"operation_id": operation_id},
            )
        except ControlServiceError as exc:
            self.store.fail(
                operation_id,
                code=exc.code,
                message=exc.public_message,
                retryable=exc.retryable,
            )
        except SafeOperationError as exc:
            self.store.fail(operation_id, error=exc)
        except Exception:
            LOGGER.exception("Console operation %s failed", operation_id)
            self.store.fail(
                operation_id,
                code="operation_failed",
                message="The operation could not be completed. Inspect operator logs for details.",
                retryable=True,
            )

    def _execute_benchmark(
        self, suite_id: str, deployment_id: str, telemetry: bool
    ) -> Mapping[str, Any]:
        # The existing CLI function is already the tested application service
        # for bundle creation, attestation, registry history, and DuckDB index.
        # Calling it directly keeps the console out of shell-command territory.
        from .cli import CliState, _execute_benchmark

        run, _ = asyncio.run(
            _execute_benchmark(
                CliState(paths=self.paths),
                suite_id,
                deployment_id,
                None,
                telemetry,
                None,
            )
        )
        return {
            "run_id": run.run_id,
            "status": str(run.run.get("status", "unknown")),
            "pass_rate": run.summary.get("pass_rate"),
        }


def _nvidia_driver_version() -> str | None:
    try:
        completed = __import__("subprocess").run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    first = completed.stdout.splitlines()[0].strip() if completed.stdout.splitlines() else ""
    return first if first and len(first) <= 64 else None


def install_console(
    app: FastAPI,
    paths: LabPaths,
    *,
    runtime_manager: RuntimeManager | None = None,
    controller: ConsoleController | None = None,
    static_directory: str | Path | None = None,
) -> ConsoleController:
    """Install management routes and the compiled SPA on an existing app."""

    owner = controller or ConsoleController(paths, runtime_manager=runtime_manager)
    router = APIRouter(prefix="/api/v1", tags=["console"])

    @router.get("/models")
    async def models() -> dict[str, Any]:
        return await asyncio.to_thread(owner.portfolio)

    @router.get("/runtime")
    async def runtime() -> JSONResponse:
        document = await asyncio.to_thread(owner.runtime)
        return JSONResponse(document, headers={"ETag": f'"{document["revision"]}"'})

    @router.get("/storage")
    async def storage() -> dict[str, Any]:
        return await asyncio.to_thread(owner.storage)

    @router.get("/system")
    async def system() -> dict[str, Any]:
        return await asyncio.to_thread(owner.system)

    @router.get("/runs")
    async def runs() -> dict[str, Any]:
        return await asyncio.to_thread(owner.runs)

    @router.get("/runs/{run_id}")
    async def run(run_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(owner.run, run_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Benchmark run was not found.")

    @router.get("/operations/{operation_id}")
    async def operation(operation_id: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(owner.operation, operation_id)
        except (OperationNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="Operation was not found.")

    def require_preconditions(
        if_match: str | None, idempotency_key: str | None
    ) -> JSONResponse | None:
        if not if_match:
            return _public_error(
                428, "precondition_required", "A current runtime If-Match value is required."
            )
        if not idempotency_key:
            return _public_error(
                428, "idempotency_required", "An Idempotency-Key header is required."
            )
        return None

    async def admit_operation(
        kind: Literal["activate", "stop", "benchmark"],
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> JSONResponse:
        try:
            document, replayed = await asyncio.to_thread(
                owner.admit,
                kind,
                payload,
                idempotency_key=idempotency_key,
            )
        except OperationBusyError as exc:
            return _public_error(
                409,
                "operation_busy",
                "Another lifecycle operation is already running.",
                retryable=True,
                details={"active_operation_id": exc.active_operation.operation_id},
            )
        except IdempotencyConflictError:
            return _public_error(
                409,
                "idempotency_conflict",
                "That Idempotency-Key belongs to a different request.",
            )
        except ValueError:
            return _public_error(
                400,
                "invalid_idempotency_key",
                "Idempotency-Key must be a valid printable identifier.",
            )
        status = 200 if replayed else 202
        return JSONResponse(
            document,
            status_code=status,
            headers={"Location": f'/api/v1/operations/{document["id"]}'},
        )

    @router.post("/runtime/activations", response_model=None)
    async def activate(
        body: ActivationRequest,
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        invalid = require_preconditions(if_match, idempotency_key)
        if invalid is not None:
            return invalid
        return await admit_operation(
            "activate",
            {
                "deployment_id": body.deployment_id,
                "catalog_revision": body.catalog_revision,
                "runtime_revision": if_match.strip('"'),
            },
            idempotency_key,
        )

    @router.delete("/runtime/active", response_model=None)
    async def stop(
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        invalid = require_preconditions(if_match, idempotency_key)
        if invalid is not None:
            return invalid
        return await admit_operation(
            "stop",
            {"runtime_revision": if_match.strip('"')},
            idempotency_key,
        )

    @router.post("/benchmarks", response_model=None)
    async def benchmark(
        body: BenchmarkRequest,
        if_match: str | None = Header(default=None, alias="If-Match"),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        invalid = require_preconditions(if_match, idempotency_key)
        if invalid is not None:
            return invalid
        return await admit_operation(
            "benchmark",
            {
                "deployment_id": body.deployment_id,
                "suite_id": body.suite_id,
                "telemetry": body.telemetry,
                "runtime_revision": if_match.strip('"'),
            },
            idempotency_key,
        )

    @router.get("/events")
    async def events(request: Request) -> StreamingResponse:
        cursor = request.headers.get("last-event-id")

        async def stream() -> AsyncIterator[str]:
            anchor = owner.broker.cursor
            runtime_document = await asyncio.to_thread(owner.runtime, check_health=True)
            snapshot = {
                "id": anchor,
                "type": "snapshot",
                "runtime": runtime_document,
            }
            yield (
                f"id: {anchor}\n"
                "event: snapshot\n"
                "data: " + json.dumps(snapshot, separators=(",", ":")) + "\n\n"
            )
            after = cursor or anchor
            loop = asyncio.get_running_loop()
            heartbeat_at = loop.time() + 15.0
            while True:
                if await request.is_disconnected():
                    return
                replay = owner.broker.replay(after)
                if replay.reset_required:
                    runtime_document = await asyncio.to_thread(
                        owner.runtime, check_health=True
                    )
                    payload = {
                        "id": replay.cursor,
                        "type": "snapshot",
                        "runtime": runtime_document,
                    }
                    yield (
                        f"id: {replay.cursor}\n"
                        "event: snapshot\n"
                        "data: "
                        + json.dumps(payload, separators=(",", ":"))
                        + "\n\n"
                    )
                    after = replay.cursor
                    continue
                if not replay.events:
                    if loop.time() >= heartbeat_at:
                        yield ": heartbeat\n\n"
                        heartbeat_at = loop.time() + 15.0
                    await asyncio.sleep(0.25)
                    continue
                for event in replay.events:
                    if event.event_type == "runtime.changed":
                        payload = {
                            "id": event.event_id,
                            "type": "runtime.changed",
                            "runtime": await asyncio.to_thread(
                                owner.runtime, check_health=True
                            ),
                        }
                        event_name = "runtime.changed"
                    else:
                        try:
                            operation_document = await asyncio.to_thread(
                                owner.operation, event.operation_id or ""
                            )
                        except Exception:
                            continue
                        payload = {
                            "id": event.event_id,
                            "type": "operation.changed",
                            "operation": operation_document,
                        }
                        event_name = "operation.changed"
                    yield (
                        f"id: {event.event_id}\n"
                        f"event: {event_name}\n"
                        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                    )
                    after = event.event_id

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    app.include_router(router)
    app.state.console_controller = owner

    dist = Path(static_directory) if static_directory else Path(__file__).with_name("web_dist")

    @app.get("/", include_in_schema=False)
    async def console_root() -> RedirectResponse:
        return RedirectResponse("/ui/", status_code=307)

    @app.get("/ui", include_in_schema=False)
    async def console_no_slash() -> RedirectResponse:
        return RedirectResponse("/ui/", status_code=307)

    @app.get("/ui/{asset_path:path}", include_in_schema=False, response_model=None)
    async def console_asset(asset_path: str) -> FileResponse | JSONResponse:
        if not dist.is_dir():
            return _public_error(
                503,
                "console_not_built",
                "The web console has not been built for this installation.",
            )
        relative = Path(asset_path or "index.html")
        candidate = (dist / relative).resolve()
        root = dist.resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            response = FileResponse(candidate)
            if relative.parts and relative.parts[0] == "assets":
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "no-cache"
            return response
        if relative.parts and relative.parts[0] == "assets":
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})

    return owner


__all__ = [
    "ActivationRequest",
    "BenchmarkRequest",
    "ConsoleController",
    "install_console",
]
