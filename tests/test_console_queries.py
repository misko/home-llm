from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm_lab.catalog import Catalog
from llm_lab.console_queries import ConsoleQueryService
from llm_lab.console_schema import (
    ArtifactInstallation,
    ConsoleArtifact,
    RuntimeCondition,
)
from llm_lab.errors import CatalogError
from llm_lab.hashing import canonical_sha256, sha256_bytes, sha256_uri
from llm_lab.paths import LabPaths
from llm_lab.registry import Registry
from llm_lab.results import ResultsStore, write_run_bundle
from llm_lab.runtime import LaunchPlan, LaunchRecord, RuntimeState, RuntimeStatus
from llm_lab.schema import (
    ArtifactFileSelector,
    ArtifactFormat,
    ArtifactManifest,
    ArtifactSpec,
    BackendKind,
    DeploymentSpec,
    FileRole,
    LicenseSpec,
    LockedFile,
    ModelSpec,
    RepositorySource,
)


SECRET = "super-secret-token"
PRIVATE_PATH = "/srv/private/models/model.gguf"
PRIVATE_URL = "http://127.0.0.1:19999/private"


class StubRuntimeManager:
    def __init__(
        self,
        paths: LabPaths,
        status: RuntimeStatus | Exception,
    ) -> None:
        self.paths = paths
        self._status = status
        self.health_checks: list[bool] = []

    def status(self, *, check_health: bool = True) -> RuntimeStatus:
        self.health_checks.append(check_health)
        if isinstance(self._status, Exception):
            raise self._status
        return self._status


def _catalog() -> Catalog:
    source_a = RepositorySource(provider="local", local_path="/private/source-a")
    source_b = RepositorySource(provider="local", local_path="/private/source-b")
    model_a = ModelSpec(
        id="model-a",
        display_name="Model A",
        family="fixture",
        description="A compact test model.",
        total_params_b=7,
        active_params_b=3,
        native_context=32768,
        modalities=("text", "image"),
        capabilities=("chat", "vision", "tools"),
        license=LicenseSpec(
            name="Apache-2.0",
            commercial_use=True,
            osi_approved=True,
        ),
        upstream=source_a,
    )
    model_b = ModelSpec(
        id="model-b",
        display_name="Model B",
        family="fixture",
        description="An uninstalled test model.",
        total_params_b=2,
        active_params_b=2,
        native_context=8192,
        license=LicenseSpec(
            name="MIT",
            commercial_use=True,
            osi_approved=True,
        ),
        upstream=source_b,
    )
    artifact_a = ArtifactSpec(
        id="artifact-a",
        model_id="model-a",
        source=source_a,
        format=ArtifactFormat.GGUF,
        quantization="Q4_K_M",
        effective_bpw=4.8,
        expected_size_bytes=7,
        files=(
            ArtifactFileSelector(pattern="model.gguf", role=FileRole.WEIGHTS),
        ),
    )
    artifact_b = ArtifactSpec(
        id="artifact-b",
        model_id="model-b",
        source=source_b,
        format=ArtifactFormat.GGUF,
        quantization="Q4_K_M",
        expected_size_bytes=7,
        files=(
            ArtifactFileSelector(pattern="model.gguf", role=FileRole.WEIGHTS),
        ),
    )
    deployment_a = DeploymentSpec(
        id="deployment-a",
        artifact_id="artifact-a",
        public_alias="local-a",
        backend=BackendKind.LLAMA_CPP,
        executable="/srv/private/bin/llama-server",
        context_size=8192,
        parallel=1,
        reasoning_mode="off",
        environment={"MODEL_TOKEN": SECRET},
        extra_args=("--log-file", "/srv/private/runtime.log"),
    )
    deployment_b = DeploymentSpec(
        id="deployment-b",
        artifact_id="artifact-b",
        public_alias="local-b",
        backend=BackendKind.EXTERNAL,
        external_base_url=PRIVATE_URL,
        context_size=4096,
        reasoning_mode="auto",
    )
    return Catalog(
        models={model_a.id: model_a, model_b.id: model_b},
        artifacts={artifact_a.id: artifact_a, artifact_b.id: artifact_b},
        deployments={deployment_a.id: deployment_a, deployment_b.id: deployment_b},
        suites={},
        runtime_locks={},
    )


def _paths(tmp_path: Path) -> LabPaths:
    repo = tmp_path / "repo"
    repo.mkdir()
    paths = LabPaths.discover(repo_root=repo, data_root=tmp_path / "private-data")
    paths.initialize()
    return paths


def _manifest(
    spec: ArtifactSpec,
    *,
    digest: str | None = None,
    resolved_revision: str | None = None,
) -> ArtifactManifest:
    digest = digest or sha256_bytes(b"weights")
    return ArtifactManifest(
        artifact_id=spec.id,
        model_id=spec.model_id,
        source=spec.source,
        resolved_revision=resolved_revision or spec.source.revision,
        format=spec.format,
        quantization=spec.quantization,
        effective_bpw=spec.effective_bpw,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        files=(
            LockedFile(
                logical_path="private/model.gguf",
                role=FileRole.WEIGHTS,
                size_bytes=7,
                sha256=digest,
                storage_uri=sha256_uri(digest),
            ),
        ),
        total_logical_bytes=7,
        tree_sha256="1" * 64,
    )


def _runtime_state(
    deployment: DeploymentSpec,
    *,
    phase: str = "ready",
    error: str | None = None,
) -> RuntimeState:
    return RuntimeState(
        schema_version=1,
        phase=phase,  # type: ignore[arg-type]
        deployment=deployment,
        artifact_path=PRIVATE_PATH,
        plan=LaunchPlan(
            kind="process",
            deployment_id=deployment.id,
            command=(PRIVATE_PATH, "--api-key", SECRET),
            environment={"API_KEY": SECRET},
            base_url=PRIVATE_URL,
            health_url=f"{PRIVATE_URL}/health",
            log_path="/srv/private/runtime.log",
            working_directory="/srv/private",
            expected_executable_root="/srv/private",
        ),
        launch=LaunchRecord(
            kind="process",
            command=(PRIVATE_PATH, "--api-key", SECRET),
            started_at="2026-09-06T12:00:00Z",
            pid=4242,
            process_create_time=100.5,
            resolved_executable=PRIVATE_PATH,
            executable_sha256="a" * 64,
            executable_version=f"server {SECRET}",
        ),
        activated_at="2026-09-06T12:00:00Z",
        error=error,
    )


def _service(
    tmp_path: Path,
    *,
    status: RuntimeStatus | Exception | None = None,
) -> tuple[ConsoleQueryService, Registry, ResultsStore, StubRuntimeManager]:
    catalog = _catalog()
    paths = _paths(tmp_path)
    registry = Registry(paths.registry_path)
    results = ResultsStore(paths.results_db_path)
    manager = StubRuntimeManager(
        paths,
        status
        or RuntimeStatus(
            active=False,
            ready=False,
            running=False,
            healthy=None,
            state=None,
        ),
    )
    service = ConsoleQueryService(
        catalog,
        registry,
        manager,  # type: ignore[arg-type]
        results,
        disk_usage=lambda _: SimpleNamespace(total=1000, used=400, free=600),
    )
    return service, registry, results, manager


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(map(_keys, value.values())))
    if isinstance(value, list):
        return set().union(*(map(_keys, value))) if value else set()
    return set()


FORBIDDEN_KEYS = {
    "artifact_path",
    "base_url",
    "command",
    "container_id",
    "container_name",
    "environment",
    "executable",
    "executable_sha256",
    "executable_version",
    "external_base_url",
    "health_url",
    "image",
    "local_path",
    "log_path",
    "mounts",
    "path",
    "pid",
    "plan",
    "process_create_time",
    "requested_image",
    "resolved_executable",
    "resolved_image",
    "runtime",
    "working_directory",
}


def _assert_redacted(value: Any) -> None:
    document = json.loads(value.model_dump_json())
    rendered = json.dumps(document, sort_keys=True)
    assert SECRET not in rendered
    assert PRIVATE_PATH not in rendered
    assert PRIVATE_URL not in rendered
    assert _keys(document).isdisjoint(FORBIDDEN_KEYS)


def test_models_join_catalog_installation_and_ready_runtime(tmp_path: Path) -> None:
    catalog = _catalog()
    active_state = _runtime_state(catalog.get_deployment("deployment-a"))
    status = RuntimeStatus(
        active=True,
        ready=True,
        running=True,
        healthy=True,
        state=active_state,
    )
    service, registry, results, manager = _service(tmp_path, status=status)
    try:
        registry.register_artifact(_manifest(catalog.get_artifact("artifact-a")))
        response = service.models()

        assert response.model_count == 2
        assert response.artifact_count == 2
        assert response.deployment_count == 2
        assert response.installed_artifact_count == 1
        assert response.active_deployment_id == "deployment-a"
        model_a, model_b = response.models
        assert model_a.id == "model-a"
        assert model_a.installed is True and model_a.active is True
        assert model_a.license.name == "Apache-2.0"
        assert model_a.artifacts[0].installation == ArtifactInstallation.INSTALLED
        assert model_a.artifacts[0].installed_size_bytes == 7
        assert model_a.deployments[0].available is True
        assert model_a.deployments[0].active is True
        assert model_a.deployments[0].ready is True
        assert model_a.deployments[0].multimodal is True
        assert model_b.installed is False and model_b.active is False
        assert model_b.artifacts[0].installation == ArtifactInstallation.MISSING
        assert model_b.deployments[0].available is False
        assert manager.health_checks == [False]
        _assert_redacted(response)
    finally:
        results.close()
        registry.close()


def test_models_mark_registered_catalog_drift_unavailable(tmp_path: Path) -> None:
    service, registry, results, _ = _service(tmp_path)
    try:
        spec = service.catalog.get_artifact("artifact-a")
        registry.register_artifact(
            _manifest(spec, resolved_revision="different-reviewed-revision")
        )

        artifact = service.models().models[0].artifacts[0]
        assert artifact.installation == ArtifactInstallation.CATALOG_MISMATCH
        assert service.models().models[0].deployments[0].available is False
    finally:
        results.close()
        registry.close()


def test_console_dtos_are_closed_and_immutable() -> None:
    artifact = ConsoleArtifact(
        id="artifact",
        model_id="model",
        format="gguf",
        quantization="Q4",
        effective_bpw=4.5,
        installation=ArtifactInstallation.MISSING,
    )
    with pytest.raises(Exception):
        ConsoleArtifact.model_validate(
            {**artifact.model_dump(), "executable": PRIVATE_PATH}
        )
    with pytest.raises(Exception):
        artifact.id = "changed"  # type: ignore[misc]


def test_runtime_inactive_projection(tmp_path: Path) -> None:
    service, registry, results, _ = _service(tmp_path)
    try:
        response = service.runtime_status()
        assert response.condition == RuntimeCondition.INACTIVE
        assert response.active is False
        assert response.lifecycle_phase is None
        assert response.error is None
        _assert_redacted(response)
    finally:
        results.close()
        registry.close()


@pytest.mark.parametrize(
    ("phase", "running", "healthy", "ready", "condition"),
    (
        ("ready", True, True, True, RuntimeCondition.READY),
        ("ready", False, None, False, RuntimeCondition.STALE),
        ("ready", True, False, False, RuntimeCondition.UNHEALTHY),
        ("starting", True, None, False, RuntimeCondition.STARTING),
        ("stopping", True, None, False, RuntimeCondition.STOPPING),
    ),
)
def test_runtime_conditions_are_projected_without_private_state(
    tmp_path: Path,
    phase: str,
    running: bool,
    healthy: bool | None,
    ready: bool,
    condition: RuntimeCondition,
) -> None:
    catalog = _catalog()
    state = _runtime_state(catalog.get_deployment("deployment-a"), phase=phase)
    service, registry, results, _ = _service(
        tmp_path,
        status=RuntimeStatus(
            active=True,
            ready=ready,
            running=running,
            healthy=healthy,
            state=state,
        ),
    )
    try:
        response = service.runtime_status()
        assert response.condition == condition
        assert response.deployment_id == "deployment-a"
        assert response.artifact_id == "artifact-a"
        assert response.public_alias == "local-a"
        assert response.backend == "llama_cpp"
        assert response.activated_at == datetime(
            2026, 9, 6, 12, tzinfo=timezone.utc
        )
        _assert_redacted(response)
    finally:
        results.close()
        registry.close()


def test_runtime_failure_replaces_raw_error_with_stable_public_error(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    raw_error = f"failed at {PRIVATE_PATH} using token={SECRET} via {PRIVATE_URL}"
    state = _runtime_state(
        catalog.get_deployment("deployment-a"), phase="failed", error=raw_error
    )
    service, registry, results, _ = _service(
        tmp_path,
        status=RuntimeStatus(
            active=True,
            ready=False,
            running=False,
            healthy=None,
            state=state,
        ),
    )
    try:
        response = service.runtime_status()
        assert response.condition == RuntimeCondition.FAILED
        assert response.error is not None
        assert response.error.code == "runtime_failed"
        assert raw_error not in response.error.message
        _assert_redacted(response)
    finally:
        results.close()
        registry.close()


def test_runtime_inspection_exception_becomes_sanitized_unavailable_state(
    tmp_path: Path,
) -> None:
    service, registry, results, _ = _service(
        tmp_path,
        status=RuntimeError(
            f"cannot inspect {PRIVATE_PATH}; Authorization: Bearer {SECRET}"
        ),
    )
    try:
        response = service.runtime_status()
        assert response.condition == RuntimeCondition.UNAVAILABLE
        assert response.active is None
        assert response.error is not None
        assert response.error.code == "runtime_status_unavailable"
        _assert_redacted(response)
    finally:
        results.close()
        registry.close()


def test_storage_report_counts_logical_and_deduplicated_bytes_without_path(
    tmp_path: Path,
) -> None:
    service, registry, results, _ = _service(tmp_path)
    try:
        shared_digest = sha256_bytes(b"weights")
        registry.register_artifact(
            _manifest(service.catalog.get_artifact("artifact-a"), digest=shared_digest)
        )
        registry.register_artifact(
            _manifest(service.catalog.get_artifact("artifact-b"), digest=shared_digest)
        )

        response = service.storage_report()
        assert response.filesystem.total_bytes == 1000
        assert response.filesystem.used_bytes == 400
        assert response.filesystem.free_bytes == 600
        assert response.filesystem.used_percent == 40
        assert response.registered.artifact_count == 2
        assert response.registered.logical_bytes == 14
        assert response.registered.unique_blob_bytes == 7
        assert response.registered.deduplicated_bytes == 7
        _assert_redacted(response)
        assert str(service.runtime_manager.paths.data_root) not in response.model_dump_json()
    finally:
        results.close()
        registry.close()


def _index_benchmark(store: ResultsStore, root: Path) -> str:
    run_id = "run-20260906-a"
    suite_definition = {
        "id": "smoke",
        "version": "1.0",
        "cases": [{"id": "case-a"}, {"id": "case-b"}],
    }
    suite_sha = canonical_sha256(suite_definition)
    contract_definition = {
        "method": "POST",
        "endpoint": "/v1/chat/completions",
        "warmup_repetitions": 1,
        "repetitions": 1,
        "requests": [
            {
                "case_id": case_id,
                "body": {
                    "model": "$MODEL_UNDER_TEST",
                    "messages": [{"role": "user", "content": SECRET}],
                },
            }
            for case_id in ("case-a", "case-b")
        ],
    }
    contract_sha = canonical_sha256(contract_definition)
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": "2026-09-06T10:00:00Z",
        "finished_at": "2026-09-06T10:01:00Z",
        "status": "completed",
        "model": {"id": "model-a", "upstream": {"local_path": PRIVATE_PATH}},
        "artifact": {"id": "artifact-a", "path": PRIVATE_PATH},
        "deployment": {
            "id": "deployment-a",
            "executable": PRIVATE_PATH,
            "environment": {"API_KEY": SECRET},
        },
        "suite": {
            "id": "smoke",
            "version": "1.0",
            "sha256": suite_sha,
            "definition": suite_definition,
        },
        "request_contract": {
            "schema_version": 1,
            "sha256": contract_sha,
            "definition": contract_definition,
        },
        "runtime": {
            "base_url": PRIVATE_URL,
            "command": [PRIVATE_PATH, "--api-key", SECRET],
        },
    }
    samples = []
    for repetition, case_id in enumerate(("case-a", "case-b")):
        samples.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "repetition": repetition,
                "started_at": "2026-09-06T10:00:00Z",
                "latency_ms": 10.0 + repetition * 10,
                "request": {
                    "model": "local-a",
                    "messages": [{"role": "user", "content": SECRET}],
                },
                "response": {
                    "model": "local-a",
                    "choices": [{"message": {"content": SECRET}}],
                },
                "output_text": SECRET,
                "tool_names": [],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
                "scores": [],
                "passed": True,
                "error": None,
                "client_completion_tokens_per_second": 10.0 + repetition * 10,
                "server_timings": {
                    "predicted_tokens": 3,
                    "predicted_ms": 100,
                    "predicted_tokens_per_second": 30.0 + repetition * 10,
                },
            }
        )
    summary = {
        "run_id": run_id,
        "sample_count": 2,
        "successful_request_count": 2,
        "error_count": 0,
        "scoreable_sample_count": 2,
        "passed_sample_count": 2,
        "failed_sample_count": 0,
        "pass_rate": 1.0,
        "warmup_request_count": 2,
        "warmup_error_count": 0,
        "latency_ms": {"count": 2, "mean": 15, "p50": 15, "p95": 19.5},
        "performance": {
            "client_completion_tokens_per_second": {
                "count": 2,
                "mean": 15,
                "p50": 15,
                "p95": 19.5,
            },
            "server_predicted_tokens_per_second": {
                "count": 2,
                "mean": 35,
                "p50": 35,
                "p95": 39.5,
            },
        },
        "private_error": f"{PRIVATE_PATH} {SECRET}",
    }
    telemetry = [
        {
            "timestamp": "2026-09-06T10:00:00Z",
            "available": True,
            "index": 0,
            "uuid": SECRET,
            "name": "NVIDIA RTX 4090",
            "gpu_utilization_percent": 80,
            "memory_used_mib": 18000,
            "memory_total_mib": 24564,
            "power_draw_w": 350,
        },
        {
            "timestamp": "2026-09-06T10:00:01Z",
            "available": True,
            "index": 0,
            "uuid": SECRET,
            "name": "NVIDIA RTX 4090",
            "gpu_utilization_percent": 99,
            "memory_used_mib": 19000,
            "memory_total_mib": 24564,
            "power_draw_w": 390,
        },
        {
            "timestamp": "2026-09-06T10:00:02Z",
            "available": False,
            "error": f"telemetry failed at {PRIVATE_PATH}: {SECRET}",
        },
    ]
    bundle = write_run_bundle(
        root / "private-bundle",
        run,
        summary=summary,
        samples=samples,
        telemetry=telemetry,
    )
    store.append_bundle(bundle)
    return run_id


def test_benchmark_summaries_and_details_are_metric_only_and_redacted(
    tmp_path: Path,
) -> None:
    service, registry, results, _ = _service(tmp_path)
    try:
        run_id = _index_benchmark(results, tmp_path)

        listing = service.benchmark_runs(limit=10)
        assert listing.count == 1
        summary = listing.runs[0]
        assert summary.run_id == run_id
        assert summary.model_id == "model-a"
        assert summary.status == "completed"
        assert summary.sample_count == 2
        assert summary.error_count == 0
        assert summary.pass_rate == 1
        assert summary.metrics.latency_ms.mean == 15
        assert summary.metrics.client_completion_tokens_per_second.mean == 15
        assert summary.metrics.server_predicted_tokens_per_second.mean == 35
        assert summary.suite_sha256 is not None
        assert summary.request_contract_sha256 is not None

        detail = service.benchmark_run(run_id)
        assert [item.case_id for item in detail.cases] == ["case-a", "case-b"]
        assert detail.cases[0].prompt_tokens == 2
        assert detail.cases[0].completion_tokens == 3
        assert detail.cases[0].metrics.latency_ms.mean == 10
        assert (
            detail.cases[1].metrics.server_predicted_tokens_per_second.mean
            == 40
        )
        assert detail.telemetry.sample_count == 3
        assert detail.telemetry.available_sample_count == 2
        assert detail.telemetry.gpu_name == "NVIDIA RTX 4090"
        assert detail.telemetry.peak_gpu_utilization_percent == 99
        assert detail.telemetry.peak_memory_used_mib == 19000
        assert detail.telemetry.memory_total_mib == 24564
        assert detail.telemetry.peak_power_draw_w == 390

        _assert_redacted(listing)
        _assert_redacted(detail)
    finally:
        results.close()
        registry.close()


def test_benchmark_queries_validate_limits_and_identifiers(tmp_path: Path) -> None:
    service, registry, results, _ = _service(tmp_path)
    try:
        with pytest.raises(ValueError, match="between 1 and 500"):
            service.benchmark_runs(limit=0)
        with pytest.raises(ValueError, match="between 1 and 500"):
            service.benchmark_runs(limit=501)
        with pytest.raises(CatalogError, match="invalid public identifier"):
            service.benchmark_run("../../private/results")
        with pytest.raises(CatalogError, match="unknown benchmark run"):
            service.benchmark_run("unknown-run")
    finally:
        results.close()
        registry.close()


def test_malformed_private_benchmark_metadata_is_not_reflected(
    tmp_path: Path,
) -> None:
    service, registry, results, _ = _service(tmp_path)
    try:
        run_id = _index_benchmark(results, tmp_path)
        results.connection.execute(
            """
            UPDATE runs
            SET status = ?, model_id = ?, run_json = ?, summary_json = ?
            WHERE run_id = ?
            """,
            [
                f"failed: token={SECRET}",
                PRIVATE_PATH,
                json.dumps({"runtime": {"api_key": SECRET}}),
                json.dumps({"error": f"{PRIVATE_PATH}: {SECRET}"}),
                run_id,
            ],
        )

        response = service.benchmark_runs()
        run = response.runs[0]
        assert run.status == "unknown"
        assert run.model_id is None
        assert run.suite_sha256 is None
        assert run.request_contract_sha256 is None
        assert run.metrics.latency_ms.count == 0
        _assert_redacted(response)
    finally:
        results.close()
        registry.close()
