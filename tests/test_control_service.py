from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest

from llm_lab.catalog import Catalog
from llm_lab.control_service import ControlService, ControlServiceError
from llm_lab.errors import CatalogError, DeploymentError
from llm_lab.paths import LabPaths
from llm_lab.runtime import LaunchPlan, LaunchRecord, RuntimeState, RuntimeStatus
from llm_lab.schema import (
    ArtifactFileSelector,
    ArtifactFormat,
    ArtifactManifest,
    ArtifactSpec,
    BenchmarkCase,
    BenchmarkSuite,
    ChatMessage,
    DeploymentSpec,
    FileRole,
    LicenseSpec,
    LockedFile,
    ModelSpec,
    RepositorySource,
)
from llm_lab.storage import VerificationReport


_PAYLOAD = b"weights"
_PRIVATE_SOURCE = "/private/catalog/source"
_PRIVATE_LOG = "/private/runtime/model.log"
_PRIVATE_SECRET = "do-not-return-this-token"


def _paths(tmp_path: Path) -> LabPaths:
    repo = tmp_path / "repo"
    repo.mkdir()
    return LabPaths.discover(repo_root=repo, data_root=tmp_path / "data")


def _catalog(*, port: int = 18089, suite_version: str = "1.0.0") -> Catalog:
    source = RepositorySource(
        provider="local", local_path=_PRIVATE_SOURCE, revision="fixture-v1"
    )
    model = ModelSpec(
        id="model-a",
        display_name="Model A",
        family="fixture",
        description="fixture model",
        total_params_b=1,
        active_params_b=1,
        native_context=4096,
        capabilities=("chat",),
        license=LicenseSpec(name="fixture"),
        upstream=source,
    )
    artifact = ArtifactFileSelector(pattern="model.gguf", role=FileRole.WEIGHTS)
    artifact_model = ArtifactSpec(
        id="artifact-a",
        model_id=model.id,
        source=source,
        format=ArtifactFormat.GGUF,
        quantization="fixture",
        expected_size_bytes=len(_PAYLOAD),
        files=(artifact,),
    )
    deployment = DeploymentSpec(
        id="deployment-a",
        artifact_id=artifact_model.id,
        public_alias="local-a",
        backend="mock",
        port=port,
        environment={"PRIVATE_TOKEN": _PRIVATE_SECRET},
        startup_timeout_seconds=1,
    )
    other = DeploymentSpec(
        id="deployment-b",
        artifact_id=artifact_model.id,
        public_alias="local-b",
        backend="mock",
        port=port + 1,
        startup_timeout_seconds=1,
    )
    suite = BenchmarkSuite(
        id="smoke",
        version=suite_version,
        description="fixture smoke suite",
        kind="smoke",
        cases=(
            BenchmarkCase(
                id="chat",
                messages=(ChatMessage(role="user", content="ready?"),),
            ),
        ),
    )
    return Catalog(
        models={model.id: model},
        artifacts={artifact_model.id: artifact_model},
        deployments={deployment.id: deployment, other.id: other},
        suites={suite.id: suite},
        runtime_locks={},
    )


def _manifest(catalog: Catalog) -> ArtifactManifest:
    spec = catalog.get_artifact("artifact-a")
    return ArtifactManifest(
        artifact_id=spec.id,
        model_id=spec.model_id,
        source=spec.source,
        resolved_revision=spec.source.revision,
        format=spec.format,
        quantization=spec.quantization,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        files=(
            LockedFile(
                logical_path="model.gguf",
                role=FileRole.WEIGHTS,
                size_bytes=len(_PAYLOAD),
                sha256="a" * 64,
                storage_uri=f"sha256:{'a' * 64}",
            ),
        ),
        total_logical_bytes=len(_PAYLOAD),
        tree_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )


class FakeRegistry:
    def __init__(self, manifest: ArtifactManifest | None) -> None:
        self.manifest = manifest

    def find_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        if self.manifest is None or self.manifest.artifact_id != artifact_id:
            return None
        return self.manifest


class FakeArtifactStore:
    def __init__(
        self,
        manifest: ArtifactManifest | None,
        *,
        verification_error: Exception | None = None,
    ) -> None:
        self.registry = FakeRegistry(manifest)
        self.manifest = manifest
        self.verification_error = verification_error
        self.verify_calls: list[tuple[ArtifactManifest, bool]] = []

    def __enter__(self) -> "FakeArtifactStore":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def verify(
        self, artifact: ArtifactManifest, *, verify_view: bool = True
    ) -> VerificationReport:
        self.verify_calls.append((artifact, verify_view))
        if self.verification_error is not None:
            raise self.verification_error
        return VerificationReport(
            artifact_id=artifact.artifact_id,
            manifest_sha256=artifact.manifest_sha256 or "",
            file_count=len(artifact.files),
            total_logical_bytes=artifact.total_logical_bytes,
            view_verified=verify_view,
        )


def _state(
    deployment: DeploymentSpec,
    *,
    activated_at: str = "2026-01-01T00:00:00+00:00",
    phase: str = "ready",
) -> RuntimeState:
    return RuntimeState(
        schema_version=1,
        phase=phase,  # type: ignore[arg-type]
        deployment=deployment,
        artifact_path="/private/views/artifact-a",
        plan=LaunchPlan(
            kind="process",
            deployment_id=deployment.id,
            command=("/private/bin/backend", "--token", _PRIVATE_SECRET),
            environment={"PRIVATE_TOKEN": _PRIVATE_SECRET},
            base_url=f"http://127.0.0.1:{deployment.port}",
            health_url=f"http://127.0.0.1:{deployment.port}/health",
            log_path=_PRIVATE_LOG,
        ),
        launch=LaunchRecord(
            kind="process",
            command=("/private/bin/backend", "--token", _PRIVATE_SECRET),
            started_at=activated_at,
            pid=4001,
            process_create_time=123.5,
            resolved_executable="/private/bin/backend",
            executable_sha256="d" * 64,
            executable_version="private build details",
        ),
        activated_at=activated_at,
        error=None,
    )


def _status(state: RuntimeState | None, *, ready: bool | None = None) -> RuntimeStatus:
    is_ready = state is not None and state.phase == "ready" if ready is None else ready
    return RuntimeStatus(
        active=state is not None,
        ready=is_ready,
        running=state is not None,
        healthy=True if state is not None else None,
        state=state,
    )


class FakeRuntimeManager:
    def __init__(
        self,
        status: RuntimeStatus,
        *,
        activation_error: Exception | None = None,
        stop_error: Exception | None = None,
    ) -> None:
        self.current = status
        self.activation_error = activation_error
        self.stop_error = stop_error
        self.activate_calls: list[tuple[DeploymentSpec, Path, Any]] = []
        self.stop_calls = 0
        self.lease_entries = 0

    def status(self, *, check_health: bool = True) -> RuntimeStatus:
        return self.current

    def activate(
        self,
        deployment: DeploymentSpec,
        artifact_path: Path,
        *,
        runtime_lock: Any = None,
    ) -> RuntimeState:
        self.activate_calls.append((deployment, artifact_path, runtime_lock))
        if self.activation_error is not None:
            raise self.activation_error
        state = _state(deployment, activated_at="2026-01-02T00:00:00+00:00")
        self.current = _status(state)
        return state

    def stop(self) -> RuntimeState | None:
        self.stop_calls += 1
        if self.stop_error is not None:
            raise self.stop_error
        state = self.current.state
        self.current = _status(None)
        return state

    @contextmanager
    def benchmark_lease(
        self, *, check_health: bool = True
    ) -> Iterator[RuntimeStatus]:
        self.lease_entries += 1
        yield self.current


def _service(
    tmp_path: Path,
    catalog_loader: Any,
    runtime: FakeRuntimeManager,
    store: FakeArtifactStore,
) -> ControlService:
    return ControlService(
        _paths(tmp_path),
        catalog_loader=catalog_loader,
        runtime_manager=runtime,
        artifact_store_factory=lambda: store,
        transition_lock_factory=nullcontext,
    )


def _serialized(value: Any) -> str:
    return json.dumps(value.to_dict(), sort_keys=True)


def test_artifact_and_deployment_preflight_are_catalog_bound_and_redacted(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    store = FakeArtifactStore(_manifest(catalog))
    runtime = FakeRuntimeManager(_status(None))
    service = _service(tmp_path, lambda: catalog, runtime, store)

    artifact = service.artifact_preflight("artifact-a")
    deployment = service.deployment_preflight("deployment-a")

    assert artifact.verified and artifact.manifest_sha256 == "c" * 64
    assert deployment.artifact == artifact
    assert deployment.runtime_revision.startswith("rt1-")
    assert deployment.catalog_revision.startswith("cat1-")
    assert deployment.revision.startswith("pre1-")
    assert deployment.etag == f'"{deployment.revision}"'
    assert store.verify_calls and all(call[1] for call in store.verify_calls)
    rendered = _serialized(deployment)
    assert _PRIVATE_SOURCE not in rendered
    assert _PRIVATE_LOG not in rendered
    assert _PRIVATE_SECRET not in rendered
    assert "command" not in rendered and "environment" not in rendered


@pytest.mark.parametrize(
    ("method", "identifier", "code"),
    [
        ("artifact", "missing-artifact", "unknown_artifact"),
        ("deployment", "missing-deployment", "unknown_deployment"),
        ("deployment", "../../private", "invalid_catalog_id"),
    ],
)
def test_only_catalog_ids_are_accepted(
    tmp_path: Path, method: str, identifier: str, code: str
) -> None:
    catalog = _catalog()
    service = _service(
        tmp_path,
        lambda: catalog,
        FakeRuntimeManager(_status(None)),
        FakeArtifactStore(_manifest(catalog)),
    )

    with pytest.raises(ControlServiceError) as caught:
        if method == "artifact":
            service.artifact_preflight(identifier)
        else:
            service.deployment_preflight(identifier)

    assert caught.value.code == code
    assert identifier not in json.dumps(caught.value.to_dict())


def test_missing_registered_artifact_has_a_sanitized_error(tmp_path: Path) -> None:
    catalog = _catalog()
    service = _service(
        tmp_path,
        lambda: catalog,
        FakeRuntimeManager(_status(None)),
        FakeArtifactStore(None),
    )

    with pytest.raises(ControlServiceError) as caught:
        service.deployment_preflight("deployment-a")

    assert caught.value.code == "artifact_not_installed"
    rendered = json.dumps(caught.value.to_dict())
    assert _PRIVATE_SOURCE not in rendered
    assert "missing artifact" not in rendered


def test_activation_rejects_a_stale_runtime_revision(tmp_path: Path) -> None:
    catalog = _catalog()
    runtime = FakeRuntimeManager(_status(None))
    service = _service(
        tmp_path, lambda: catalog, runtime, FakeArtifactStore(_manifest(catalog))
    )
    preflight = service.deployment_preflight("deployment-a")
    runtime.current = _status(_state(catalog.get_deployment("deployment-b")))

    with pytest.raises(ControlServiceError) as caught:
        service.activate(
            "deployment-a",
            expected_runtime_revision=preflight.runtime_revision,
        )

    assert caught.value.code == "stale_runtime"
    assert caught.value.retryable
    assert runtime.activate_calls == []


def test_activation_etag_rejects_catalog_changes_after_preflight(
    tmp_path: Path,
) -> None:
    first = _catalog(port=18089)
    changed = _catalog(port=18099)
    catalogs = iter((first, changed))
    runtime = FakeRuntimeManager(_status(None))
    service = _service(
        tmp_path,
        lambda: next(catalogs),
        runtime,
        FakeArtifactStore(_manifest(first)),
    )
    preflight = service.deployment_preflight("deployment-a")

    with pytest.raises(ControlServiceError) as caught:
        service.activate("deployment-a", if_match=preflight.etag)

    assert caught.value.code == "stale_preflight"
    assert runtime.activate_calls == []


def test_activation_success_calls_runtime_domain_api_and_returns_safe_result(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    runtime = FakeRuntimeManager(_status(None))
    service = _service(
        tmp_path, lambda: catalog, runtime, FakeArtifactStore(_manifest(catalog))
    )
    preflight = service.deployment_preflight("deployment-a")

    result = service.activate("deployment-a", if_match=preflight.etag)

    assert result.changed and not result.idempotent and result.ready
    assert result.deployment_id == "deployment-a"
    assert len(runtime.activate_calls) == 1
    selected, artifact_path, runtime_lock = runtime.activate_calls[0]
    assert selected is catalog.get_deployment("deployment-a")
    assert artifact_path == service.paths.view_root / "artifact-a"
    assert runtime_lock is None
    rendered = _serialized(result)
    assert _PRIVATE_SOURCE not in rendered
    assert _PRIVATE_LOG not in rendered
    assert _PRIVATE_SECRET not in rendered
    assert "/private/" not in rendered


def test_activation_of_same_ready_catalog_target_is_idempotent(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    active = _state(catalog.get_deployment("deployment-a"))
    runtime = FakeRuntimeManager(_status(active))
    service = _service(
        tmp_path, lambda: catalog, runtime, FakeArtifactStore(_manifest(catalog))
    )
    preflight = service.deployment_preflight("deployment-a")

    result = service.activate("deployment-a", if_match=preflight.etag)

    assert not result.changed and result.idempotent and result.ready
    assert runtime.activate_calls == []


def test_activation_propagates_rollback_failure_without_leaking_raw_error(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    runtime = FakeRuntimeManager(
        _status(None),
        activation_error=DeploymentError(
            "activation failed at /private/bin/backend; rollback also failed: "
            f"token={_PRIVATE_SECRET}"
        ),
    )
    service = _service(
        tmp_path, lambda: catalog, runtime, FakeArtifactStore(_manifest(catalog))
    )
    preflight = service.deployment_preflight("deployment-a")

    with pytest.raises(ControlServiceError) as caught:
        service.activate("deployment-a", if_match=preflight.etag)

    error = caught.value
    assert error.code == "activation_rollback_failed"
    assert error.details == {
        "rollback_failed": True,
        "operator_attention_required": True,
    }
    rendered = json.dumps(error.to_dict())
    assert "/private/" not in rendered
    assert _PRIVATE_SECRET not in rendered


def test_stop_uses_runtime_etag_and_is_idempotent_when_already_inactive(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    runtime = FakeRuntimeManager(
        _status(_state(catalog.get_deployment("deployment-a")))
    )
    service = _service(
        tmp_path, lambda: catalog, runtime, FakeArtifactStore(_manifest(catalog))
    )
    before = service.runtime_snapshot(check_health=False)

    stopped = service.stop(if_match=before.etag, deployment_id="deployment-a")
    inactive = service.runtime_snapshot(check_health=False)
    repeated = service.stop(if_match=inactive.etag)

    assert stopped.stopped and not stopped.idempotent
    assert stopped.previous_deployment_id == "deployment-a"
    assert not repeated.stopped and repeated.idempotent
    assert runtime.stop_calls == 1


def test_stop_rejects_stale_revision_and_does_not_signal_runtime(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    runtime = FakeRuntimeManager(
        _status(_state(catalog.get_deployment("deployment-a")))
    )
    service = _service(
        tmp_path, lambda: catalog, runtime, FakeArtifactStore(_manifest(catalog))
    )
    stale = service.runtime_snapshot(check_health=False)
    runtime.current = _status(
        _state(
            catalog.get_deployment("deployment-b"),
            activated_at="2026-02-01T00:00:00+00:00",
        )
    )

    with pytest.raises(ControlServiceError) as caught:
        service.stop(expected_runtime_revision=stale.revision)

    assert caught.value.code == "stale_runtime"
    assert runtime.stop_calls == 0


def test_emergency_stop_without_target_does_not_depend_on_catalog(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    runtime = FakeRuntimeManager(
        _status(_state(catalog.get_deployment("deployment-a")))
    )

    def unavailable_catalog() -> Catalog:
        raise CatalogError(f"catalog unavailable at {_PRIVATE_SOURCE}")

    service = _service(
        tmp_path,
        unavailable_catalog,
        runtime,
        FakeArtifactStore(_manifest(catalog)),
    )
    # The read comes directly from the runtime and is intentionally still
    # available for emergency shutdown when catalog loading is broken.
    snapshot = service.runtime_snapshot(check_health=False)

    result = service.stop(if_match=snapshot.etag)

    assert result.stopped
    assert runtime.stop_calls == 1


def test_benchmark_validation_holds_lease_and_returns_only_catalog_identity(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    runtime = FakeRuntimeManager(
        _status(_state(catalog.get_deployment("deployment-a")))
    )
    service = _service(
        tmp_path, lambda: catalog, runtime, FakeArtifactStore(_manifest(catalog))
    )
    snapshot = service.runtime_snapshot()

    launch = service.validate_benchmark_launch(
        "smoke",
        "deployment-a",
        if_match=snapshot.etag,
        collect_telemetry=False,
    )

    assert launch.suite_id == "smoke"
    assert launch.deployment_id == "deployment-a"
    assert launch.artifact_id == "artifact-a"
    assert launch.collect_telemetry is False
    assert launch.launch_revision.startswith("bench1-")
    assert runtime.lease_entries == 1
    rendered = _serialized(launch)
    assert "base_url" not in rendered and "endpoint" not in rendered
    assert _PRIVATE_SOURCE not in rendered
    assert _PRIVATE_LOG not in rendered
    assert _PRIVATE_SECRET not in rendered


def test_benchmark_validation_rejects_wrong_active_deployment(tmp_path: Path) -> None:
    catalog = _catalog()
    runtime = FakeRuntimeManager(
        _status(_state(catalog.get_deployment("deployment-b")))
    )
    service = _service(
        tmp_path, lambda: catalog, runtime, FakeArtifactStore(_manifest(catalog))
    )
    snapshot = service.runtime_snapshot()

    with pytest.raises(ControlServiceError) as caught:
        service.validate_benchmark_launch(
            "smoke", "deployment-a", if_match=snapshot.etag
        )

    assert caught.value.code == "deployment_not_active"


def test_benchmark_rejects_stale_catalog_revision(tmp_path: Path) -> None:
    first = _catalog(suite_version="1.0.0")
    changed = _catalog(suite_version="2.0.0")
    active = _state(changed.get_deployment("deployment-a"))
    runtime = FakeRuntimeManager(_status(active))
    service = _service(
        tmp_path,
        lambda: changed,
        runtime,
        FakeArtifactStore(_manifest(changed)),
    )
    stale_catalog_revision = ControlService._catalog_revision(first)
    snapshot = service.runtime_snapshot()

    with pytest.raises(ControlServiceError) as caught:
        service.validate_benchmark_launch(
            "smoke",
            "deployment-a",
            if_match=snapshot.etag,
            expected_catalog_revision=stale_catalog_revision,
        )

    assert caught.value.code == "stale_catalog"


def test_preconditions_are_mandatory_for_all_mutating_or_launch_operations(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    runtime = FakeRuntimeManager(_status(None))
    service = _service(
        tmp_path, lambda: catalog, runtime, FakeArtifactStore(_manifest(catalog))
    )

    calls = (
        lambda: service.activate("deployment-a"),
        lambda: service.stop(),
        lambda: service.validate_benchmark_launch("smoke", "deployment-a"),
    )
    for call in calls:
        with pytest.raises(ControlServiceError) as caught:
            call()
        assert caught.value.code == "precondition_required"


def test_artifact_catalog_mismatch_is_rejected_without_field_details(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    changed_manifest = _manifest(catalog).model_copy(
        update={"resolved_revision": f"secret-{_PRIVATE_SECRET}"}
    )
    # A local source's friendly revision is not immutable, so use a mismatch
    # that is always bound by the control service.
    changed_manifest = changed_manifest.model_copy(update={"quantization": "other"})
    service = _service(
        tmp_path,
        lambda: catalog,
        FakeRuntimeManager(_status(None)),
        FakeArtifactStore(changed_manifest),
    )

    with pytest.raises(ControlServiceError) as caught:
        service.artifact_preflight("artifact-a")

    assert caught.value.code == "artifact_catalog_mismatch"
    rendered = json.dumps(caught.value.to_dict())
    assert "quantization" not in rendered
    assert _PRIVATE_SECRET not in rendered
