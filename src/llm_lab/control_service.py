"""Catalog-bound application services for privileged control operations.

The CLI and HTTP console both need to perform the same checks before changing
the model runtime.  This module is deliberately transport-agnostic: callers
provide catalog IDs and optimistic-concurrency tokens, while the service calls
the existing domain objects directly.  It never executes a command shell and
never returns runtime paths, commands, environment variables, or raw domain
errors.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol

from .catalog import Catalog
from .errors import LabError
from .hashing import canonical_sha256
from .paths import LabPaths
from .runtime import (
    ExclusiveGpuLock,
    RuntimeLockVerification,
    RuntimeManager,
    RuntimeState,
    RuntimeStatus,
    verify_runtime_lock,
)
from .schema import (
    ArtifactManifest,
    ArtifactSpec,
    BenchmarkSuite,
    DeploymentSpec,
    RuntimeLockSpec,
)
from .storage import ArtifactStore, VerificationReport, artifact_path_matches


_REVISION_SCHEMA = 1


class _RuntimeManager(Protocol):
    def status(self, *, check_health: bool = True) -> RuntimeStatus: ...

    def activate(
        self,
        deployment: DeploymentSpec,
        artifact_path: str | Path | None = None,
        *,
        runtime_lock: RuntimeLockSpec | None = None,
    ) -> RuntimeState: ...

    def stop(self) -> RuntimeState | None: ...

    def benchmark_lease(
        self, *, check_health: bool = True
    ) -> AbstractContextManager[RuntimeStatus]: ...


class _ArtifactRegistry(Protocol):
    def find_artifact(self, artifact_id: str) -> ArtifactManifest | None: ...


class _ArtifactStore(Protocol):
    registry: _ArtifactRegistry

    def __enter__(self) -> "_ArtifactStore": ...

    def __exit__(self, *_: object) -> None: ...

    def verify(
        self,
        artifact: str | ArtifactManifest | Mapping[str, Any],
        *,
        verify_view: bool = True,
    ) -> VerificationReport: ...


CatalogLoader = Callable[[], Catalog]
ArtifactStoreFactory = Callable[[], _ArtifactStore]
RuntimeLockVerifier = Callable[
    [LabPaths, DeploymentSpec, RuntimeLockSpec, str | Path | None],
    RuntimeLockVerification,
]
TransitionLockFactory = Callable[[], AbstractContextManager[Any]]


class ControlServiceError(LabError):
    """A stable, safe error suitable for a CLI or HTTP response.

    The original exception remains available through Python exception chaining
    for trusted local diagnostics, but it is intentionally not retained in the
    serialized representation.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, str | bool | int | None] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.retryable = retryable
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.public_message,
                "retryable": self.retryable,
                "details": dict(self.details),
            }
        }


@dataclass(frozen=True, slots=True)
class ArtifactPreflight:
    artifact_id: str
    model_id: str
    manifest_sha256: str
    file_count: int
    total_logical_bytes: int
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    revision: str
    etag: str
    active: bool
    ready: bool
    running: bool
    healthy: bool | None
    deployment_id: str | None
    public_alias: str | None
    phase: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeploymentPreflight:
    revision: str
    etag: str
    catalog_revision: str
    runtime_revision: str
    runtime_etag: str
    deployment_id: str
    public_alias: str
    model_id: str
    artifact: ArtifactPreflight
    runtime_lock_id: str | None
    already_active: bool
    ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActivationResult:
    deployment_id: str
    public_alias: str
    artifact_id: str
    changed: bool
    idempotent: bool
    ready: bool
    phase: str
    runtime_revision: str
    etag: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StopResult:
    stopped: bool
    idempotent: bool
    previous_deployment_id: str | None
    runtime_revision: str
    etag: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkLaunchInput:
    suite_id: str
    suite_version: str
    deployment_id: str
    public_alias: str
    model_id: str
    artifact_id: str
    collect_telemetry: bool
    runtime_revision: str
    runtime_etag: str
    catalog_revision: str
    launch_revision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PreparedDeployment:
    catalog: Catalog
    catalog_revision: str
    deployment: DeploymentSpec
    artifact_spec: ArtifactSpec
    artifact_manifest: ArtifactManifest
    artifact: ArtifactPreflight
    runtime_lock: RuntimeLockSpec | None
    runtime_verification: RuntimeLockVerification | None
    status: RuntimeStatus
    runtime: RuntimeSnapshot
    result: DeploymentPreflight


def etag_for_revision(revision: str) -> str:
    """Return a strong HTTP ETag for one service revision."""

    if not isinstance(revision, str) or not revision or any(
        character in revision for character in ('"', "\r", "\n")
    ):
        raise ValueError("revision cannot be represented as a strong ETag")
    return f'"{revision}"'


class ControlService:
    """Safe application boundary shared by interactive control surfaces."""

    def __init__(
        self,
        paths: LabPaths | None = None,
        *,
        catalog_loader: CatalogLoader | None = None,
        artifact_store_factory: ArtifactStoreFactory | None = None,
        runtime_manager: _RuntimeManager | None = None,
        runtime_lock_verifier: RuntimeLockVerifier = verify_runtime_lock,
        transition_lock_factory: TransitionLockFactory | None = None,
    ) -> None:
        self.paths = paths or LabPaths.discover()
        self._catalog_loader = catalog_loader or (
            lambda: Catalog.load(self.paths.catalog_root)
        )
        self._artifact_store_factory = artifact_store_factory or (
            lambda: ArtifactStore(self.paths)
        )
        self._runtime_manager = runtime_manager or RuntimeManager(self.paths)
        self._runtime_lock_verifier = runtime_lock_verifier
        self._transition_lock_factory = transition_lock_factory or (
            lambda: ExclusiveGpuLock(
                self.paths.data_root / "state/control-service.lock",
                timeout_seconds=30.0,
            )
        )

    def artifact_preflight(self, artifact_id: str) -> ArtifactPreflight:
        """Verify one catalog artifact and its materialized view."""

        self._require_id(artifact_id, "artifact")
        with self._transition_guard():
            catalog = self._load_catalog()
            return self._artifact_preflight(catalog, artifact_id)[2]

    def runtime_snapshot(self, *, check_health: bool = True) -> RuntimeSnapshot:
        """Return a redacted runtime view and its optimistic-concurrency token."""

        if not isinstance(check_health, bool):
            raise ControlServiceError(
                "invalid_request", "Runtime health selection must be a boolean."
            )
        with self._transition_guard():
            try:
                status = self._runtime_manager.status(check_health=check_health)
            except Exception as exc:
                raise ControlServiceError(
                    "runtime_unavailable",
                    "Runtime status is unavailable.",
                    retryable=True,
                ) from exc
            return self._runtime_snapshot(status)

    def deployment_preflight(self, deployment_id: str) -> DeploymentPreflight:
        """Verify a catalog deployment and bind it to current runtime state.

        The returned ETag covers the catalog, verified artifact manifest, target
        deployment, runtime lock, and current runtime revision.  Passing it as
        ``if_match`` to :meth:`activate` detects changes after confirmation.
        """

        self._require_id(deployment_id, "deployment")
        with self._transition_guard():
            catalog = self._load_catalog()
            return self._prepare_deployment(catalog, deployment_id).result

    # A concise alias reads naturally in both CLI and HTTP adapters.
    preflight_deployment = deployment_preflight
    preflight_activation = deployment_preflight

    def activate(
        self,
        deployment_id: str,
        *,
        if_match: str | None = None,
        expected_runtime_revision: str | None = None,
        expected_catalog_revision: str | None = None,
    ) -> ActivationResult:
        """Activate a catalog deployment after optimistic-concurrency checks.

        ``if_match`` is the preferred precondition and accepts the ETag returned
        by :meth:`deployment_preflight`.  The explicit runtime/catalog revision
        arguments support non-HTTP callers.  At least ``if_match`` or
        ``expected_runtime_revision`` is required.
        """

        self._require_id(deployment_id, "deployment")
        if if_match is None and expected_runtime_revision is None:
            raise ControlServiceError(
                "precondition_required",
                "Activation requires a current deployment preflight revision.",
            )

        with self._transition_guard():
            catalog = self._load_catalog()
            prepared = self._prepare_deployment(catalog, deployment_id)
            if if_match is not None:
                self._require_revision(
                    if_match,
                    prepared.result.revision,
                    code="stale_preflight",
                    message="Deployment preflight is stale; review it again.",
                )
            if expected_runtime_revision is not None:
                self._require_revision(
                    expected_runtime_revision,
                    prepared.runtime.revision,
                    code="stale_runtime",
                    message="Runtime state changed; refresh before activating.",
                )
            if expected_catalog_revision is not None:
                self._require_revision(
                    expected_catalog_revision,
                    prepared.catalog_revision,
                    code="stale_catalog",
                    message="Catalog content changed; review the deployment again.",
                )

            current = prepared.status.state
            if (
                current is not None
                and prepared.status.ready
                and current.deployment.id == deployment_id
                and current.deployment == prepared.deployment
            ):
                return self._activation_result(current, changed=False, idempotent=True)

            try:
                state = self._runtime_manager.activate(
                    prepared.deployment,
                    self.paths.view_root / prepared.deployment.artifact_id,
                    runtime_lock=prepared.runtime_lock,
                )
            except Exception as exc:
                self._raise_activation_error(exc)

            if state.deployment != prepared.deployment or state.phase != "ready":
                raise ControlServiceError(
                    "activation_failed",
                    "Deployment activation did not publish the expected ready state.",
                    retryable=True,
                )
            return self._activation_result(state, changed=True, idempotent=False)

    def stop(
        self,
        *,
        if_match: str | None = None,
        expected_runtime_revision: str | None = None,
        deployment_id: str | None = None,
    ) -> StopResult:
        """Stop the active runtime after a runtime revision/ETag precondition."""

        if deployment_id is not None:
            self._require_id(deployment_id, "deployment")
        if if_match is None and expected_runtime_revision is None:
            raise ControlServiceError(
                "precondition_required",
                "Stopping requires a current runtime revision.",
            )

        with self._transition_guard():
            if deployment_id is not None:
                catalog = self._load_catalog()
                self._catalog_deployment(catalog, deployment_id)
            try:
                status = self._runtime_manager.status(check_health=False)
            except Exception as exc:
                raise ControlServiceError(
                    "runtime_unavailable",
                    "Runtime status is unavailable.",
                    retryable=True,
                ) from exc
            snapshot = self._runtime_snapshot(status)
            if if_match is not None:
                self._require_revision(
                    if_match,
                    snapshot.revision,
                    code="stale_runtime",
                    message="Runtime state changed; refresh before stopping.",
                )
            if expected_runtime_revision is not None:
                self._require_revision(
                    expected_runtime_revision,
                    snapshot.revision,
                    code="stale_runtime",
                    message="Runtime state changed; refresh before stopping.",
                )

            active = status.state
            if active is None:
                return StopResult(
                    stopped=False,
                    idempotent=True,
                    previous_deployment_id=None,
                    runtime_revision=snapshot.revision,
                    etag=snapshot.etag,
                )
            if deployment_id is not None and active.deployment.id != deployment_id:
                raise ControlServiceError(
                    "deployment_not_active",
                    "The requested deployment is not the active deployment.",
                )
            try:
                stopped = self._runtime_manager.stop()
            except Exception as exc:
                raise ControlServiceError(
                    "stop_failed",
                    "The active deployment could not be stopped safely.",
                    retryable=True,
                    details={"operator_attention_required": True},
                ) from exc
            if stopped is None:
                raise ControlServiceError(
                    "runtime_changed",
                    "Runtime state changed while the stop was being applied.",
                    retryable=True,
                )
            inactive = self._runtime_snapshot(_inactive_status())
            return StopResult(
                stopped=True,
                idempotent=False,
                previous_deployment_id=active.deployment.id,
                runtime_revision=inactive.revision,
                etag=inactive.etag,
            )

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
        """Validate benchmark input while retaining the runtime benchmark lease.

        Callers that immediately execute a benchmark should keep their work
        inside this context so activation/stop cannot interleave.  The yielded
        object is deliberately sanitized and contains no endpoint or secret.
        """

        self._require_id(suite_id, "benchmark suite")
        self._require_id(deployment_id, "deployment")
        if not isinstance(collect_telemetry, bool):
            raise ControlServiceError(
                "invalid_request", "Benchmark telemetry selection must be a boolean."
            )
        if expected_runtime_revision is None and if_match is None:
            raise ControlServiceError(
                "precondition_required",
                "Benchmark launch requires a current runtime revision.",
            )

        with self._transition_guard():
            catalog = self._load_catalog()
            suite = self._catalog_suite(catalog, suite_id)
            try:
                lease = self._runtime_manager.benchmark_lease(check_health=True)
                with lease as status:
                    prepared = self._prepare_deployment(
                        catalog, deployment_id, status=status
                    )
                    if expected_runtime_revision is not None:
                        self._require_revision(
                            expected_runtime_revision,
                            prepared.runtime.revision,
                            code="stale_runtime",
                            message=(
                                "Runtime state changed; refresh before benchmarking."
                            ),
                        )
                    if if_match is not None:
                        # Benchmark ETags are runtime ETags; deployment-preflight
                        # ETags intentionally have a different revision prefix.
                        self._require_revision(
                            if_match,
                            prepared.runtime.revision,
                            code="stale_runtime",
                            message=(
                                "Runtime state changed; refresh before benchmarking."
                            ),
                        )
                    if expected_catalog_revision is not None:
                        self._require_revision(
                            expected_catalog_revision,
                            prepared.catalog_revision,
                            code="stale_catalog",
                            message=(
                                "Catalog content changed; review the benchmark again."
                            ),
                        )
                    self._validate_active_benchmark_runtime(prepared)
                    launch_revision = "bench1-" + canonical_sha256(
                        {
                            "schema_version": _REVISION_SCHEMA,
                            "suite": suite.model_dump(mode="json"),
                            "deployment_revision": prepared.result.revision,
                            "collect_telemetry": collect_telemetry,
                        }
                    )
                    yield BenchmarkLaunchInput(
                        suite_id=suite.id,
                        suite_version=suite.version,
                        deployment_id=prepared.deployment.id,
                        public_alias=prepared.deployment.public_alias,
                        model_id=prepared.artifact_spec.model_id,
                        artifact_id=prepared.artifact_spec.id,
                        collect_telemetry=collect_telemetry,
                        runtime_revision=prepared.runtime.revision,
                        runtime_etag=prepared.runtime.etag,
                        catalog_revision=prepared.catalog_revision,
                        launch_revision=launch_revision,
                    )
            except ControlServiceError:
                raise
            except Exception as exc:
                raise ControlServiceError(
                    "benchmark_unavailable",
                    "Benchmark launch validation failed.",
                    retryable=True,
                ) from exc

    def validate_benchmark_launch(
        self,
        suite_id: str,
        deployment_id: str,
        *,
        expected_runtime_revision: str | None = None,
        if_match: str | None = None,
        expected_catalog_revision: str | None = None,
        collect_telemetry: bool = True,
    ) -> BenchmarkLaunchInput:
        """Validate and return sanitized benchmark launch input."""

        with self.benchmark_launch(
            suite_id,
            deployment_id,
            expected_runtime_revision=expected_runtime_revision,
            if_match=if_match,
            expected_catalog_revision=expected_catalog_revision,
            collect_telemetry=collect_telemetry,
        ) as launch:
            return launch

    # Transport adapters can use the longer operation-oriented names without
    # introducing wrappers that might accidentally omit a precondition.
    activate_deployment = activate
    stop_deployment = stop
    validate_benchmark = validate_benchmark_launch

    def _prepare_deployment(
        self,
        catalog: Catalog,
        deployment_id: str,
        *,
        status: RuntimeStatus | None = None,
    ) -> _PreparedDeployment:
        deployment = self._catalog_deployment(catalog, deployment_id)
        artifact_spec, manifest, artifact = self._artifact_preflight(
            catalog, deployment.artifact_id
        )
        runtime_lock: RuntimeLockSpec | None = None
        runtime_verification: RuntimeLockVerification | None = None
        if deployment.runtime_lock_id is not None:
            try:
                runtime_lock = catalog.get_runtime_lock(deployment.runtime_lock_id)
                runtime_verification = self._runtime_lock_verifier(
                    self.paths,
                    deployment,
                    runtime_lock,
                    self.paths.view_root / deployment.artifact_id,
                )
            except Exception as exc:
                raise ControlServiceError(
                    "runtime_verification_failed",
                    "The reviewed runtime could not be verified.",
                ) from exc

        if status is None:
            try:
                status = self._runtime_manager.status(check_health=True)
            except Exception as exc:
                raise ControlServiceError(
                    "runtime_unavailable",
                    "Runtime status is unavailable.",
                    retryable=True,
                ) from exc
        runtime = self._runtime_snapshot(status)
        catalog_revision = self._catalog_revision(catalog)
        revision = "pre1-" + canonical_sha256(
            {
                "schema_version": _REVISION_SCHEMA,
                "catalog_revision": catalog_revision,
                "runtime_revision": runtime.revision,
                "deployment": deployment.model_dump(mode="json"),
                "artifact_manifest_sha256": artifact.manifest_sha256,
                "runtime_lock": (
                    None
                    if runtime_lock is None
                    else runtime_lock.model_dump(mode="json")
                ),
            }
        )
        already_active = (
            status.state is not None
            and status.state.deployment.id == deployment.id
            and status.state.deployment == deployment
        )
        result = DeploymentPreflight(
            revision=revision,
            etag=etag_for_revision(revision),
            catalog_revision=catalog_revision,
            runtime_revision=runtime.revision,
            runtime_etag=runtime.etag,
            deployment_id=deployment.id,
            public_alias=deployment.public_alias,
            model_id=artifact_spec.model_id,
            artifact=artifact,
            runtime_lock_id=deployment.runtime_lock_id,
            already_active=already_active,
            ready=already_active and status.ready,
        )
        return _PreparedDeployment(
            catalog=catalog,
            catalog_revision=catalog_revision,
            deployment=deployment,
            artifact_spec=artifact_spec,
            artifact_manifest=manifest,
            artifact=artifact,
            runtime_lock=runtime_lock,
            runtime_verification=runtime_verification,
            status=status,
            runtime=runtime,
            result=result,
        )

    def _artifact_preflight(
        self, catalog: Catalog, artifact_id: str
    ) -> tuple[ArtifactSpec, ArtifactManifest, ArtifactPreflight]:
        try:
            spec = catalog.get_artifact(artifact_id)
        except Exception as exc:
            raise ControlServiceError(
                "unknown_artifact", "Unknown catalog artifact."
            ) from exc

        try:
            with self._artifact_store_factory() as store:
                manifest = store.registry.find_artifact(spec.id)
                if manifest is None:
                    raise ControlServiceError(
                        "artifact_not_installed",
                        "The selected artifact is not installed.",
                    )
                _assert_artifact_matches_catalog(spec, manifest)
                report = store.verify(manifest, verify_view=True)
        except ControlServiceError:
            raise
        except Exception as exc:
            raise ControlServiceError(
                "artifact_verification_failed",
                "The selected artifact failed integrity verification.",
            ) from exc

        if not manifest.manifest_sha256 or not report.view_verified:
            raise ControlServiceError(
                "artifact_verification_failed",
                "The selected artifact failed integrity verification.",
            )
        result = ArtifactPreflight(
            artifact_id=spec.id,
            model_id=spec.model_id,
            manifest_sha256=manifest.manifest_sha256,
            file_count=report.file_count,
            total_logical_bytes=report.total_logical_bytes,
        )
        return spec, manifest, result

    def _validate_active_benchmark_runtime(
        self, prepared: _PreparedDeployment
    ) -> None:
        active = prepared.status.state
        if active is None or not prepared.status.ready:
            raise ControlServiceError(
                "runtime_not_ready",
                "A ready active deployment is required for benchmarking.",
                retryable=True,
            )
        if active.deployment.id != prepared.deployment.id:
            raise ControlServiceError(
                "deployment_not_active",
                "The selected deployment is not the ready active deployment.",
            )
        if active.deployment != prepared.deployment:
            raise ControlServiceError(
                "stale_active_deployment",
                "The active deployment was launched from stale catalog content.",
            )
        verification = prepared.runtime_verification
        if verification is not None and (
            active.launch.resolved_executable != verification.binary
            or active.launch.executable_sha256 != verification.binary_sha256
            or active.launch.executable_version != verification.executable_version
        ):
            raise ControlServiceError(
                "runtime_identity_mismatch",
                "The active runtime does not match its reviewed lock.",
            )

    def _activation_result(
        self, state: RuntimeState, *, changed: bool, idempotent: bool
    ) -> ActivationResult:
        revision = _runtime_state_revision(state)
        return ActivationResult(
            deployment_id=state.deployment.id,
            public_alias=state.deployment.public_alias,
            artifact_id=state.deployment.artifact_id,
            changed=changed,
            idempotent=idempotent,
            ready=state.phase == "ready",
            phase=state.phase,
            runtime_revision=revision,
            etag=etag_for_revision(revision),
        )

    def _runtime_snapshot(self, status: RuntimeStatus) -> RuntimeSnapshot:
        revision = _runtime_state_revision(status.state)
        state = status.state
        return RuntimeSnapshot(
            revision=revision,
            etag=etag_for_revision(revision),
            active=state is not None,
            ready=status.ready,
            running=status.running,
            healthy=status.healthy,
            deployment_id=None if state is None else state.deployment.id,
            public_alias=None if state is None else state.deployment.public_alias,
            phase=None if state is None else state.phase,
        )

    def _load_catalog(self) -> Catalog:
        try:
            catalog = self._catalog_loader()
        except Exception as exc:
            raise ControlServiceError(
                "catalog_unavailable", "The catalog could not be loaded."
            ) from exc
        if not isinstance(catalog, Catalog):
            raise ControlServiceError(
                "catalog_unavailable", "The catalog could not be loaded."
            )
        return catalog

    @staticmethod
    def _catalog_revision(catalog: Catalog) -> str:
        payload = {
            name: [
                entries[key].model_dump(mode="json") for key in sorted(entries)
            ]
            for name, entries in (
                ("models", catalog.models),
                ("artifacts", catalog.artifacts),
                ("deployments", catalog.deployments),
                ("suites", catalog.suites),
                ("runtime_locks", catalog.runtime_locks),
            )
        }
        return "cat1-" + canonical_sha256(
            {"schema_version": _REVISION_SCHEMA, "catalog": payload}
        )

    @staticmethod
    def _catalog_deployment(catalog: Catalog, deployment_id: str) -> DeploymentSpec:
        try:
            return catalog.get_deployment(deployment_id)
        except Exception as exc:
            raise ControlServiceError(
                "unknown_deployment", "Unknown catalog deployment."
            ) from exc

    @staticmethod
    def _catalog_suite(catalog: Catalog, suite_id: str) -> BenchmarkSuite:
        try:
            return catalog.get_suite(suite_id)
        except Exception as exc:
            raise ControlServiceError(
                "unknown_suite", "Unknown catalog benchmark suite."
            ) from exc

    @staticmethod
    def _require_id(value: str, kind: str) -> None:
        if (
            not isinstance(value, str)
            or not value
            or not ("a" <= value[0] <= "z" or "0" <= value[0] <= "9")
            or any(
                not (
                    "a" <= character <= "z"
                    or "0" <= character <= "9"
                    or character in ".-_"
                )
                for character in value
            )
        ):
            raise ControlServiceError(
                "invalid_catalog_id", f"A valid catalog {kind} ID is required."
            )

    @staticmethod
    def _require_revision(
        supplied: str,
        current: str,
        *,
        code: str,
        message: str,
    ) -> None:
        if _unquote_etag(supplied) != current:
            raise ControlServiceError(
                code,
                message,
                retryable=True,
                details={
                    "current_revision": current,
                    "current_etag": etag_for_revision(current),
                },
            )

    @contextmanager
    def _transition_guard(self) -> Iterator[None]:
        try:
            self.paths.initialize()
            with self._transition_lock_factory():
                yield
        except ControlServiceError:
            raise
        except Exception as exc:
            raise ControlServiceError(
                "control_unavailable",
                "The control plane is temporarily unavailable.",
                retryable=True,
            ) from exc

    @staticmethod
    def _raise_activation_error(exc: Exception) -> NoReturn:
        raw = str(exc).lower()
        rollback_failed = (
            "rollback also failed" in raw or "rollback was not attempted" in raw
        )
        cleanup_uncertain = (
            "may still own" in raw
            or "cleanup could not be confirmed" in raw
            or "detached runtime" in raw
        )
        if rollback_failed or cleanup_uncertain:
            raise ControlServiceError(
                "activation_rollback_failed",
                "Activation failed and runtime ownership requires operator attention.",
                retryable=False,
                details={
                    "rollback_failed": rollback_failed,
                    "operator_attention_required": True,
                },
            ) from exc
        raise ControlServiceError(
            "activation_failed",
            "Deployment activation failed without leaving an untracked runtime.",
            retryable=True,
            details={"rollback_failed": False, "operator_attention_required": False},
        ) from exc


def _assert_artifact_matches_catalog(
    spec: ArtifactSpec, manifest: ArtifactManifest
) -> None:
    """Reject a registered artifact that no longer matches its catalog ID."""

    mismatches: list[str] = []
    for field in ("artifact_id", "model_id", "source", "format", "quantization"):
        expected = spec.id if field == "artifact_id" else getattr(spec, field)
        if getattr(manifest, field) != expected:
            mismatches.append(field)
    if spec.effective_bpw != manifest.effective_bpw:
        mismatches.append("effective_bpw")
    if (
        spec.expected_size_bytes is not None
        and manifest.total_logical_bytes != spec.expected_size_bytes
    ):
        mismatches.append("expected_size_bytes")
    if (
        len(spec.source.revision) >= 40
        and all(character in "0123456789abcdef" for character in spec.source.revision)
        and manifest.resolved_revision != spec.source.revision
    ):
        mismatches.append("resolved_revision")
    for locked in manifest.files:
        selectors = [
            selector
            for selector in spec.files
            if artifact_path_matches(locked.logical_path, selector.pattern)
        ]
        if not selectors or all(selector.role != locked.role for selector in selectors):
            mismatches.append("file_selection")
    for selector in spec.files:
        if selector.required and not any(
            artifact_path_matches(locked.logical_path, selector.pattern)
            and locked.role == selector.role
            for locked in manifest.files
        ):
            mismatches.append("required_selector")
    if mismatches:
        raise ControlServiceError(
            "artifact_catalog_mismatch",
            "The installed artifact no longer matches its catalog definition.",
        )


def _runtime_state_revision(state: RuntimeState | None) -> str:
    if state is None:
        payload: dict[str, Any] = {
            "schema_version": _REVISION_SCHEMA,
            "active": False,
        }
    else:
        payload = {
            "schema_version": _REVISION_SCHEMA,
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


def _unquote_etag(value: str) -> str:
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if candidate.startswith("W/") or "," in candidate or candidate == "*":
        return ""
    if len(candidate) >= 2 and candidate[0] == candidate[-1] == '"':
        candidate = candidate[1:-1]
    if '"' in candidate or "\r" in candidate or "\n" in candidate:
        return ""
    return candidate


def _inactive_status() -> RuntimeStatus:
    return RuntimeStatus(
        active=False,
        ready=False,
        running=False,
        healthy=None,
        state=None,
    )


__all__ = [
    "ActivationResult",
    "ArtifactPreflight",
    "BenchmarkLaunchInput",
    "ControlService",
    "ControlServiceError",
    "DeploymentPreflight",
    "RuntimeSnapshot",
    "StopResult",
    "etag_for_revision",
]
