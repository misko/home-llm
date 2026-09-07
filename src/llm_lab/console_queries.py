"""Read-only projection service for the LLM Lab Console.

The service joins existing domain objects, but never serializes those objects
directly.  In particular, runtime state and benchmark bundle documents are
private records: their commands, paths, environments, URLs, prompts, responses,
and exception text must not cross the console disclosure boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

from .catalog import Catalog
from .console_schema import (
    ArtifactInstallation,
    BenchmarkCaseSummary,
    BenchmarkRunDetail,
    BenchmarkRunMetrics,
    BenchmarkRunsResponse,
    BenchmarkRunSummary,
    BenchmarkTelemetrySummary,
    ConsoleArtifact,
    ConsoleDeployment,
    ConsoleError,
    ConsoleLicense,
    ConsoleModel,
    MetricDistribution,
    ModelCatalogResponse,
    RuntimeCondition,
    RuntimeStatusResponse,
    StorageFilesystem,
    StorageRegistered,
    StorageReportResponse,
)
from .errors import BenchmarkError, CatalogError
from .hashing import canonical_sha256
from .registry import Registry
from .results import ResultsStore
from .runtime import RuntimeManager, RuntimeStatus
from .schema import ArtifactManifest, ArtifactSpec


_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_RUN_STATUSES = {
    "running",
    "completed",
    "completed_with_errors",
    "failed",
    "cancelled",
}


class ConsoleQueryService:
    """Build sanitized console responses from existing read seams.

    ``disk_usage`` is injectable to make capacity behavior deterministic in
    tests.  The default data root comes from the runtime manager, ensuring the
    query and lifecycle services describe the same LLM Lab installation.
    """

    def __init__(
        self,
        catalog: Catalog,
        registry: Registry,
        runtime_manager: RuntimeManager,
        results_store: ResultsStore,
        *,
        disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
    ) -> None:
        self.catalog = catalog
        self.registry = registry
        self.runtime_manager = runtime_manager
        self.results_store = results_store
        self._disk_usage = disk_usage

    def models(self) -> ModelCatalogResponse:
        """Return the joined model/artifact/deployment catalog projection."""

        manifests = {
            manifest.artifact_id: manifest
            for manifest in self.registry.list_artifacts()
        }
        runtime = self.runtime_status(check_health=False)
        active_deployment_id = (
            runtime.deployment_id if runtime.active is True else None
        )
        runtime_ready = runtime.ready is True

        artifacts_by_model: dict[str, list[ConsoleArtifact]] = {
            model_id: [] for model_id in self.catalog.models
        }
        installed_artifact_count = 0
        available_artifacts: set[str] = set()
        for artifact_id in sorted(self.catalog.artifacts):
            spec = self.catalog.artifacts[artifact_id]
            manifest = manifests.get(artifact_id)
            artifact = _artifact_projection(spec, manifest)
            artifacts_by_model[spec.model_id].append(artifact)
            if artifact.installation != ArtifactInstallation.MISSING:
                installed_artifact_count += 1
            if artifact.installation == ArtifactInstallation.INSTALLED:
                available_artifacts.add(artifact.id)

        deployments_by_model: dict[str, list[ConsoleDeployment]] = {
            model_id: [] for model_id in self.catalog.models
        }
        for deployment_id in sorted(self.catalog.deployments):
            deployment = self.catalog.deployments[deployment_id]
            artifact = self.catalog.artifacts[deployment.artifact_id]
            model = self.catalog.models[artifact.model_id]
            active = deployment.id == active_deployment_id
            deployments_by_model[model.id].append(
                ConsoleDeployment(
                    id=deployment.id,
                    artifact_id=deployment.artifact_id,
                    public_alias=deployment.public_alias,
                    backend=deployment.backend.value,
                    context_size=deployment.context_size,
                    parallel=deployment.parallel,
                    reasoning_mode=deployment.reasoning_mode,
                    multimodal=any(
                        modality != "text" for modality in model.modalities
                    ),
                    available=deployment.artifact_id in available_artifacts,
                    active=active,
                    ready=active and runtime_ready,
                )
            )

        models: list[ConsoleModel] = []
        for model_id in sorted(self.catalog.models):
            model = self.catalog.models[model_id]
            artifacts = tuple(artifacts_by_model[model_id])
            deployments = tuple(deployments_by_model[model_id])
            models.append(
                ConsoleModel(
                    id=model.id,
                    display_name=model.display_name,
                    family=model.family,
                    description=model.description,
                    total_params_b=model.total_params_b,
                    active_params_b=model.active_params_b,
                    native_context=model.native_context,
                    modalities=model.modalities,
                    capabilities=model.capabilities,
                    license=ConsoleLicense(
                        name=model.license.name,
                        commercial_use=model.license.commercial_use,
                        osi_approved=model.license.osi_approved,
                        acceptance_required=model.license.acceptance_required,
                    ),
                    installed=any(
                        item.installation == ArtifactInstallation.INSTALLED
                        for item in artifacts
                    ),
                    active=any(item.active for item in deployments),
                    artifacts=artifacts,
                    deployments=deployments,
                )
            )

        return ModelCatalogResponse(
            models=tuple(models),
            model_count=len(models),
            artifact_count=len(self.catalog.artifacts),
            deployment_count=len(self.catalog.deployments),
            installed_artifact_count=installed_artifact_count,
            active_deployment_id=active_deployment_id,
        )

    # Endpoint-oriented alias retained for readability at call sites.
    list_models = models

    def runtime_status(self, *, check_health: bool = True) -> RuntimeStatusResponse:
        """Return runtime state without commands, paths, URLs, or raw errors."""

        try:
            status = self.runtime_manager.status(check_health=check_health)
        except Exception:
            return RuntimeStatusResponse(
                condition=RuntimeCondition.UNAVAILABLE,
                active=None,
                ready=None,
                running=None,
                healthy=None,
                lifecycle_phase=None,
                error=ConsoleError(
                    code="runtime_status_unavailable",
                    message="Runtime status is temporarily unavailable.",
                ),
            )
        return _runtime_projection(status)

    get_runtime_status = runtime_status

    def storage_report(self) -> StorageReportResponse:
        """Return filesystem capacity and registered logical/unique bytes."""

        usage = self._disk_usage(self.runtime_manager.paths.data_root)
        manifests = self.registry.list_artifacts()
        logical_bytes = sum(item.total_logical_bytes for item in manifests)
        digests: dict[str, int] = {}
        for manifest in manifests:
            for item in manifest.files:
                prior_size = digests.setdefault(item.sha256, item.size_bytes)
                if prior_size != item.size_bytes:
                    raise CatalogError(
                        "registered artifact metadata is internally inconsistent"
                    )
        unique_blob_bytes = sum(digests.values())
        total_bytes = max(0, int(usage.total))
        used_bytes = max(0, int(usage.used))
        free_bytes = max(0, int(usage.free))
        used_percent = (
            min(100.0, max(0.0, used_bytes * 100.0 / total_bytes))
            if total_bytes
            else 0.0
        )
        return StorageReportResponse(
            filesystem=StorageFilesystem(
                total_bytes=total_bytes,
                used_bytes=used_bytes,
                free_bytes=free_bytes,
                used_percent=used_percent,
            ),
            registered=StorageRegistered(
                artifact_count=len(manifests),
                logical_bytes=logical_bytes,
                unique_blob_bytes=unique_blob_bytes,
                deduplicated_bytes=logical_bytes - unique_blob_bytes,
            ),
        )

    get_storage_report = storage_report

    def benchmark_runs(self, *, limit: int = 50) -> BenchmarkRunsResponse:
        """Return newest indexed runs without bundle paths or raw documents."""

        if limit <= 0 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        try:
            cursor = self.results_store.connection.execute(
                """
                SELECT run_id, started_at, finished_at, status, model_id,
                       artifact_id, deployment_id, suite_id, suite_version,
                       sample_count, error_count, pass_rate, run_json,
                       summary_json
                FROM runs
                ORDER BY started_at DESC, run_id
                LIMIT ?
                """,
                [limit],
            )
            rows = cursor.fetchall()
        except Exception as exc:
            raise BenchmarkError("benchmark results are temporarily unavailable") from exc
        runs = tuple(_run_summary(row) for row in rows)
        return BenchmarkRunsResponse(runs=runs, count=len(runs))

    list_benchmark_runs = benchmark_runs

    def benchmark_run(self, run_id: str) -> BenchmarkRunDetail:
        """Return one sanitized run plus case and aggregate telemetry metrics."""

        run_id = _required_public_id(run_id, "run")
        try:
            row = self.results_store.connection.execute(
                """
                SELECT run_id, started_at, finished_at, status, model_id,
                       artifact_id, deployment_id, suite_id, suite_version,
                       sample_count, error_count, pass_rate, run_json,
                       summary_json
                FROM runs
                WHERE run_id = ?
                """,
                [run_id],
            ).fetchone()
            if row is None:
                raise CatalogError("unknown benchmark run")
            case_rows = self.results_store.compare_rows([run_id])
            telemetry_rows = self.results_store.connection.execute(
                """
                SELECT available, gpu_name, gpu_utilization_percent,
                       memory_used_mib, memory_total_mib, power_draw_w
                FROM telemetry
                WHERE run_id = ?
                ORDER BY timestamp
                """,
                [run_id],
            ).fetchall()
        except CatalogError:
            raise
        except Exception as exc:
            raise BenchmarkError("benchmark results are temporarily unavailable") from exc

        summary = _run_summary(row)
        return BenchmarkRunDetail(
            **summary.model_dump(mode="python"),
            cases=tuple(_case_summary(item) for item in case_rows),
            telemetry=_telemetry_summary(telemetry_rows),
        )

    get_benchmark_run = benchmark_run


def _artifact_projection(
    spec: ArtifactSpec,
    manifest: ArtifactManifest | None,
) -> ConsoleArtifact:
    if manifest is None:
        installation = ArtifactInstallation.MISSING
        installed_size = None
        manifest_sha256 = None
    else:
        matches = (
            manifest.model_id == spec.model_id
            and manifest.source == spec.source
            and manifest.resolved_revision == spec.source.revision
            and manifest.format == spec.format
            and manifest.quantization == spec.quantization
            and (
                spec.expected_size_bytes is None
                or manifest.total_logical_bytes == spec.expected_size_bytes
            )
        )
        installation = (
            ArtifactInstallation.INSTALLED
            if matches
            else ArtifactInstallation.CATALOG_MISMATCH
        )
        installed_size = manifest.total_logical_bytes
        manifest_sha256 = manifest.manifest_sha256
    return ConsoleArtifact(
        id=spec.id,
        model_id=spec.model_id,
        format=spec.format.value,
        quantization=spec.quantization,
        effective_bpw=spec.effective_bpw,
        expected_size_bytes=spec.expected_size_bytes,
        installation=installation,
        installed_size_bytes=installed_size,
        manifest_sha256=manifest_sha256,
    )


def _runtime_projection(status: RuntimeStatus) -> RuntimeStatusResponse:
    state = status.state
    if not status.active and state is None:
        return RuntimeStatusResponse(
            condition=RuntimeCondition.INACTIVE,
            active=False,
            ready=False,
            running=False,
            healthy=None,
            lifecycle_phase=None,
        )
    if state is None:
        return RuntimeStatusResponse(
            condition=RuntimeCondition.UNAVAILABLE,
            active=None,
            ready=None,
            running=None,
            healthy=None,
            lifecycle_phase=None,
            error=ConsoleError(
                code="runtime_status_unavailable",
                message="Runtime status is temporarily unavailable.",
            ),
        )

    error: ConsoleError | None = None
    if state.phase == "failed":
        condition = RuntimeCondition.FAILED
        error = ConsoleError(
            code="runtime_failed",
            message="The active runtime failed. Inspect operator logs for details.",
        )
    elif not status.running:
        condition = RuntimeCondition.STALE
    elif status.healthy is False:
        condition = RuntimeCondition.UNHEALTHY
    elif state.phase == "ready" and status.ready:
        condition = RuntimeCondition.READY
    elif state.phase == "starting":
        condition = RuntimeCondition.STARTING
    elif state.phase == "stopping":
        condition = RuntimeCondition.STOPPING
    else:
        condition = RuntimeCondition.UNHEALTHY

    return RuntimeStatusResponse(
        condition=condition,
        active=status.active,
        ready=status.ready,
        running=status.running,
        healthy=status.healthy,
        lifecycle_phase=state.phase,
        deployment_id=state.deployment.id,
        artifact_id=state.deployment.artifact_id,
        public_alias=state.deployment.public_alias,
        backend=state.deployment.backend.value,
        activated_at=_parse_datetime(state.activated_at),
        error=error,
    )


def _run_summary(row: Sequence[Any]) -> BenchmarkRunSummary:
    (
        run_id,
        started_at,
        finished_at,
        status,
        model_id,
        artifact_id,
        deployment_id,
        suite_id,
        suite_version,
        sample_count,
        error_count,
        pass_rate,
        run_json,
        summary_json,
    ) = row
    run_id = _required_public_id(run_id, "run")
    summary = _json_mapping(summary_json)
    run = _json_mapping(run_json)
    return BenchmarkRunSummary(
        run_id=run_id,
        started_at=_parse_datetime(started_at),
        finished_at=_parse_datetime(finished_at),
        status=_run_status(status),
        model_id=_optional_public_id(model_id),
        artifact_id=_optional_public_id(artifact_id),
        deployment_id=_optional_public_id(deployment_id),
        suite_id=_optional_public_id(suite_id),
        suite_version=_optional_public_id(suite_version),
        suite_sha256=_verified_nested_digest(run, "suite"),
        request_contract_sha256=_verified_nested_digest(run, "request_contract"),
        sample_count=_nonnegative_int(sample_count),
        error_count=_nonnegative_int(error_count),
        warmup_error_count=_nonnegative_int(summary.get("warmup_error_count")),
        scoreable_sample_count=_nonnegative_int(
            summary.get("scoreable_sample_count")
        ),
        pass_rate=_fraction(pass_rate),
        metrics=_summary_metrics(summary),
    )


def _summary_metrics(summary: Mapping[str, Any]) -> BenchmarkRunMetrics:
    performance = (
        summary.get("performance")
        if isinstance(summary.get("performance"), Mapping)
        else {}
    )
    return BenchmarkRunMetrics(
        latency_ms=_distribution(summary.get("latency_ms")),
        client_completion_tokens_per_second=_distribution(
            performance.get("client_completion_tokens_per_second")
        ),
        server_predicted_tokens_per_second=_distribution(
            performance.get("server_predicted_tokens_per_second")
        ),
    )


def _case_summary(row: Mapping[str, Any]) -> BenchmarkCaseSummary:
    return BenchmarkCaseSummary(
        case_id=_required_public_id(row.get("case_id"), "benchmark case"),
        sample_count=_nonnegative_int(row.get("sample_count")),
        error_count=_nonnegative_int(row.get("error_count")),
        scoreable_sample_count=_nonnegative_int(
            row.get("scoreable_sample_count")
        ),
        pass_rate=_fraction(row.get("pass_rate")),
        prompt_tokens=_optional_nonnegative_int(row.get("prompt_tokens")),
        completion_tokens=_optional_nonnegative_int(
            row.get("completion_tokens")
        ),
        metrics=BenchmarkRunMetrics(
            latency_ms=MetricDistribution(
                count=_nonnegative_int(row.get("sample_count")),
                mean=_nonnegative_float(row.get("mean_latency_ms")),
                p50=_nonnegative_float(row.get("p50_latency_ms")),
                p95=_nonnegative_float(row.get("p95_latency_ms")),
            ),
            client_completion_tokens_per_second=MetricDistribution(
                count=_nonnegative_int(
                    row.get("client_completion_tokens_per_second_sample_count")
                ),
                mean=_nonnegative_float(
                    row.get("mean_client_completion_tokens_per_second")
                ),
                p50=_nonnegative_float(
                    row.get("p50_client_completion_tokens_per_second")
                ),
                p95=_nonnegative_float(
                    row.get("p95_client_completion_tokens_per_second")
                ),
            ),
            server_predicted_tokens_per_second=MetricDistribution(
                count=_nonnegative_int(
                    row.get("server_predicted_tokens_per_second_sample_count")
                ),
                mean=_nonnegative_float(
                    row.get("mean_server_predicted_tokens_per_second")
                ),
                p50=_nonnegative_float(
                    row.get("p50_server_predicted_tokens_per_second")
                ),
                p95=_nonnegative_float(
                    row.get("p95_server_predicted_tokens_per_second")
                ),
            ),
        ),
    )


def _telemetry_summary(rows: Sequence[Sequence[Any]]) -> BenchmarkTelemetrySummary:
    available_rows = [row for row in rows if row[0] is True]
    names = {
        name
        for row in available_rows
        if (name := _public_label(row[1])) is not None
    }
    gpu_name = next(iter(names)) if len(names) == 1 else None
    return BenchmarkTelemetrySummary(
        sample_count=len(rows),
        available_sample_count=len(available_rows),
        gpu_name=gpu_name,
        peak_gpu_utilization_percent=_maximum(rows, 2, upper=100.0),
        peak_memory_used_mib=_maximum(rows, 3),
        memory_total_mib=_maximum(rows, 4),
        peak_power_draw_w=_maximum(rows, 5),
    )


def _maximum(
    rows: Sequence[Sequence[Any]], index: int, *, upper: float | None = None
) -> float | None:
    values = [
        value
        for row in rows
        if row[0] is True
        and (value := _nonnegative_float(row[index])) is not None
        and (upper is None or value <= upper)
    ]
    return max(values) if values else None


def _distribution(value: Any) -> MetricDistribution:
    mapping = value if isinstance(value, Mapping) else {}
    return MetricDistribution(
        count=_nonnegative_int(mapping.get("count")),
        mean=_nonnegative_float(mapping.get("mean")),
        p50=_nonnegative_float(mapping.get("p50")),
        p95=_nonnegative_float(mapping.get("p95")),
    )


def _json_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _verified_nested_digest(value: Mapping[str, Any], key: str) -> str | None:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        return None
    digest = nested.get("sha256")
    definition = nested.get("definition")
    if (
        not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or not isinstance(definition, Mapping)
    ):
        return None
    try:
        return digest if canonical_sha256(definition) == digest else None
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _run_status(value: Any) -> str:
    return value if isinstance(value, str) and value in _KNOWN_RUN_STATUSES else "unknown"


def _required_public_id(value: Any, kind: str) -> str:
    if not isinstance(value, str) or _PUBLIC_ID.fullmatch(value) is None:
        raise CatalogError(f"{kind} has an invalid public identifier")
    return value


def _optional_public_id(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) and _PUBLIC_ID.fullmatch(value) else None


def _public_label(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(ord(character) < 32 for character in value)
        or "/" in value
        or "\\" in value
    ):
        return None
    return value


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, result)


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _nonnegative_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _fraction(value: Any) -> float | None:
    result = _nonnegative_float(value)
    return result if result is not None and result <= 1 else None


__all__ = ["ConsoleQueryService"]
