"""Public, deliberately redacted schemas for the LLM Lab Console.

These models form a disclosure boundary.  They intentionally do not mirror the
catalog, runtime state, or benchmark bundle schemas: those internal records can
contain filesystem paths, commands, environment variables, endpoint URLs, and
model input/output.  Console responses are assembled field by field instead.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConsoleDTO(BaseModel):
    """Base class for immutable, closed public response objects."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ArtifactInstallation(StrEnum):
    MISSING = "missing"
    INSTALLED = "installed"
    CATALOG_MISMATCH = "catalog_mismatch"


class RuntimeCondition(StrEnum):
    INACTIVE = "inactive"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"
    STALE = "stale"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"


class ConsoleError(ConsoleDTO):
    """A stable public error, never an exception or persisted error string."""

    code: Literal["runtime_failed", "runtime_status_unavailable"]
    message: str


class ConsoleLicense(ConsoleDTO):
    name: str
    commercial_use: bool
    osi_approved: bool
    acceptance_required: bool


class ConsoleArtifact(ConsoleDTO):
    id: str
    model_id: str
    format: str
    quantization: str
    effective_bpw: float | None
    expected_size_bytes: int | None = Field(default=None, ge=0)
    installation: ArtifactInstallation
    installed_size_bytes: int | None = Field(default=None, ge=0)
    manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class ConsoleDeployment(ConsoleDTO):
    id: str
    artifact_id: str
    public_alias: str
    backend: str
    context_size: int = Field(gt=0)
    parallel: int = Field(gt=0)
    reasoning_mode: str
    multimodal: bool
    available: bool
    active: bool
    ready: bool


class ConsoleModel(ConsoleDTO):
    id: str
    display_name: str
    family: str
    description: str
    total_params_b: float = Field(gt=0)
    active_params_b: float = Field(gt=0)
    native_context: int = Field(gt=0)
    modalities: tuple[str, ...]
    capabilities: tuple[str, ...]
    license: ConsoleLicense
    installed: bool
    active: bool
    artifacts: tuple[ConsoleArtifact, ...]
    deployments: tuple[ConsoleDeployment, ...]


class ModelCatalogResponse(ConsoleDTO):
    models: tuple[ConsoleModel, ...]
    model_count: int = Field(ge=0)
    artifact_count: int = Field(ge=0)
    deployment_count: int = Field(ge=0)
    installed_artifact_count: int = Field(ge=0)
    active_deployment_id: str | None = None


class RuntimeStatusResponse(ConsoleDTO):
    condition: RuntimeCondition
    active: bool | None
    ready: bool | None
    running: bool | None
    healthy: bool | None
    lifecycle_phase: Literal["starting", "ready", "stopping", "failed"] | None
    deployment_id: str | None = None
    artifact_id: str | None = None
    public_alias: str | None = None
    backend: str | None = None
    activated_at: datetime | None = None
    error: ConsoleError | None = None


class StorageFilesystem(ConsoleDTO):
    total_bytes: int = Field(ge=0)
    used_bytes: int = Field(ge=0)
    free_bytes: int = Field(ge=0)
    used_percent: float = Field(ge=0, le=100)


class StorageRegistered(ConsoleDTO):
    artifact_count: int = Field(ge=0)
    logical_bytes: int = Field(ge=0)
    unique_blob_bytes: int = Field(ge=0)
    deduplicated_bytes: int = Field(ge=0)


class StorageReportResponse(ConsoleDTO):
    filesystem: StorageFilesystem
    registered: StorageRegistered


class MetricDistribution(ConsoleDTO):
    count: int = Field(ge=0)
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None


class BenchmarkRunMetrics(ConsoleDTO):
    latency_ms: MetricDistribution
    client_completion_tokens_per_second: MetricDistribution
    server_predicted_tokens_per_second: MetricDistribution


class BenchmarkRunSummary(ConsoleDTO):
    run_id: str
    started_at: datetime | None
    finished_at: datetime | None
    status: Literal[
        "running",
        "completed",
        "completed_with_errors",
        "failed",
        "cancelled",
        "unknown",
    ]
    model_id: str | None
    artifact_id: str | None
    deployment_id: str | None
    suite_id: str | None
    suite_version: str | None
    suite_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    request_contract_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    sample_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    warmup_error_count: int = Field(ge=0)
    scoreable_sample_count: int = Field(ge=0)
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    metrics: BenchmarkRunMetrics


class BenchmarkRunsResponse(ConsoleDTO):
    runs: tuple[BenchmarkRunSummary, ...]
    count: int = Field(ge=0)


class BenchmarkCaseSummary(ConsoleDTO):
    case_id: str
    sample_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    scoreable_sample_count: int = Field(ge=0)
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    metrics: BenchmarkRunMetrics


class BenchmarkTelemetrySummary(ConsoleDTO):
    sample_count: int = Field(ge=0)
    available_sample_count: int = Field(ge=0)
    gpu_name: str | None
    peak_gpu_utilization_percent: float | None = Field(
        default=None, ge=0, le=100
    )
    peak_memory_used_mib: float | None = Field(default=None, ge=0)
    memory_total_mib: float | None = Field(default=None, ge=0)
    peak_power_draw_w: float | None = Field(default=None, ge=0)


class BenchmarkRunDetail(BenchmarkRunSummary):
    cases: tuple[BenchmarkCaseSummary, ...]
    telemetry: BenchmarkTelemetrySummary


__all__ = [
    "ArtifactInstallation",
    "BenchmarkCaseSummary",
    "BenchmarkRunDetail",
    "BenchmarkRunMetrics",
    "BenchmarkRunsResponse",
    "BenchmarkRunSummary",
    "BenchmarkTelemetrySummary",
    "ConsoleArtifact",
    "ConsoleDeployment",
    "ConsoleError",
    "ConsoleLicense",
    "ConsoleModel",
    "MetricDistribution",
    "ModelCatalogResponse",
    "RuntimeCondition",
    "RuntimeStatusResponse",
    "StorageFilesystem",
    "StorageRegistered",
    "StorageReportResponse",
]
