"""Strict schemas for catalog entries and immutable records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BackendKind(StrEnum):
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    SGLANG = "sglang"
    TENSORRT_LLM = "tensorrt_llm"
    EXTERNAL = "external"
    MOCK = "mock"


class ArtifactFormat(StrEnum):
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    AWQ = "awq"
    GPTQ = "gptq"
    EXL3 = "exl3"
    MXFP4 = "mxfp4"
    OTHER = "other"


class FileRole(StrEnum):
    WEIGHTS = "weights"
    SHARD_INDEX = "shard_index"
    CONFIG = "config"
    TOKENIZER = "tokenizer"
    TEMPLATE = "template"
    LICENSE = "license"
    MODEL_CARD = "model_card"
    VISION_PROJECTOR = "vision_projector"
    DRAFT_MODEL = "draft_model"
    ADAPTER = "adapter"
    CUSTOM_CODE = "custom_code"
    OTHER = "other"


class LicenseSpec(StrictModel):
    name: str
    url: str | None = None
    commercial_use: bool = True
    osi_approved: bool = False
    acceptance_required: bool = False
    notes: str | None = None


class RepositorySource(StrictModel):
    provider: Literal["huggingface", "local"] = "huggingface"
    repo_id: str | None = None
    repo_type: Literal["model", "dataset", "space"] = "model"
    revision: str = "main"
    local_path: str | None = None

    @model_validator(mode="after")
    def validate_location(self) -> "RepositorySource":
        if self.provider == "huggingface" and not self.repo_id:
            raise ValueError("Hugging Face sources require repo_id")
        if self.provider == "local" and not self.local_path:
            raise ValueError("local sources require local_path")
        return self


class ModelSpec(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    display_name: str
    family: str
    description: str
    total_params_b: float = Field(gt=0)
    active_params_b: float = Field(gt=0)
    native_context: int = Field(gt=0)
    modalities: tuple[str, ...] = ("text",)
    capabilities: tuple[str, ...] = ()
    license: LicenseSpec
    upstream: RepositorySource

    @model_validator(mode="after")
    def active_not_larger_than_total(self) -> "ModelSpec":
        if self.active_params_b > self.total_params_b:
            raise ValueError("active_params_b cannot exceed total_params_b")
        return self


class ArtifactFileSelector(StrictModel):
    pattern: str
    role: FileRole
    required: bool = True
    expected_size_bytes: int | None = Field(default=None, gt=0)
    expected_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class ArtifactSpec(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    model_id: str
    source: RepositorySource
    format: ArtifactFormat
    quantization: str
    effective_bpw: float | None = Field(default=None, gt=0)
    expected_size_bytes: int | None = Field(default=None, gt=0)
    files: tuple[ArtifactFileSelector, ...]
    requires_remote_code: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def require_files(self) -> "ArtifactSpec":
        if not self.files:
            raise ValueError("an artifact must select at least one file")
        return self


class RuntimeImage(StrictModel):
    reference: str
    digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("reference")
    @classmethod
    def reference_must_be_a_single_oci_name(cls, value: str) -> str:
        base = value.rsplit("@", 1)[0]
        if (
            not value
            or value.startswith("-")
            or any(character.isspace() or ord(character) < 32 for character in value)
            or value.count("@") > 1
            or "://" in value
            or not base
            or not base[0].isalnum()
            or any(
                not (character.isalnum() or character in "._:/-")
                for character in base
            )
        ):
            raise ValueError("runtime image reference is not a safe OCI image name")
        return value

    @model_validator(mode="after")
    def embedded_digest_must_match_lock(self) -> "RuntimeImage":
        if "@" not in self.reference:
            return self
        embedded = self.reference.rsplit("@", 1)[1]
        if not embedded.startswith("sha256:") or len(embedded) != 71 or any(
            character not in "0123456789abcdef" for character in embedded[7:]
        ):
            raise ValueError("runtime image reference contains an invalid digest")
        if self.digest is not None and embedded != self.digest:
            raise ValueError(
                "runtime image reference digest does not match the reviewed digest"
            )
        return self


class RuntimeLockSpec(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    source: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    build: dict[str, Any]
    binary: str
    binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version_contains: str
    verified: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source")
    @classmethod
    def source_must_be_an_unambiguous_https_git_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or value.startswith("-")
            or any(character.isspace() or ord(character) < 32 for character in value)
        ):
            raise ValueError(
                "runtime-lock source must be an unambiguous credential-free HTTPS URL"
            )
        return value

    @field_validator("binary")
    @classmethod
    def binary_must_be_a_safe_relative_path(cls, value: str) -> str:
        candidate = PurePosixPath(value)
        if (
            candidate.is_absolute()
            or len(candidate.parts) < 4
            or candidate.parts[:2] != ("cache", "runtimes")
            or ".." in candidate.parts
            or any(part in {"", "."} for part in candidate.parts)
        ):
            raise ValueError(
                "runtime-lock binary must be a safe path below cache/runtimes"
            )
        return value

    @field_validator("build", "verified")
    @classmethod
    def mappings_must_not_be_empty_when_required(
        cls, value: dict[str, Any], info: Any
    ) -> dict[str, Any]:
        if info.field_name == "build" and not value:
            raise ValueError("runtime-lock build recipe cannot be empty")
        return value

    @field_validator("version_contains")
    @classmethod
    def version_evidence_must_be_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("runtime-lock version_contains cannot be empty")
        return value


class DeploymentSpec(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    artifact_id: str
    public_alias: str
    backend: BackendKind
    image: RuntimeImage | None = None
    executable: str | None = None
    external_base_url: str | None = None
    runtime_lock_id: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    context_size: int = Field(default=8192, gt=0)
    parallel: int = Field(default=1, gt=0)
    gpu_layers: str | int = "all"
    flash_attention: bool = True
    kv_cache_type_k: str = "q8_0"
    kv_cache_type_v: str = "q8_0"
    reasoning_mode: Literal["off", "on", "auto"] = "auto"
    speculative_mode: Literal["none", "mtp"] = "none"
    speculative_draft_tokens: int = Field(default=2, ge=1, le=16)
    health_path: str = "/health"
    startup_timeout_seconds: float = Field(default=300.0, gt=0)
    mmproj: str | None = None
    lora_adapter: str | None = None
    extra_args: tuple[str, ...] = ()
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_backend_location(self) -> "DeploymentSpec":
        if self.backend == BackendKind.EXTERNAL and not self.external_base_url:
            raise ValueError("external deployments require external_base_url")
        if self.backend not in {BackendKind.EXTERNAL, BackendKind.MOCK}:
            if not self.image and not self.executable:
                raise ValueError("a local backend requires image or executable")
        if self.mmproj is not None and self.backend != BackendKind.LLAMA_CPP:
            raise ValueError("mmproj is supported only by llama.cpp deployments")
        if self.lora_adapter is not None and self.backend != BackendKind.LLAMA_CPP:
            raise ValueError("lora_adapter is supported only by llama.cpp deployments")
        if self.speculative_mode != "none" and self.backend != BackendKind.LLAMA_CPP:
            raise ValueError(
                "speculative decoding is supported only by llama.cpp deployments"
            )
        if self.runtime_lock_id is not None and (
            self.image is not None
            or self.backend in {BackendKind.EXTERNAL, BackendKind.MOCK}
        ):
            raise ValueError(
                "runtime_lock_id is valid only for a non-image host backend"
            )
        if self.backend == BackendKind.EXTERNAL and (
            self.image is not None or self.executable is not None or self.extra_args
        ):
            raise ValueError("external deployments cannot define local launch fields")
        return self

    @field_validator("mmproj", "lora_adapter")
    @classmethod
    def mmproj_must_be_inside_artifact_view(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = PurePosixPath(value)
        if (
            not candidate.is_absolute()
            or len(candidate.parts) < 3
            or candidate.parts[1] != "models"
            or ".." in candidate.parts
        ):
            raise ValueError("mmproj/adapter must be an absolute path below /models")
        return value

    @field_validator("health_path")
    @classmethod
    def health_path_must_be_origin_relative(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            not value.startswith("/")
            or value.startswith("//")
            or parsed.scheme
            or parsed.netloc
            or "\\" in value
        ):
            raise ValueError("health_path must be a single-slash origin-relative path")
        return value


class LockedFile(StrictModel):
    logical_path: str
    role: FileRole
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    storage_uri: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    required: bool = True


class ArtifactManifest(StrictModel):
    schema_version: Literal[1] = 1
    artifact_id: str
    model_id: str
    source: RepositorySource
    resolved_revision: str
    format: ArtifactFormat
    quantization: str
    effective_bpw: float | None = None
    created_at: datetime
    files: tuple[LockedFile, ...]
    total_logical_bytes: int = Field(ge=0)
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


class GenerationConfig(StrictModel):
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    max_tokens: int = Field(default=128, gt=0)
    seed: int = 42


class Expectation(StrictModel):
    kind: Literal[
        "exact",
        "contains",
        "regex",
        "json_schema",
        "tool_name",
        "nonempty",
    ]
    value: Any = None
    case_sensitive: bool = False


class BenchmarkCase(StrictModel):
    id: str
    messages: tuple[ChatMessage, ...]
    generation: GenerationConfig | None = None
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    expectations: tuple[Expectation, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @field_validator("request_overrides")
    @classmethod
    def protect_request_identity(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = {
            "model",
            "messages",
            "temperature",
            "top_p",
            "max_tokens",
            "seed",
            "stream",
        }
        conflicts = sorted(reserved.intersection(value))
        if conflicts:
            raise ValueError(
                "request_overrides cannot replace identity, workload, or "
                f"typed generation fields: {conflicts}"
            )
        return value


class BenchmarkSuite(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: str
    description: str
    kind: Literal["smoke", "quality", "performance", "capability", "golden"]
    defaults: GenerationConfig = GenerationConfig()
    warmup_repetitions: int = Field(default=0, ge=0)
    repetitions: int = Field(default=1, gt=0)
    cases: tuple[BenchmarkCase, ...]

    @model_validator(mode="after")
    def unique_case_ids(self) -> "BenchmarkSuite":
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark case ids must be unique")
        if not ids:
            raise ValueError("a benchmark suite requires at least one case")
        return self
