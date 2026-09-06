from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm_lab.schema import (
    ArtifactSpec,
    BenchmarkCase,
    BenchmarkSuite,
    ChatMessage,
    DeploymentSpec,
    ModelSpec,
    RepositorySource,
    RuntimeImage,
    RuntimeLockSpec,
)


@pytest.mark.parametrize(
    "source",
    (
        "-c core.sshCommand=evil",
        "file:///tmp/unreviewed",
        "ssh://example.invalid/repo.git",
        "https://user:secret@example.invalid/repo.git",
        "https://example.invalid/repo.git?ref=mutable",
    ),
)
def test_runtime_lock_rejects_ambiguous_or_credentialed_source(source: str) -> None:
    with pytest.raises(ValueError, match="credential-free HTTPS"):
        RuntimeLockSpec(
            id="locked-runtime",
            source=source,
            commit="a" * 40,
            build={"generator": "fixture"},
            binary="cache/runtimes/runtime/server",
            binary_sha256="b" * 64,
            version_contains="fixture",
        )


@pytest.mark.parametrize(
    "binary",
    (
        "/tmp/unreviewed/server",
        "../cache/runtimes/runtime/server",
        "cache/llama.cpp/build/bin/llama-server",
        "cache/runtimes/server",
    ),
)
def test_runtime_lock_binary_is_confined_to_dedicated_runtime_namespace(
    binary: str,
) -> None:
    with pytest.raises(ValueError, match="below cache/runtimes"):
        RuntimeLockSpec(
            id="locked-runtime",
            source="https://example.invalid/runtime.git",
            commit="a" * 40,
            build={"generator": "fixture"},
            binary=binary,
            binary_sha256="b" * 64,
            version_contains="fixture",
        )


@pytest.mark.parametrize(
    ("folder", "schema"),
    [
        ("models", ModelSpec),
        ("artifacts", ArtifactSpec),
        ("deployments", DeploymentSpec),
        ("suites", BenchmarkSuite),
        ("runtime-locks", RuntimeLockSpec),
    ],
)
def test_checked_in_catalog_entries_validate(folder: str, schema: type) -> None:
    paths = sorted((Path("catalog") / folder).glob("*.yaml"))
    assert paths, f"expected catalog/{folder} fixtures"
    for path in paths:
        schema.model_validate(yaml.safe_load(path.read_text()))


def test_model_rejects_active_parameters_above_total() -> None:
    payload = yaml.safe_load(Path("catalog/models/ling-3.0-tiny.yaml").read_text())
    payload["active_params_b"] = payload["total_params_b"] + 1

    with pytest.raises(ValidationError, match="active_params_b"):
        ModelSpec.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"provider": "huggingface", "revision": "main"},
        {"provider": "local", "revision": "main"},
    ],
)
def test_repository_source_requires_provider_location(payload: dict) -> None:
    with pytest.raises(ValidationError):
        RepositorySource.model_validate(payload)


def test_external_deployment_requires_base_url() -> None:
    payload = yaml.safe_load(Path("catalog/deployments/mock-canary.yaml").read_text())
    payload["backend"] = "external"
    payload.pop("executable", None)

    with pytest.raises(ValidationError, match="external_base_url"):
        DeploymentSpec.model_validate(payload)


def test_runtime_image_cannot_bypass_a_separate_reviewed_digest() -> None:
    first = "sha256:" + "a" * 64
    second = "sha256:" + "b" * 64

    with pytest.raises(ValidationError, match="does not match"):
        RuntimeImage(reference=f"example/image@{first}", digest=second)


@pytest.mark.parametrize(
    "reference",
    (
        "-v/etc:/models",
        " example/image:latest",
        "https://example.invalid/image:latest",
        "example/image@sha256:" + "a" * 64 + "@sha256:" + "b" * 64,
    ),
)
def test_runtime_image_rejects_option_like_or_malformed_references(
    reference: str,
) -> None:
    with pytest.raises(ValidationError, match="safe OCI image name"):
        RuntimeImage(reference=reference)


def test_deployment_cannot_combine_container_image_with_host_runtime_lock() -> None:
    with pytest.raises(ValidationError, match="non-image host backend"):
        DeploymentSpec(
            id="ambiguous-runtime",
            artifact_id="artifact",
            public_alias="ambiguous-runtime",
            backend="llama_cpp",
            image=RuntimeImage(reference="example.invalid/llama:server"),
            runtime_lock_id="host-lock",
        )


@pytest.mark.parametrize(
    "health_path",
    (
        "https://attacker.invalid/health",
        "//attacker.invalid/health",
        "\\\\attacker.invalid\\health",
    ),
)
def test_health_path_cannot_redirect_readiness_to_another_origin(
    health_path: str,
) -> None:
    payload = yaml.safe_load(Path("catalog/deployments/mock-canary.yaml").read_text())
    payload["health_path"] = health_path

    with pytest.raises(ValidationError, match="origin-relative"):
        DeploymentSpec.model_validate(payload)


def test_suite_rejects_duplicate_case_ids() -> None:
    payload = yaml.safe_load(Path("catalog/suites/smoke.yaml").read_text())
    payload["cases"].append(dict(payload["cases"][0]))

    with pytest.raises(ValidationError, match="case ids"):
        BenchmarkSuite.model_validate(payload)


@pytest.mark.parametrize("suite_id", ("../escape", "/absolute", "space name", "UPPER"))
def test_suite_id_is_safe_for_immutable_run_paths(suite_id: str) -> None:
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        BenchmarkSuite(
            id=suite_id,
            version="1",
            description="unsafe id",
            kind="smoke",
            cases=(
                BenchmarkCase(
                    id="case",
                    messages=(ChatMessage(role="user", content="hello"),),
                ),
            ),
        )


def test_schema_forbids_unknown_fields() -> None:
    payload = yaml.safe_load(Path("catalog/models/qwen3.8-27b.yaml").read_text())
    payload["untracked_setting"] = True

    with pytest.raises(ValidationError, match="untracked_setting"):
        ModelSpec.model_validate(payload)
