from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_lab.catalog import Catalog, load_catalog
from llm_lab.errors import CatalogError


def _model(model_id: str = "model-a") -> dict:
    return {
        "schema_version": 1,
        "id": model_id,
        "display_name": "Model A",
        "family": "example",
        "description": "test model",
        "total_params_b": 7.0,
        "active_params_b": 7.0,
        "native_context": 8192,
        "modalities": ["text"],
        "capabilities": ["chat"],
        "license": {"name": "Apache-2.0", "osi_approved": True},
        "upstream": {
            "provider": "huggingface",
            "repo_id": "example/model-a",
            "revision": "a" * 40,
        },
    }


def _artifact(artifact_id: str = "artifact-a", model_id: str = "model-a") -> dict:
    return {
        "schema_version": 1,
        "id": artifact_id,
        "model_id": model_id,
        "source": {
            "provider": "huggingface",
            "repo_id": "example/model-a-gguf",
            "revision": "b" * 40,
        },
        "format": "gguf",
        "quantization": "Q4_K_M",
        "files": [{"pattern": "*.gguf", "role": "weights"}],
    }


def _deployment(
    deployment_id: str = "deployment-a", artifact_id: str = "artifact-a"
) -> dict:
    return {
        "schema_version": 1,
        "id": deployment_id,
        "artifact_id": artifact_id,
        "public_alias": "chat-default",
        "backend": "mock",
    }


def _suite() -> dict:
    return {
        "schema_version": 1,
        "id": "smoke",
        "version": "1",
        "description": "smoke suite",
        "kind": "smoke",
        "cases": [
            {
                "id": "hello",
                "messages": [{"role": "user", "content": "hello"}],
                "expectations": [{"kind": "nonempty"}],
            }
        ],
    }


def _runtime_lock(lock_id: str = "runtime-a") -> dict:
    return {
        "schema_version": 1,
        "id": lock_id,
        "source": "https://example.invalid/runtime.git",
        "commit": "a" * 40,
        "build": {"generator": "cmake"},
        "binary": "cache/runtimes/runtime/server",
        "binary_sha256": "b" * 64,
        "version_contains": "commit aaaaaaa",
    }


def test_load_unified_catalog_and_getters(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": [_model()],
                "artifacts": [_artifact()],
                "deployments": [_deployment()],
                "suites": [_suite()],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    catalog = load_catalog(path)
    assert catalog.get_model("model-a").display_name == "Model A"
    assert catalog.get_artifact("artifact-a").model_id == "model-a"
    assert catalog.get_deployment("deployment-a").artifact_id == "artifact-a"
    assert catalog.get_suite("smoke").cases[0].id == "hello"
    with pytest.raises(CatalogError, match="unknown model"):
        catalog.get_model("missing")


def test_load_directory_and_keyed_aggregate(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models/model.yaml").write_text(
        yaml.safe_dump({"model-a": {k: v for k, v in _model().items() if k != "id"}}),
        encoding="utf-8",
    )
    (tmp_path / "artifacts.yaml").write_text(
        yaml.safe_dump([_artifact()]), encoding="utf-8"
    )

    catalog = Catalog.load(tmp_path)
    assert set(catalog.models) == {"model-a"}
    assert set(catalog.artifacts) == {"artifact-a"}


def test_duplicate_ids_across_files_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    directory.mkdir()
    for name in ("one.yaml", "two.yaml"):
        (directory / name).write_text(yaml.safe_dump(_model()), encoding="utf-8")

    with pytest.raises(CatalogError, match="duplicate model id"):
        Catalog.load(tmp_path)


def test_catalog_rejects_floating_huggingface_revisions(tmp_path: Path) -> None:
    model = _model()
    model["upstream"]["revision"] = "main"
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump({"models": [model]}), encoding="utf-8")

    with pytest.raises(CatalogError, match="immutable Hugging Face commit"):
        Catalog.load(path)


def test_duplicate_yaml_mapping_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text("models: []\nmodels: []\n", encoding="utf-8")
    with pytest.raises(CatalogError, match="duplicate key"):
        Catalog.load(path)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            {"models": [_model()], "artifacts": [_artifact(model_id="missing")]},
            "references unknown model",
        ),
        (
            {
                "models": [_model()],
                "artifacts": [_artifact()],
                "deployments": [_deployment(artifact_id="missing")],
            },
            "references unknown artifact",
        ),
    ],
)
def test_cross_references_are_validated(
    tmp_path: Path, document: dict, message: str
) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(CatalogError, match=message):
        Catalog.load(path)


def test_schema_is_strict_about_unknown_fields(tmp_path: Path) -> None:
    model = _model()
    model["typo_field"] = True
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump({"models": [model]}), encoding="utf-8")
    with pytest.raises(CatalogError, match="extra_forbidden"):
        Catalog.load(path)


def test_host_deployment_requires_and_resolves_runtime_lock(tmp_path: Path) -> None:
    deployment = _deployment()
    deployment.update(
        backend="llama_cpp",
        executable="/opt/llama-server",
        runtime_lock_id="runtime-a",
    )
    document = {
        "models": [_model()],
        "artifacts": [_artifact()],
        "deployments": [deployment],
        "runtime_locks": [_runtime_lock()],
    }
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    catalog = Catalog.load(path)
    assert catalog.get_runtime_lock("runtime-a").binary == (
        "cache/runtimes/runtime/server"
    )

    deployment["runtime_lock_id"] = "missing"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(CatalogError, match="unknown runtime lock"):
        Catalog.load(path)


def test_unlocked_host_and_mutable_container_are_rejected(tmp_path: Path) -> None:
    deployment = _deployment()
    deployment.update(backend="llama_cpp", executable="/opt/llama-server")
    document = {
        "models": [_model()],
        "artifacts": [_artifact()],
        "deployments": [deployment],
    }
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(CatalogError, match="reviewed runtime lock"):
        Catalog.load(path)

    deployment.pop("executable")
    deployment["image"] = {"reference": "example.invalid/runtime:latest"}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(CatalogError, match="immutable image digest"):
        Catalog.load(path)
