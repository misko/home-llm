"""Strict, cross-referenced YAML catalog loading."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, TypeVar, cast

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from .errors import CatalogError
from .schema import (
    ArtifactSpec,
    BenchmarkSuite,
    DeploymentSpec,
    ModelSpec,
    RuntimeLockSpec,
)


CatalogEntry = (
    ModelSpec | ArtifactSpec | DeploymentSpec | BenchmarkSuite | RuntimeLockSpec
)
EntryT = TypeVar("EntryT", bound=CatalogEntry)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


_SCHEMAS: dict[str, type[CatalogEntry]] = {
    "models": ModelSpec,
    "artifacts": ArtifactSpec,
    "deployments": DeploymentSpec,
    "suites": BenchmarkSuite,
    "runtime_locks": RuntimeLockSpec,
}

_CATEGORY_DIRECTORIES = {"runtime_locks": "runtime-locks"}
_IMMUTABLE_HF_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True, slots=True)
class Catalog:
    """An immutable, validated view of the declarative catalog."""

    models: Mapping[str, ModelSpec]
    artifacts: Mapping[str, ArtifactSpec]
    deployments: Mapping[str, DeploymentSpec]
    suites: Mapping[str, BenchmarkSuite]
    runtime_locks: Mapping[str, RuntimeLockSpec]

    def __post_init__(self) -> None:
        # A frozen dataclass does not make dictionaries immutable on its own.
        for field_name in _SCHEMAS:
            value = dict(cast(Mapping[str, CatalogEntry], getattr(self, field_name)))
            object.__setattr__(self, field_name, MappingProxyType(value))

    @classmethod
    def load(cls, root: str | Path) -> "Catalog":
        """Load a unified YAML file or a conventional catalog directory.

        A directory may contain ``catalog.yaml``, aggregate files such as
        ``models.yaml``, and/or individual entries below ``models/``,
        ``artifacts/``, ``deployments/``, ``suites/`` and ``runtime-locks/``.
        Both ``.yaml`` and ``.yml`` are recognized.
        """

        return _CatalogBuilder(Path(root)).load()

    def get_model(self, model_id: str) -> ModelSpec:
        return _required(self.models, model_id, "model")

    def get_artifact(self, artifact_id: str) -> ArtifactSpec:
        return _required(self.artifacts, artifact_id, "artifact")

    def get_deployment(self, deployment_id: str) -> DeploymentSpec:
        return _required(self.deployments, deployment_id, "deployment")

    def get_suite(self, suite_id: str) -> BenchmarkSuite:
        return _required(self.suites, suite_id, "benchmark suite")

    def get_runtime_lock(self, lock_id: str) -> RuntimeLockSpec:
        return _required(self.runtime_locks, lock_id, "runtime lock")


def _required(
    entries: Mapping[str, EntryT], entry_id: str, description: str
) -> EntryT:
    try:
        return entries[entry_id]
    except KeyError as exc:
        raise CatalogError(f"unknown {description} id {entry_id!r}") from exc


def load_catalog(root: str | Path) -> Catalog:
    """Convenience wrapper for :meth:`Catalog.load`."""

    return Catalog.load(root)


class _CatalogBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.entries: dict[str, dict[str, CatalogEntry]] = {
            category: {} for category in _SCHEMAS
        }
        self.origins: dict[tuple[str, str], Path] = {}

    def load(self) -> Catalog:
        if not self.root.exists():
            raise CatalogError(f"catalog path does not exist: {self.root}")

        if self.root.is_file():
            self._load_unified(self.root)
        elif self.root.is_dir():
            self._load_directory()
        else:
            raise CatalogError(f"catalog path is not a file or directory: {self.root}")

        self._validate_references()
        return Catalog(
            models=cast(dict[str, ModelSpec], self.entries["models"]),
            artifacts=cast(dict[str, ArtifactSpec], self.entries["artifacts"]),
            deployments=cast(
                dict[str, DeploymentSpec], self.entries["deployments"]
            ),
            suites=cast(dict[str, BenchmarkSuite], self.entries["suites"]),
            runtime_locks=cast(
                dict[str, RuntimeLockSpec], self.entries["runtime_locks"]
            ),
        )

    def _load_directory(self) -> None:
        for name in ("catalog.yaml", "catalog.yml"):
            path = self.root / name
            if path.is_file():
                self._load_unified(path)

        for category in _SCHEMAS:
            for suffix in (".yaml", ".yml"):
                aggregate = self.root / f"{category}{suffix}"
                if aggregate.is_file():
                    self._load_category_file(aggregate, category)

            directory = self.root / _CATEGORY_DIRECTORIES.get(category, category)
            if directory.is_dir():
                files = sorted(
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
                )
                for path in files:
                    self._load_category_file(path, category)

    def _load_unified(self, path: Path) -> None:
        for document in _yaml_documents(path):
            if not isinstance(document, Mapping):
                raise CatalogError(
                    f"{path}: unified catalog document must be a mapping"
                )
            keys = set(document)
            unknown = keys.difference(_SCHEMAS)
            if unknown:
                rendered = ", ".join(sorted(map(str, unknown)))
                raise CatalogError(
                    f"{path}: unknown top-level catalog section(s): {rendered}"
                )
            for category, section in document.items():
                self._add_section(str(category), section, path)

    def _load_category_file(self, path: Path, category: str) -> None:
        for document in _yaml_documents(path):
            if isinstance(document, Mapping) and category in document:
                if set(document) != {category}:
                    extras = set(document).difference({category})
                    rendered = ", ".join(sorted(map(str, extras)))
                    raise CatalogError(
                        f"{path}: {category!r} wrapper has unexpected key(s): "
                        f"{rendered}"
                    )
                document = document[category]
            elif isinstance(document, Mapping) and set(document).intersection(
                _SCHEMAS
            ):
                raise CatalogError(
                    f"{path}: expected {category!r} entries, found another "
                    "catalog section"
                )
            self._add_section(category, document, path)

    def _add_section(self, category: str, section: Any, path: Path) -> None:
        if category not in _SCHEMAS:
            raise CatalogError(f"{path}: unknown catalog section {category!r}")
        for raw in _expand_entries(section, path, category):
            schema = _SCHEMAS[category]
            try:
                entry = schema.model_validate(raw)
            except ValidationError as exc:
                raise CatalogError(
                    f"{path}: invalid {category[:-1]} entry: {exc}"
                ) from exc
            entry_id = entry.id
            key = (category, entry_id)
            if entry_id in self.entries[category]:
                first = self.origins[key]
                raise CatalogError(
                    f"duplicate {category[:-1]} id {entry_id!r} in {first} "
                    f"and {path}"
                )
            self.entries[category][entry_id] = entry
            self.origins[key] = path

    def _validate_references(self) -> None:
        models = self.entries["models"]
        artifacts = self.entries["artifacts"]
        deployments = self.entries["deployments"]
        runtime_locks = self.entries["runtime_locks"]

        for entry in models.values():
            model = cast(ModelSpec, entry)
            if (
                model.upstream.provider == "huggingface"
                and not _IMMUTABLE_HF_REVISION.fullmatch(model.upstream.revision)
            ):
                raise CatalogError(
                    f"model {model.id!r} must pin an immutable Hugging Face "
                    "commit revision"
                )

        for entry in artifacts.values():
            artifact = cast(ArtifactSpec, entry)
            if (
                artifact.source.provider == "huggingface"
                and not _IMMUTABLE_HF_REVISION.fullmatch(artifact.source.revision)
            ):
                raise CatalogError(
                    f"artifact {artifact.id!r} must pin an immutable Hugging Face "
                    "commit revision"
                )
            if artifact.model_id not in models:
                raise CatalogError(
                    f"artifact {artifact.id!r} references unknown model "
                    f"{artifact.model_id!r}"
                )

        aliases: dict[str, str] = {}
        for entry in deployments.values():
            deployment = cast(DeploymentSpec, entry)
            if deployment.artifact_id not in artifacts:
                raise CatalogError(
                    f"deployment {deployment.id!r} references unknown artifact "
                    f"{deployment.artifact_id!r}"
                )
            previous = aliases.get(deployment.public_alias)
            if previous is not None:
                raise CatalogError(
                    f"deployments {previous!r} and {deployment.id!r} share "
                    f"public_alias {deployment.public_alias!r}"
                )
            aliases[deployment.public_alias] = deployment.id
            if (
                deployment.runtime_lock_id is not None
                and deployment.runtime_lock_id not in runtime_locks
            ):
                raise CatalogError(
                    f"deployment {deployment.id!r} references unknown runtime lock "
                    f"{deployment.runtime_lock_id!r}"
                )
            if (
                deployment.backend.value not in {"mock", "external"}
                and deployment.image is None
                and deployment.runtime_lock_id is None
            ):
                raise CatalogError(
                    f"host deployment {deployment.id!r} must reference a "
                    "reviewed runtime lock"
                )
            if deployment.image is not None and (
                "@sha256:" not in deployment.image.reference
                and deployment.image.digest is None
            ):
                raise CatalogError(
                    f"container deployment {deployment.id!r} must pin an "
                    "immutable image digest"
                )


def _yaml_documents(path: Path) -> Iterable[Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            documents = list(yaml.load_all(stream, Loader=_UniqueKeyLoader))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"could not load YAML catalog {path}: {exc}") from exc
    if not documents:
        raise CatalogError(f"{path}: empty YAML file")
    for index, document in enumerate(documents, start=1):
        if document is None:
            raise CatalogError(f"{path}: YAML document {index} is empty")
        yield document


def load_runtime_lock_file(path: str | Path) -> RuntimeLockSpec:
    """Load one runtime lock with the catalog's duplicate-key protection."""

    lock_path = Path(path)
    documents = list(_yaml_documents(lock_path))
    if len(documents) != 1 or not isinstance(documents[0], Mapping):
        raise CatalogError(f"{lock_path}: runtime lock must be one YAML mapping")
    try:
        return RuntimeLockSpec.model_validate(documents[0])
    except ValidationError as exc:
        raise CatalogError(f"{lock_path}: invalid runtime lock: {exc}") from exc


def _expand_entries(
    section: Any, path: Path, category: str
) -> Iterable[Mapping[str, Any]]:
    if isinstance(section, list):
        candidates = section
    elif isinstance(section, Mapping) and "id" in section:
        candidates = [section]
    elif isinstance(section, Mapping):
        # A keyed aggregate is convenient for hand-maintained catalogs.  The
        # key is authoritative and may omit the redundant ``id`` field.
        candidates = []
        for keyed_id, value in section.items():
            if not isinstance(keyed_id, str) or not isinstance(value, Mapping):
                raise CatalogError(
                    f"{path}: keyed {category} entries must map string ids to mappings"
                )
            candidate = dict(value)
            embedded_id = candidate.get("id")
            if embedded_id is not None and embedded_id != keyed_id:
                raise CatalogError(
                    f"{path}: keyed id {keyed_id!r} disagrees with embedded "
                    f"id {embedded_id!r}"
                )
            candidate["id"] = keyed_id
            candidates.append(candidate)
    else:
        raise CatalogError(
            f"{path}: {category} section must be an entry, list, or keyed mapping"
        )

    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            raise CatalogError(
                f"{path}: {category} entry {index} must be a mapping"
            )
        yield cast(Mapping[str, Any], candidate)
