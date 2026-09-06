from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from llm_lab.errors import CatalogError, IntegrityError, StoragePolicyError
from llm_lab.hashing import sha256_bytes, sha256_uri
from llm_lab.registry import Registry, seal_manifest
from llm_lab.schema import (
    ArtifactFormat,
    ArtifactManifest,
    FileRole,
    LockedFile,
    RepositorySource,
)


def _manifest(artifact_id: str, payload: bytes = b"weights") -> ArtifactManifest:
    digest = sha256_bytes(payload)
    return ArtifactManifest(
        artifact_id=artifact_id,
        model_id="model-a",
        source=RepositorySource(provider="local", local_path="/source"),
        resolved_revision="local-v1",
        format=ArtifactFormat.GGUF,
        quantization="Q4_K_M",
        effective_bpw=4.8,
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        files=(
            LockedFile(
                logical_path="model.gguf",
                role=FileRole.WEIGHTS,
                size_bytes=len(payload),
                sha256=digest,
                storage_uri=sha256_uri(digest),
            ),
        ),
        total_logical_bytes=len(payload),
        tree_sha256=sha256_bytes(b"tree-" + payload),
    )


def test_initialize_and_artifact_registration_are_idempotent(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.initialize()
    first = registry.register_artifact(_manifest("artifact-a"))
    second = registry.register_artifact(first)

    assert first.manifest_sha256 == second.manifest_sha256
    assert registry.get_artifact("artifact-a") == first
    assert len(registry.list_artifacts()) == 1
    registry.close()


def test_artifact_id_cannot_be_reused_for_different_content(tmp_path: Path) -> None:
    with Registry(tmp_path / "registry.sqlite") as registry:
        registry.register_artifact(_manifest("artifact-a", b"one"))
        with pytest.raises(CatalogError, match="different immutable content"):
            registry.register_artifact(_manifest("artifact-a", b"two"))


def test_bad_manifest_digest_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest("artifact-a").model_copy(
        update={"manifest_sha256": "0" * 64}
    )
    with Registry(tmp_path / "registry.sqlite") as registry:
        with pytest.raises(IntegrityError, match="manifest digest mismatch"):
            registry.register_artifact(manifest)


def test_alias_history_and_rollback(tmp_path: Path) -> None:
    with Registry(tmp_path / "registry.sqlite") as registry:
        for artifact_id in ("artifact-a", "artifact-b", "artifact-c"):
            registry.register_artifact(_manifest(artifact_id, artifact_id.encode()))

        registry.set_alias("chat/default", "artifact-a")
        registry.set_alias("chat/default", "artifact-b")
        registry.set_alias("chat/default", "artifact-c", note="candidate")
        restored = registry.rollback_alias("chat/default", steps=2)

        assert restored.artifact_id == "artifact-a"
        assert restored.generation == 4
        assert registry.resolve_alias("chat/default") == "artifact-a"
        history = registry.list_alias_history("chat/default")
        assert history[0].action == "rollback"
        assert history[0].old_artifact_id == "artifact-c"
        assert history[0].new_artifact_id == "artifact-a"


def test_failed_alias_registration_rolls_back_all_rows(tmp_path: Path) -> None:
    with Registry(tmp_path / "registry.sqlite") as registry:
        with pytest.raises(CatalogError, match="unknown registered artifact"):
            registry.set_alias("chat/default", "missing")
        assert registry.find_alias("chat/default") is None
        assert registry.list_alias_history("chat/default") == ()


def test_run_and_generic_history_registration(tmp_path: Path) -> None:
    with Registry(tmp_path / "registry.sqlite") as registry:
        artifact = registry.register_artifact(_manifest("artifact-a"))
        run = registry.register_run(
            "run-1",
            suite_id="smoke",
            deployment_id="deployment-a",
            artifact_id=artifact.artifact_id,
            metadata={"tokens_per_second": 42.5},
        )
        repeated = registry.register_run(
            "run-1",
            suite_id="smoke",
            deployment_id="deployment-a",
            artifact_id=artifact.artifact_id,
            metadata={"tokens_per_second": 42.5},
        )
        event_id = registry.record_history(
            "verification.passed", "artifact", artifact.artifact_id, {"files": 1}
        )

        assert repeated == run
        assert registry.get_run("run-1").metadata["tokens_per_second"] == 42.5
        assert event_id > 0
        assert any(
            item.event_type == "verification.passed"
            for item in registry.list_history(entity_id="artifact-a")
        )


def test_seal_manifest_is_stable() -> None:
    first = seal_manifest(_manifest("artifact-a"))
    second = seal_manifest(first)
    assert first == second


def test_registry_rejects_database_symlink_without_touching_victim(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim.sqlite"
    with Registry(victim):
        pass
    before = victim.read_bytes()
    link = tmp_path / "registry.sqlite"
    link.symlink_to(victim)

    with pytest.raises(StoragePolicyError, match="registry database"):
        Registry(link)

    assert victim.read_bytes() == before


def test_registry_rejects_symlinked_parent_without_touching_victim(
    tmp_path: Path,
) -> None:
    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    marker = victim_dir / "marker"
    marker.write_bytes(b"unchanged")
    redirected = tmp_path / "redirected"
    redirected.symlink_to(victim_dir, target_is_directory=True)

    with pytest.raises(StoragePolicyError, match="safe real directory"):
        Registry(redirected / "catalog.sqlite")

    assert marker.read_bytes() == b"unchanged"
    assert not (victim_dir / "catalog.sqlite").exists()


@pytest.mark.parametrize("suffix", ("-journal", "-wal", "-shm"))
def test_registry_rejects_symlinked_database_sidecars_without_touching_victim(
    tmp_path: Path,
    suffix: str,
) -> None:
    database = tmp_path / "catalog.sqlite"
    victim = tmp_path / "victim-sidecar"
    victim.write_bytes(b"do-not-change")
    Path(f"{database}{suffix}").symlink_to(victim)

    with pytest.raises(StoragePolicyError, match="database sidecar"):
        Registry(database)

    assert victim.read_bytes() == b"do-not-change"


def test_registry_rechecks_sidecars_after_enabling_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "catalog.sqlite"
    wal = Path(f"{database}-wal")
    victim = tmp_path / "victim-wal"
    victim.write_bytes(b"do-not-change")
    real_connect = sqlite3.connect

    class PlantingConnection:
        def __init__(self, connection: sqlite3.Connection) -> None:
            object.__setattr__(self, "connection", connection)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.connection, name)

        def __setattr__(self, name: str, value: Any) -> None:
            setattr(self.connection, name, value)

        def execute(self, statement: str, *args: Any) -> Any:
            result = self.connection.execute(statement, *args)
            if statement.strip().upper() == "PRAGMA JOURNAL_MODE = WAL":
                assert not wal.exists()
                wal.symlink_to(victim)
            return result

    def connect(*args: Any, **kwargs: Any) -> PlantingConnection:
        return PlantingConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr("llm_lab.registry.sqlite3.connect", connect)

    with pytest.raises(StoragePolicyError, match="database sidecar -wal"):
        Registry(database)

    assert victim.read_bytes() == b"do-not-change"
