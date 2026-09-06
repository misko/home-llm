from __future__ import annotations

import errno
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from llm_lab.errors import CatalogError, IntegrityError, StoragePolicyError
from llm_lab.hashing import sha256_bytes
from llm_lab.paths import LabPaths
from llm_lab.schema import (
    ArtifactFileSelector,
    ArtifactFormat,
    ArtifactSpec,
    FileRole,
    RepositorySource,
)
from llm_lab.storage import ArtifactStore, artifact_path_matches


def _paths(tmp_path: Path) -> LabPaths:
    repo = tmp_path / "repo"
    repo.mkdir()
    return LabPaths(repo_root=repo, data_root=tmp_path / "data")


def _artifact(
    artifact_id: str,
    source: Path,
    *,
    expected_size: int | None = None,
    pattern: str = "model.gguf",
) -> ArtifactSpec:
    return ArtifactSpec(
        id=artifact_id,
        model_id="model-a",
        source=RepositorySource(provider="local", local_path=str(source)),
        format=ArtifactFormat.GGUF,
        quantization="Q4_K_M",
        effective_bpw=4.8,
        expected_size_bytes=expected_size,
        files=(
            ArtifactFileSelector(
                pattern=pattern,
                role=FileRole.WEIGHTS,
                required=True,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("logical_path", "pattern", "expected"),
    [
        ("model.gguf", "model.gguf", True),
        ("sub/model.gguf", "model.gguf", False),
        ("sub/model.gguf", "*.gguf", False),
        ("model.gguf", "**/*.gguf", True),
        ("a/b/model.gguf", "**/*.gguf", True),
        ("a/b/model.gguf", "a/*.gguf", False),
        ("a/model.gguf", "a/*.gguf", True),
    ],
)
def test_artifact_selector_matching_is_rooted(
    logical_path: str, pattern: str, expected: bool
) -> None:
    assert artifact_path_matches(logical_path, pattern) is expected


def test_promote_builds_manifest_view_registry_and_alias(tmp_path: Path) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    payload = b"gguf-model-bytes"
    (tree / "model.gguf").write_bytes(payload)
    paths = _paths(tmp_path)

    with ArtifactStore(paths) as store:
        result = store.promote(
            _artifact("artifact-a", tree, expected_size=len(payload)),
            tree,
            "a" * 40,
            alias="chat/default",
        )

        assert result.new_blob_count == 1
        assert result.new_bytes == len(payload)
        assert result.manifest_path.is_file()
        assert (result.view_path / "model.gguf").read_bytes() == payload
        blob = store.blob_path(result.manifest.files[0].sha256)
        assert not os.path.samefile(tree / "model.gguf", blob)
        # Default copy mode keeps the upstream/download cache independent.
        (tree / "model.gguf").write_bytes(b"source may change")
        assert store.registry.resolve_alias("chat/default") == "artifact-a"
        report = store.verify("artifact-a")
        assert report.file_count == 1
        assert report.total_logical_bytes == len(payload)


def test_manifest_identity_is_portable_across_import_times(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "model.gguf").write_bytes(b"same-reviewed-weights")
    moments = iter(
        (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=1),
        )
    )

    class FixtureClock:
        @classmethod
        def now(cls, _timezone: object) -> datetime:
            return next(moments)

    monkeypatch.setattr("llm_lab.storage.datetime", FixtureClock)
    manifests = []
    for suffix in ("a", "b"):
        paths = LabPaths(
            repo_root=tmp_path / f"repo-{suffix}",
            data_root=tmp_path / f"data-{suffix}",
        )
        with ArtifactStore(paths) as store:
            manifests.append(
                store.promote(
                    _artifact("artifact-a", tree),
                    tree,
                    "local-v1",
                ).manifest
            )

    assert manifests[0].created_at != manifests[1].created_at
    assert manifests[0].tree_sha256 == manifests[1].tree_sha256
    assert manifests[0].manifest_sha256 == manifests[1].manifest_sha256


def test_failed_registry_publication_removes_frozen_view_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "model.gguf").write_bytes(b"weights")
    paths = _paths(tmp_path)

    with ArtifactStore(paths) as store:
        def fail_registration(*_args: object, **_kwargs: object) -> None:
            raise CatalogError("deliberate registry failure")

        monkeypatch.setattr(store.registry, "register_artifact", fail_registration)
        with pytest.raises(CatalogError, match="deliberate registry failure"):
            store.promote(_artifact("artifact-a", tree), tree, "local")

        assert not store.view_path("artifact-a").exists()
        assert not (paths.manifest_root / "artifact-a").exists()
        assert not list(paths.view_root.glob(".artifact-a.*.tmp"))
        assert store.registry.find_artifact("artifact-a") is None


def test_explicit_view_permission_repair_refreezes_verified_legacy_view(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "model.gguf").write_bytes(b"weights")
    paths = _paths(tmp_path)

    with ArtifactStore(paths) as store:
        promoted = store.promote(_artifact("artifact-a", tree), tree, "local")
        promoted.view_path.chmod(0o750)
        with pytest.raises(IntegrityError, match="view.*corrupt"):
            store.verify("artifact-a")

        repaired = store.repair_view_permissions("artifact-a")

        assert repaired.view_verified is True
        assert promoted.view_path.stat().st_mode & 0o222 == 0
        assert store.verify("artifact-a") == repaired


def test_promotion_deduplicates_identical_bytes(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    payload = b"same-quantized-weights"
    (one / "model.gguf").write_bytes(payload)
    (two / "model.gguf").write_bytes(payload)
    paths = _paths(tmp_path)

    with ArtifactStore(paths) as store:
        first = store.promote(_artifact("artifact-a", one), one, "local-1")
        second = store.promote(_artifact("artifact-b", two), two, "local-2")

        assert first.new_blob_count == 1
        assert second.new_blob_count == 0
        assert second.reused_bytes == len(payload)
        assert len(list(paths.blob_root.iterdir())) == 1
        assert os.path.samefile(
            first.view_path / "model.gguf", second.view_path / "model.gguf"
        )


def test_required_selector_must_match(tmp_path: Path) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    paths = _paths(tmp_path)
    with ArtifactStore(paths) as store:
        with pytest.raises(IntegrityError, match="matched no files"):
            store.promote(_artifact("artifact-a", tree), tree, "local")
        assert store.registry.list_artifacts() == ()


def test_expected_size_must_match_selected_files(tmp_path: Path) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "model.gguf").write_bytes(b"1234")
    paths = _paths(tmp_path)
    with ArtifactStore(paths) as store:
        with pytest.raises(IntegrityError, match="catalog expected 5"):
            store.promote(
                _artifact("artifact-a", tree, expected_size=5), tree, "local"
            )


def test_insufficient_free_reserve_writes_no_blobs(tmp_path: Path) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "model.gguf").write_bytes(b"weights")
    paths = _paths(tmp_path)
    with ArtifactStore(paths, free_reserve_bytes=10**30) as store:
        with pytest.raises(StoragePolicyError, match="storage policy"):
            store.promote(_artifact("artifact-a", tree), tree, "local")
        assert list(paths.blob_root.iterdir()) == []
        assert store.registry.list_artifacts() == ()


def test_verification_detects_blob_corruption(tmp_path: Path) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "model.gguf").write_bytes(b"good")
    paths = _paths(tmp_path)
    with ArtifactStore(paths) as store:
        result = store.promote(_artifact("artifact-a", tree), tree, "local")
        blob = store.blob_path(result.manifest.files[0].sha256)
        blob.chmod(0o644)
        blob.write_bytes(b"evil")
        with pytest.raises(IntegrityError, match="corruption"):
            store.verify("artifact-a")


def test_verification_detects_persisted_manifest_corruption(tmp_path: Path) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "model.gguf").write_bytes(b"good")
    paths = _paths(tmp_path)
    with ArtifactStore(paths) as store:
        result = store.promote(_artifact("artifact-a", tree), tree, "local")
        result.manifest_path.chmod(0o644)
        result.manifest_path.write_text("{}", encoding="utf-8")
        with pytest.raises(IntegrityError, match="manifest is corrupt"):
            store.verify("artifact-a")


def test_string_artifact_id_cannot_be_shadowed_by_cwd_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_tree = tmp_path / "first"
    second_tree = tmp_path / "second"
    first_tree.mkdir()
    second_tree.mkdir()
    (first_tree / "model.gguf").write_bytes(b"good")
    (second_tree / "model.gguf").write_bytes(b"safe")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    paths = _paths(tmp_path)

    with ArtifactStore(paths) as store:
        first = store.promote(_artifact("artifact-a", first_tree), first_tree, "a")
        second = store.promote(_artifact("artifact-b", second_tree), second_tree, "b")
        (cwd / "artifact-a").write_bytes(second.manifest_path.read_bytes())

        first_blob = store.blob_path(first.manifest.files[0].sha256)
        first_blob.chmod(0o644)
        first_blob.write_bytes(b"evil")
        monkeypatch.chdir(cwd)

        with pytest.raises(IntegrityError, match="corruption"):
            store.verify("artifact-a")
        assert store.verify(Path("artifact-a")).artifact_id == "artifact-b"


def test_hardlink_failure_falls_back_to_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    source = tree / "model.gguf"
    source.write_bytes(b"weights")
    paths = _paths(tmp_path)

    def no_hardlinks(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr("llm_lab.storage.os.link", no_hardlinks)
    with ArtifactStore(paths, blob_install_mode="hardlink") as store:
        result = store.promote(_artifact("artifact-a", tree), tree, "local")
        assert store.verify(result.manifest).file_count == 1
        assert not os.path.samefile(
            source, store.blob_path(result.manifest.files[0].sha256)
        )


def test_explicit_hardlink_mode_adopts_and_protects_source_inode(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    source = tree / "model.gguf"
    source.write_bytes(b"weights")
    paths = _paths(tmp_path)

    with ArtifactStore(paths, blob_install_mode="hardlink") as store:
        result = store.promote(_artifact("artifact-a", tree), tree, "local")
        blob = store.blob_path(result.manifest.files[0].sha256)
        assert os.path.samefile(source, blob)
        assert source.stat().st_mode & 0o222 == 0

        # The opt-in tradeoff is explicit: an owner can make the shared inode
        # writable again, and verification must then catch the CAS corruption.
        source.chmod(0o644)
        source.write_bytes(b"tamper!")
        with pytest.raises(IntegrityError, match="corruption"):
            store.verify("artifact-a")


def test_gc_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with ArtifactStore(paths) as store:
        payload = b"unreferenced"
        digest = sha256_bytes(payload)
        orphan = store.blob_path(digest)
        orphan.write_bytes(payload)

        preview = store.gc(dry_run=True)
        assert preview.candidate_count == 1
        assert preview.candidate_bytes == len(payload)
        assert orphan.exists()

        applied = store.gc(dry_run=False)
        assert applied.removed_count == 1
        assert not orphan.exists()


def test_gc_holds_storage_lock_while_snapshotting_roots(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "model.gguf").write_bytes(b"new-live-blob")
    paths = _paths(tmp_path)
    roots_snapshotted = Event()
    release_root_scan = Event()
    promotion_attempted = Event()
    promotion_acquired_lock = Event()
    errors: list[BaseException] = []

    with ArtifactStore(paths) as gc_store, ArtifactStore(paths) as promote_store:
        original_live_manifests = gc_store._live_manifests

        def paused_live_manifests(keep_artifact_ids):
            result = original_live_manifests(keep_artifact_ids)
            roots_snapshotted.set()
            if not release_root_scan.wait(5):
                raise RuntimeError("test timed out while root scan was paused")
            return result

        gc_store._live_manifests = paused_live_manifests
        original_promotion_lock = promote_store._storage_lock

        @contextmanager
        def observed_promotion_lock():
            promotion_attempted.set()
            with original_promotion_lock():
                promotion_acquired_lock.set()
                yield

        promote_store._storage_lock = observed_promotion_lock

        def collect() -> None:
            try:
                gc_store.gc(dry_run=False)
            except BaseException as exc:  # surfaced in the main test thread below
                errors.append(exc)

        def promote() -> None:
            try:
                promote_store.promote(
                    _artifact("race-artifact", tree), tree, "local"
                )
            except BaseException as exc:  # surfaced in the main test thread below
                errors.append(exc)

        gc_thread = Thread(target=collect)
        gc_thread.start()
        assert roots_snapshotted.wait(5)
        promotion_thread = Thread(target=promote)
        promotion_thread.start()
        assert promotion_attempted.wait(5)

        # Promotion must be blocked until GC finishes both its root snapshot and
        # candidate sweep.  The old ordering acquired the promotion lock here.
        promotion_was_blocked = not promotion_acquired_lock.wait(0.25)
        release_root_scan.set()
        gc_thread.join(5)
        promotion_thread.join(5)

        assert promotion_was_blocked
        assert not gc_thread.is_alive()
        assert not promotion_thread.is_alive()
        assert errors == []
        assert promote_store.verify("race-artifact").file_count == 1


@pytest.mark.parametrize("relative", ("blobs/sha256", "manifests", "views"))
def test_store_rejects_symlinked_managed_roots(
    tmp_path: Path, relative: str
) -> None:
    paths = _paths(tmp_path)
    paths.data_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    managed = paths.data_root / relative
    managed.parent.mkdir(parents=True, exist_ok=True)
    managed.symlink_to(external, target_is_directory=True)

    with pytest.raises(StoragePolicyError, match="symbolic link"):
        ArtifactStore(paths)


def test_store_rejects_a_symlinked_data_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    linked_data = tmp_path / "linked-data"
    linked_data.symlink_to(external, target_is_directory=True)

    with pytest.raises(StoragePolicyError, match="data root.*symbolic link"):
        ArtifactStore(LabPaths(repo_root=repo, data_root=linked_data))
    with pytest.raises(StoragePolicyError, match="data root.*symbolic link"):
        ArtifactStore(linked_data)


@pytest.mark.parametrize("managed_kind", ("manifest", "view"))
def test_promotion_rejects_symlinked_artifact_destination(
    tmp_path: Path, managed_kind: str
) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "model.gguf").write_bytes(b"weights")
    paths = _paths(tmp_path)
    external = tmp_path / "external"
    external.mkdir()

    with ArtifactStore(paths) as store:
        destination = (
            paths.manifest_root / "artifact-a"
            if managed_kind == "manifest"
            else paths.view_root / "artifact-a"
        )
        destination.symlink_to(external, target_is_directory=True)

        with pytest.raises(StoragePolicyError, match="symbolic link"):
            store.promote(_artifact("artifact-a", tree), tree, "local")

    assert list(external.iterdir()) == []


def test_gc_rechecks_blob_root_and_never_deletes_through_symlink(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    victim = external / ("a" * 64)
    victim.write_bytes(b"unrelated")

    with ArtifactStore(paths) as store:
        paths.blob_root.rmdir()
        paths.blob_root.symlink_to(external, target_is_directory=True)

        with pytest.raises(StoragePolicyError, match="symbolic link"):
            store.gc(dry_run=False)

    assert victim.read_bytes() == b"unrelated"


def test_verification_rejects_view_symlink_to_equal_external_bytes(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    payload = b"weights"
    (tree / "model.gguf").write_bytes(payload)
    paths = _paths(tmp_path)

    with ArtifactStore(paths) as store:
        promoted = store.promote(_artifact("artifact-a", tree), tree, "local")
        external = tmp_path / "external.gguf"
        external.write_bytes(payload)
        view_member = promoted.view_path / "model.gguf"
        promoted.view_path.chmod(0o750)
        view_member.unlink()
        view_member.symlink_to(external)

        with pytest.raises(IntegrityError, match="view.*corrupt"):
            store.verify("artifact-a")


def test_verification_rejects_writable_private_view_copy(tmp_path: Path) -> None:
    tree = tmp_path / "source"
    tree.mkdir()
    payload = b"weights"
    (tree / "model.gguf").write_bytes(payload)
    paths = _paths(tmp_path)

    with ArtifactStore(paths) as store:
        promoted = store.promote(_artifact("artifact-a", tree), tree, "local")
        view_member = promoted.view_path / "model.gguf"
        assert promoted.view_path.stat().st_mode & 0o222 == 0
        promoted.view_path.chmod(0o750)
        view_member.unlink()
        view_member.write_bytes(payload)
        view_member.chmod(0o666)
        promoted.view_path.chmod(0o555)

        with pytest.raises(IntegrityError, match="view.*corrupt"):
            store.verify("artifact-a")


def test_hf_pull_resolves_once_and_pins_download(tmp_path: Path) -> None:
    tree = tmp_path / "snapshot"
    tree.mkdir()
    payload = b"downloaded"
    (tree / "model.gguf").write_bytes(payload)
    paths = _paths(tmp_path)
    resolved = "c" * 40
    source = RepositorySource(
        provider="huggingface", repo_id="example/model", revision="main"
    )
    artifact = _artifact("artifact-a", tree, expected_size=len(payload)).model_copy(
        update={"source": source}
    )
    calls: list[dict] = []

    class FakeApi:
        def repo_info(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["revision"] == "main"
            return SimpleNamespace(sha=resolved)

    def fake_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(tree)

    with ArtifactStore(paths) as store:
        pulled = store.pull_huggingface(
            artifact, api=FakeApi(), snapshot_download_fn=fake_download
        )

    assert pulled.resolved_revision == resolved
    assert pulled.selected_files == ("model.gguf",)
    assert calls[0]["revision"] == resolved
    assert calls[0]["allow_patterns"] == ["model.gguf"]
