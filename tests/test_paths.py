from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from llm_lab.errors import StoragePolicyError
from llm_lab.paths import (
    DATA_DIRECTORIES,
    DATA_ROOT_SENTINEL,
    LabPaths,
    install_immutable_file_beneath,
)


def test_initialize_creates_complete_data_plane(tmp_path) -> None:
    paths = LabPaths(repo_root=tmp_path / "repo", data_root=tmp_path / "data")
    paths.initialize()

    assert paths.data_root.is_dir()
    assert (paths.data_root / DATA_ROOT_SENTINEL).read_text() == (
        '{"owner":"llm-lab","schema_version":1}\n'
    )
    for relative in DATA_DIRECTORIES:
        assert (paths.data_root / relative).is_dir()


def test_discover_honors_explicit_paths(tmp_path) -> None:
    paths = LabPaths.discover(tmp_path / "repo", tmp_path / "data")

    assert paths.repo_root == (tmp_path / "repo").resolve()
    assert paths.data_root == (tmp_path / "data").resolve()


def test_initialize_rejects_symlinked_managed_directory(tmp_path: Path) -> None:
    data = tmp_path / "data"
    external = tmp_path / "external"
    data.mkdir()
    external.mkdir()
    (data / "state").symlink_to(external, target_is_directory=True)
    paths = LabPaths(repo_root=tmp_path / "repo", data_root=data)

    with pytest.raises(StoragePolicyError, match="safe real directory"):
        paths.initialize()

    assert list(external.iterdir()) == []


def test_initialize_rejects_symlinked_data_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(external, target_is_directory=True)

    with pytest.raises(StoragePolicyError, match="safe real directory"):
        LabPaths(repo_root=tmp_path / "repo", data_root=linked).initialize()


def test_initialize_rejects_symlinked_data_root_ancestor(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(external, target_is_directory=True)

    with pytest.raises(StoragePolicyError, match="safe real directory"):
        LabPaths(
            repo_root=tmp_path / "repo",
            data_root=linked / "lab-data",
        ).initialize()

    assert not (external / "lab-data").exists()


def test_initialize_refuses_to_claim_unrelated_nonempty_directory(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    victim = data / "personal-notes.txt"
    victim.write_text("do not adopt", encoding="utf-8")

    with pytest.raises(StoragePolicyError, match="refusing to claim"):
        LabPaths(repo_root=tmp_path / "repo", data_root=data).initialize()

    assert victim.read_text(encoding="utf-8") == "do not adopt"


def test_initialize_rejects_tampered_or_linked_sentinel(tmp_path: Path) -> None:
    paths = LabPaths(repo_root=tmp_path / "repo", data_root=tmp_path / "data")
    paths.initialize()
    marker = paths.data_root / DATA_ROOT_SENTINEL
    marker.unlink()
    victim = tmp_path / "victim"
    victim.write_text("safe", encoding="utf-8")
    marker.symlink_to(victim)

    with pytest.raises(StoragePolicyError, match="sentinel"):
        paths.initialize()

    assert victim.read_text(encoding="utf-8") == "safe"


def test_immutable_installer_publishes_exact_reviewed_bytes_read_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    candidate = tmp_path / "candidate-server"
    payload = b"#!/bin/sh\necho reviewed-runtime\n"
    candidate.write_bytes(payload)
    candidate.chmod(0o555)
    digest = hashlib.sha256(payload).hexdigest()

    installed = install_immutable_file_beneath(
        root,
        "cache/runtime/bin/server",
        candidate,
        expected_sha256=digest,
        purpose="test runtime",
    )

    assert installed == root / "cache/runtime/bin/server"
    assert installed.read_bytes() == payload
    assert stat.S_IMODE(installed.stat().st_mode) == 0o555
    assert not list(installed.parent.glob(".*.install-*"))


def test_immutable_installer_digest_mismatch_preserves_existing_runtime(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    destination = root / "cache/runtime/bin/server"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"trusted-production")
    destination.chmod(0o555)
    candidate = tmp_path / "candidate-server"
    candidate.write_bytes(b"unreviewed-candidate")

    with pytest.raises(StoragePolicyError, match="candidate digest"):
        install_immutable_file_beneath(
            root,
            "cache/runtime/bin/server",
            candidate,
            expected_sha256="0" * 64,
            purpose="test runtime",
        )

    assert destination.read_bytes() == b"trusted-production"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert not list(destination.parent.glob(".*.install-*"))


@pytest.mark.parametrize("attack", ("ancestor", "destination"))
def test_immutable_installer_rejects_destination_symlinks_without_touching_victim(
    tmp_path: Path,
    attack: str,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "server"
    marker.write_bytes(b"do-not-change")
    candidate = tmp_path / "candidate-server"
    candidate.write_bytes(b"reviewed-candidate")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()

    if attack == "ancestor":
        (root / "cache").symlink_to(victim, target_is_directory=True)
        relative = "cache/server"
    else:
        (root / "cache").mkdir()
        (root / "cache/server").symlink_to(marker)
        relative = "cache/server"

    with pytest.raises(StoragePolicyError, match="unsafe|safe real directory"):
        install_immutable_file_beneath(
            root,
            relative,
            candidate,
            expected_sha256=digest,
            purpose="test runtime",
        )

    assert marker.read_bytes() == b"do-not-change"
    assert not list(root.rglob(".*.install-*"))


def test_immutable_installer_remains_bound_to_open_directory_during_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    (root / "cache/runtime").mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    victim_server = victim / "server"
    victim_server.write_bytes(b"do-not-change")
    candidate = tmp_path / "candidate-server"
    candidate.write_bytes(b"reviewed-candidate")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    real_replace = os.replace
    swapped = False

    def swap_parent_then_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            (root / "cache").rename(root / "original-cache")
            (root / "cache").symlink_to(victim, target_is_directory=True)
        real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr("llm_lab.paths.os.replace", swap_parent_then_replace)
    install_immutable_file_beneath(
        root,
        "cache/runtime/server",
        candidate,
        expected_sha256=digest,
        purpose="test runtime",
    )

    assert swapped is True
    assert victim_server.read_bytes() == b"do-not-change"
    assert (root / "original-cache/runtime/server").read_bytes() == b"reviewed-candidate"
