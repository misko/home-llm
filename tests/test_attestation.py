from __future__ import annotations

from pathlib import Path

import pytest

from llm_lab.attestation import (
    create_gateway_attestation,
    load_or_create_gateway_key,
    verify_gateway_attestation,
)
from llm_lab.errors import IntegrityError
from llm_lab.paths import LabPaths


def _paths(tmp_path: Path) -> LabPaths:
    paths = LabPaths(
        repo_root=tmp_path / "repo",
        data_root=tmp_path / "data",
    )
    paths.initialize()
    return paths


def test_gateway_key_is_private_stable_and_challenge_bound(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    first = load_or_create_gateway_key(paths)
    second = load_or_create_gateway_key(paths)
    assert first == second
    assert len(first) == 32
    assert paths.gateway_attestation_key_path.stat().st_mode & 0o777 == 0o600

    attestation = create_gateway_attestation(
        paths,
        challenge="a" * 64,
        origin="http://127.0.0.1:14000",
        active=True,
        ready=True,
        deployment="deployment",
        model="model",
    )
    verified = verify_gateway_attestation(
        paths,
        attestation,
        challenge="a" * 64,
        origin="http://127.0.0.1:14000",
        deployment="deployment",
        model="model",
    )
    assert verified == attestation
    with pytest.raises(IntegrityError, match="payload does not match"):
        verify_gateway_attestation(
            paths,
            attestation,
            challenge="b" * 64,
            origin="http://127.0.0.1:14000",
            deployment="deployment",
            model="model",
        )


def test_gateway_key_path_never_follows_a_symbolic_link(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    victim = tmp_path / "victim"
    victim.write_bytes(b"v" * 32)
    paths.gateway_attestation_key_path.symlink_to(victim)

    with pytest.raises(IntegrityError, match="attestation key"):
        load_or_create_gateway_key(paths)

    assert victim.read_bytes() == b"v" * 32
