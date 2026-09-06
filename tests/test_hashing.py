from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_lab.hashing import (
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
    sha256_stream,
    sha256_uri,
)


def test_canonical_json_is_order_independent_and_compact() -> None:
    left = {"z": [3, 2, 1], "a": "雪", "nested": {"b": True, "a": None}}
    right = {"nested": {"a": None, "b": True}, "a": "雪", "z": [3, 2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json(left) == (
        '{"a":"雪","nested":{"a":null,"b":true},"z":[3,2,1]}'
    )
    assert canonical_sha256(left) == canonical_sha256(right)


def test_canonical_json_handles_domain_values() -> None:
    @dataclass
    class Record:
        path: Path
        when: datetime

    value = Record(Path("model.gguf"), datetime(2025, 1, 2, tzinfo=timezone.utc))
    assert canonical_json(value) == (
        '{"path":"model.gguf","when":"2025-01-02T00:00:00+00:00"}'
    )


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_stream_and_file_hashing(tmp_path: Path) -> None:
    payload = (b"model-weights\x00" * 10_000) + b"tail"
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)

    expected = sha256_bytes(payload)
    assert sha256_file(path, chunk_size=97) == expected
    with path.open("rb") as stream:
        assert sha256_stream(stream, chunk_size=113) == expected
    assert sha256_uri(expected) == f"sha256:{expected}"


def test_hashing_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        sha256_uri("ABC")
    with pytest.raises(ValueError):
        sha256_stream(__import__("io").BytesIO(b"x"), chunk_size=0)
