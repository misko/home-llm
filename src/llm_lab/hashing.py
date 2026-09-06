"""Deterministic serialization and hashing helpers.

The registry treats bytes, rather than filenames, as identity.  This module is
deliberately small so that every component computes object and manifest
digests in exactly the same way.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO


DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


def _json_default(value: Any) -> Any:
    """Convert the few domain values accepted by canonical JSON.

    Pydantic models are handled without importing pydantic here, keeping this
    utility usable during early startup and in small maintenance scripts.
    """

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize *value* to stable UTF-8 JSON bytes.

    Object keys are sorted, insignificant whitespace is omitted, Unicode is
    preserved, and NaN/infinity are rejected.  The result is suitable for
    hashing immutable manifests.  This is a deliberately documented local
    canonical form, not a claim of full RFC 8785 number normalization.
    """

    return json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    """Return :func:`canonical_json_bytes` decoded as text."""

    return canonical_json_bytes(value).decode("utf-8")


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Return the lowercase hexadecimal SHA-256 digest of *data*."""

    return hashlib.sha256(data).hexdigest()


def sha256_stream(
    stream: BinaryIO, *, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> str:
    """Hash a binary stream from its current position to EOF."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(
    path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> str:
    """Hash a file without loading it into memory."""

    with Path(path).open("rb") as stream:
        return sha256_stream(stream, chunk_size=chunk_size)


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON encoding of *value*."""

    return sha256_bytes(canonical_json_bytes(value))


def sha256_uri(hex_digest: str) -> str:
    """Return a ``sha256:<hex>`` URI after validating the digest."""

    if len(hex_digest) != 64 or any(
        character not in "0123456789abcdef" for character in hex_digest
    ):
        raise ValueError("expected a lowercase 64-character SHA-256 digest")
    return f"sha256:{hex_digest}"


def verify_file_sha256(path: str | Path, expected: str) -> bool:
    """Return whether *path* has the expected lowercase SHA-256 digest."""

    return sha256_file(path) == expected
