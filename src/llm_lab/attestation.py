"""Challenge-response identity for the local LLM Lab gateway."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .errors import IntegrityError
from .paths import LabPaths


DEFAULT_GATEWAY_ORIGIN = "http://127.0.0.1:14000"
GATEWAY_KEY_NAME = "gateway-attestation.key"
_KEY_BYTES = 32


def normalize_gateway_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise IntegrityError(f"invalid gateway origin: {value!r}")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), "", "", ""))


def configured_gateway_origin() -> str:
    return normalize_gateway_origin(
        os.environ.get("LLM_LAB_GATEWAY_URL", DEFAULT_GATEWAY_ORIGIN)
    )


def _open_state_directory(paths: LabPaths) -> int:
    paths.initialize()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        return os.open(paths.data_root / "state", flags)
    except OSError as exc:
        raise IntegrityError(f"cannot safely open gateway state directory: {exc}") from exc


def _validate_key_descriptor(descriptor: int, state_fd: int, path: Path) -> None:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(
            GATEWAY_KEY_NAME,
            dir_fd=state_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise IntegrityError(f"cannot validate gateway attestation key {path}: {exc}") from exc
    if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(named.st_mode):
        raise IntegrityError(f"gateway attestation key is not a regular file: {path}")
    if opened.st_nlink != 1:
        raise IntegrityError(f"gateway attestation key must not be hard-linked: {path}")
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise IntegrityError(f"gateway attestation key changed while opening: {path}")
    if opened.st_mode & 0o077:
        raise IntegrityError(f"gateway attestation key permissions are too broad: {path}")
    if hasattr(os, "getuid") and opened.st_uid != os.getuid():
        raise IntegrityError(f"gateway attestation key has the wrong owner: {path}")
    if opened.st_size != _KEY_BYTES:
        raise IntegrityError(f"gateway attestation key has the wrong size: {path}")


def load_or_create_gateway_key(paths: LabPaths) -> bytes:
    """Return the private local key without following links or widening access."""

    state_fd = _open_state_directory(paths)
    path = paths.data_root / "state" / GATEWAY_KEY_NAME
    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(GATEWAY_KEY_NAME, read_flags, dir_fd=state_fd)
        except FileNotFoundError:
            key = secrets.token_bytes(_KEY_BYTES)
            create_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(
                    GATEWAY_KEY_NAME,
                    create_flags,
                    0o600,
                    dir_fd=state_fd,
                )
            except FileExistsError:
                descriptor = os.open(GATEWAY_KEY_NAME, read_flags, dir_fd=state_fd)
            else:
                try:
                    view = memoryview(key)
                    while view:
                        written = os.write(descriptor, view)
                        view = view[written:]
                    os.fsync(descriptor)
                    os.fsync(state_fd)
                finally:
                    os.close(descriptor)
                return key
        except OSError as exc:
            raise IntegrityError(f"cannot open gateway attestation key {path}: {exc}") from exc

        try:
            _validate_key_descriptor(descriptor, state_fd, path)
            key = b""
            while len(key) < _KEY_BYTES:
                block = os.read(descriptor, _KEY_BYTES - len(key))
                if not block:
                    break
                key += block
            if len(key) != _KEY_BYTES:
                raise IntegrityError(f"gateway attestation key is truncated: {path}")
            return key
        finally:
            os.close(descriptor)
    finally:
        os.close(state_fd)


def _message(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def create_gateway_attestation(
    paths: LabPaths,
    *,
    challenge: str,
    origin: str,
    active: bool,
    ready: bool,
    deployment: str | None,
    model: str | None,
) -> dict[str, Any]:
    if len(challenge) != 64 or any(character not in "0123456789abcdef" for character in challenge):
        raise IntegrityError("gateway attestation challenge must be 32-byte lowercase hex")
    payload = {
        "version": 1,
        "challenge": challenge,
        "origin": normalize_gateway_origin(origin),
        "active": active,
        "ready": ready,
        "deployment": deployment,
        "model": model,
    }
    signature = hmac.new(
        load_or_create_gateway_key(paths), _message(payload), hashlib.sha256
    ).hexdigest()
    return {"payload": payload, "hmac_sha256": signature}


def verify_gateway_attestation(
    paths: LabPaths,
    value: Any,
    *,
    challenge: str,
    origin: str,
    deployment: str,
    model: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError("gateway response has no attestation object")
    payload = value.get("payload")
    signature = value.get("hmac_sha256")
    if not isinstance(payload, Mapping) or not isinstance(signature, str):
        raise IntegrityError("gateway attestation is malformed")
    expected_payload = {
        "version": 1,
        "challenge": challenge,
        "origin": normalize_gateway_origin(origin),
        "active": True,
        "ready": True,
        "deployment": deployment,
        "model": model,
    }
    if dict(payload) != expected_payload:
        raise IntegrityError("gateway attestation payload does not match the benchmark target")
    expected_signature = hmac.new(
        load_or_create_gateway_key(paths),
        _message(expected_payload),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise IntegrityError("gateway attestation signature is invalid")
    return {"payload": expected_payload, "hmac_sha256": signature}
