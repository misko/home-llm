"""Small deterministic utilities shared by preparation and the training worker."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def check_space(root: Path, planned_bytes: int = 0) -> None:
    free = shutil.disk_usage(root).free
    if free - planned_bytes < 540_000_000_000:
        raise RuntimeError(f'Insufficient space: {free} free, {planned_bytes} planned, 540GB reserve')


def normalize(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text.replace('\r\n', '\n').replace('\x00', '')).strip()


def split_document(text_hash: str, url: str | None, seed: int) -> str:
    if url:
        parsed = urlsplit(url)
        key = urlunsplit(('', parsed.netloc.lower(), parsed.path.rstrip('/'), '', ''))
    else:
        key = text_hash
    bucket = int(identity([seed, key])[:8], 16) % 100
    return 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'


def chunks(ids: list[int], length: int, eos: int):
    # End-of-document is appended once, not at every chunk boundary.
    ids = ids + [eos]
    for start in range(0, len(ids), length):
        chunk = ids[start:start + length]
        if len(chunk) >= 2:
            yield chunk


def elapsed_total(previous: float, started: float) -> float:
    return previous + time.monotonic() - started


def verify_checkpoint(path: Path) -> dict:
    manifest = json.loads((path / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if Path(name).name != name or digest(path / name) != expected:
            raise ValueError(f'Invalid checkpoint file: {name}')
    return manifest
