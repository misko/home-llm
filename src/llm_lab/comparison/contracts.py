from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Case(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.:/-]+$")
    group: str
    task: str
    partition: Literal["development", "final"]
    messages: list[dict[str, str]]
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    scorer: Literal["exact", "number", "choice", "json", "tool", "ifeval", "humaneval_plus", "rubric"]
    expected: Any
    metadata: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_messages(self):
        if not self.messages or any(set(m) != {"role", "content"} or m["role"] not in {"system", "user", "assistant"} for m in self.messages):
            raise ValueError("Cases require nonempty text-only chat messages")
        return self


class Outcome(StrictModel):
    case_id: str
    arm: str
    case_sha256: str
    protocol_sha256: str
    status: Literal["scored", "pending_judgment", "infrastructure_error"]
    metrics: dict[str, float] = Field(default_factory=dict)
    response: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    latency_seconds: float = Field(ge=0)
    ttft_seconds: float | None = Field(default=None, ge=0)
    usage: dict[str, Any] = Field(default_factory=dict)
    server_timings: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def finite(self):
        if any(not math.isfinite(x) for x in self.metrics.values()):
            raise ValueError("Metrics must be finite")
        return self


def load_cases(path: Path) -> list[Case]:
    cases = [Case.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("Case manifest is empty or contains duplicate IDs")
    partitions: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for case in cases:
        if partitions.setdefault(case.group, case.partition) != case.partition:
            raise ValueError("Prompt group crosses development/final boundary")
        prompt = fingerprint(case.messages)
        if prompts.setdefault(prompt, case.partition) != case.partition:
            raise ValueError("Identical prompt crosses development/final boundary")
    return cases
