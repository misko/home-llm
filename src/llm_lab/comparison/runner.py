from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from .contracts import Case, Outcome, atomic_json, fingerprint
from .scoring import score


async def generate(client: httpx.AsyncClient, model: str, case: Case, sampling: dict[str, Any]) -> dict[str, Any]:
    body = {"model": model, "messages": case.messages, **sampling,
            "max_tokens": case.max_tokens, "stream": True, "stream_options": {"include_usage": True}}
    if case.tools:
        body.update(tools=case.tools, tool_choice="auto")
    started = time.monotonic()
    ttft = None
    text = ""
    calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    finish = None
    done = False
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                done = True
                break
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError(f"Backend streaming error: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            if event.get('timings'):
                timings=event['timings']
            for choice in event.get("choices", []):
                if choice.get("index", 0) != 0:
                    raise ValueError("Only one response per request is supported")
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("tool_calls"):
                    if ttft is None:
                        ttft = time.monotonic() - started
                text += delta.get("content") or ""
                if delta.get("reasoning_content"):
                    raise ValueError("Thinking output appeared in the nonthinking track")
                for fragment in delta.get("tool_calls", []):
                    current = calls.setdefault(fragment["index"], {"function": {"name": "", "arguments": ""}})
                    if fragment.get("id"):
                        current["id"] = fragment["id"]
                    for key in ("name", "arguments"):
                        current["function"][key] += fragment.get("function", {}).get(key) or ""
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    if not done or finish is None:
        raise ValueError("Incomplete stream: missing completion marker or finish reason")
    return {"response": text, "tool_calls": [calls[k] for k in sorted(calls)],
            "finish_reason": finish, "latency_seconds": time.monotonic() - started,
            "ttft_seconds": ttft, "usage": usage,'server_timings':timings}


def latest_outcomes(root: Path) -> dict[str, Outcome]:
    results = {}
    for directory in sorted((root / "cases").glob("*")):
        files = sorted(directory.glob("attempt-*.json"))
        if files:
            row = Outcome.model_validate_json(files[-1].read_text())
            if row.case_id in results:
                raise ValueError("Duplicate case result")
            results[row.case_id] = row
    return results


async def run_cases(root: Path, arm: str, cases: list[Case], protocol: dict[str, Any], client: httpx.AsyncClient,
                    model: str, *, retry_infrastructure: bool = False, on_result=None) -> list[Outcome]:
    # Caller must hold the Lab benchmark lease (or disposable-model training lease).
    contract = {"arm": arm, "protocol": protocol, "cases": [c.model_dump() for c in cases]}
    protocol_hash = fingerprint(contract)
    run_path = root / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text())["sha256"] != protocol_hash:
            raise ValueError("Cannot resume with changed cases, model provenance, or protocol")
    else:
        atomic_json(run_path, {"sha256": protocol_hash, **contract})
    existing = latest_outcomes(root)
    output = []
    for case in cases:
        case_hash = fingerprint(case.model_dump())
        prior = existing.get(case.id)
        if prior:
            if prior.protocol_sha256 != protocol_hash or prior.case_sha256 != case_hash or prior.arm != arm:
                raise ValueError("Existing case result has incompatible provenance")
            if prior.status != "infrastructure_error" or not retry_infrastructure:
                output.append(prior)
                continue
        started = time.monotonic()
        try:
            generated = await generate(client, model, case, protocol["sampling"])
            status, metrics, details = score(case, generated["response"], generated["tool_calls"], generated["finish_reason"])
            row = Outcome(case_id=case.id, arm=arm, case_sha256=case_hash, protocol_sha256=protocol_hash,
                          status=status, metrics=metrics, details=details, **generated)
        except Exception as exc:
            row = Outcome(case_id=case.id, arm=arm, case_sha256=case_hash, protocol_sha256=protocol_hash,
                          status="infrastructure_error", latency_seconds=time.monotonic() - started,
                          error=f"{type(exc).__name__}: {exc}")
        directory = root / "cases" / fingerprint(case.id)
        attempt = len(list(directory.glob("attempt-*.json"))) + 1
        atomic_json(directory / f"attempt-{attempt:04d}.json", row.model_dump())
        output.append(row)
        if on_result:
            on_result(row)
        if row.status == "infrastructure_error":
            raise RuntimeError(f"Evaluation stopped; saved infrastructure failure for {case.id}: {row.error}")
    return output
