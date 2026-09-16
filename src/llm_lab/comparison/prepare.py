"""Freeze pinned public sources and deterministic local tasks without using a GPU."""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import httpx

from .contracts import Case, atomic_json, file_hash, fingerprint, load_cases


SOURCES = {
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "ifeval_input_data.jsonl", "apache-2.0"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet", "mit"),
    "mmlu-pro": ("TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "data/test-00000-of-00001.parquet", "mit"),
}
HUMANEVAL_FILE = 'HumanEvalPlus-v0.1.10.jsonl'
HUMANEVAL_SHA256 = '42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f'
XSTEST_REVISION = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
SEED = 20260913


def ordered(rows, key):
    return sorted(rows, key=lambda row: fingerprint([SEED, key(row)]))


def user_case(task, identifier, prompt, scorer, expected, *, partition="final", group=None, max_tokens=1024, metadata=None, tools=None):
    return Case(id=f"{task}/{identifier}", group=group or f"{task}/{identifier}", task=task,
                partition=partition, messages=[{"role": "user", "content": prompt}], scorer=scorer,
                expected=expected, max_tokens=max_tokens, metadata=metadata or {}, tools=tools or [])


def local_cases(partition: str) -> list[Case]:
    cases = []
    for task, count in [("grounded-local", 64), ("tool-json-local", 64), ("context-local", 24)]:
        for i in range(count):
            key = f"{partition}-{i}"
            token = fingerprint([SEED, task, key])[:10].upper()
            value = 1000 + int(token[:4], 16) % 8000
            group = f"{task}/{key}"
            metadata = {"fixture_version": 1, "subtype": ""}
            if task == "grounded-local":
                unknown = i % 2 == 1
                prompt = f"Use only this fictional record: Project {token} uses port {value}. "
                prompt += "What is its launch date? If absent, return UNKNOWN only." if unknown else "Return its port number only."
                expected = "UNKNOWN" if unknown else str(value)
                metadata["subtype"] = "abstention" if unknown else "lookup"
                case = user_case(task, key, prompt, "exact", expected, partition=partition, max_tokens=128, metadata=metadata)
            elif task == "context-local":
                lines = 16 if i % 3 == 0 else 128 if i % 3 == 1 else 768
                records = [f"item{j}: value{j}" for j in range(lines)]
                position = [0, lines // 2, lines - 1][(i // 3) % 3]
                records[position] = f"target: {token}"
                prompt = "Read these records:\n" + "\n".join(records) + "\nReturn the target value only."
                case = user_case(task, key, prompt, "exact", token, partition=partition, max_tokens=128,
                                 metadata={"records": lines, "position": position, "subtype": "retrieval"})
                if i % 4 == 3:
                    updated = token[::-1]
                    case.messages += [{"role": "assistant", "content": token}, {"role": "user", "content": f"Update target to {updated}. Return its current value only."}]
                    case.expected = updated
                    case.metadata["subtype"] = "conversation-update"
            else:
                schema = {"type": "object", "properties": {"project": {"type": "string"}, "port": {"type": "integer"}}, "required": ["project", "port"], "additionalProperties": False}
                arguments = {"project": token, "port": value}
                tool = {"type": "function", "function": {"name": "set_port", "description": "Set a fictional project's port. Requires both project and port; do not invent missing values.", "parameters": schema}}
                if i % 4 == 0:
                    answer_schema = {**schema, "properties": {"project": {"const": token}, "port": {"const": value}}}
                    case = user_case(task, key, f"Return only JSON with project equal to {token} and port equal to {value}. No other keys or Markdown.", "json", answer_schema, partition=partition, max_tokens=128)
                    case.metadata["subtype"] = "unconstrained-json"
                else:
                    mode = ["unused", "call", "clarify", "no_tool"][i % 4]
                    if mode == "call":
                        prompt = f"Use set_port to set project {token} to port {value}."
                    elif mode == "clarify":
                        prompt = f"Set the port for project {token}. If a required value is missing, ask exactly: Which port?"
                    else:
                        prompt = f"Do not change anything. Return only this project identifier: {token}"
                    expected = {"mode": mode, "name": "set_port", "arguments": arguments, "schema": schema,
                                "answer": "Which port?" if mode == "clarify" else token}
                    case = user_case(task, key, prompt, "tool", expected, partition=partition, max_tokens=128, tools=[tool], metadata={"subtype": mode})
            case.group = group
            cases.append(case)
    return cases


def build_cases(root: Path) -> tuple[list[Case], dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    source_root = root / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    rows = {}
    provenance = {}
    for task, (repo, revision, name, license_name) in SOURCES.items():
        path = Path(hf_hub_download(repo, name, repo_type="dataset", revision=revision, local_dir=source_root / task))
        rows[task] = pq.read_table(path).to_pylist() if name.endswith(".parquet") else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        provenance[task] = {"repo": repo, "revision": revision, "file": name, "sha256": file_hash(path), "license": license_name}
    human_path = source_root / HUMANEVAL_FILE
    human_url = 'https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz'
    if not human_path.exists():
        response = httpx.get(human_url, follow_redirects=True, timeout=60)
        response.raise_for_status()
        human_path.write_bytes(gzip.decompress(response.content))
    if file_hash(human_path) != HUMANEVAL_SHA256:
        raise ValueError('Official HumanEval+ release checksum differs')
    rows['humaneval-plus'] = [json.loads(line) for line in human_path.read_text().splitlines() if line.strip()]
    provenance['humaneval-plus'] = {'url':human_url, 'revision':'v0.1.10', 'file':HUMANEVAL_FILE, 'sha256':HUMANEVAL_SHA256,
                                  'evaluator_commit':'26d6d00bb1fd0fa37f39c99d5290da67891d1c5e', 'license':'apache-2.0'}
    xstest_path = source_root / "xstest.csv"
    url = f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_REVISION}/xstest_prompts.csv"
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    if xstest_path.exists() and xstest_path.read_bytes() != response.content:
        raise ValueError("Cached XSTest content differs from pinned source")
    xstest_path.write_bytes(response.content)
    provenance["xstest"] = {"url": url, "revision": XSTEST_REVISION, "sha256": file_hash(xstest_path), "license": "cc-by-4.0"}
    cases = []
    for i, row in enumerate(ordered(rows["ifeval"], lambda r: r["key"])[:160]):
        cases.append(user_case("ifeval", row["key"], row["prompt"], "ifeval", row, partition="final" if i < 128 else "development", max_tokens=2048))
    for i, row in enumerate(ordered(rows["gsm8k"], lambda r: r["question"])[:160]):
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        cases.append(user_case("gsm8k", fingerprint(row["question"])[:16], row["question"] + "\nEnd with 'Final answer: NUMBER'.", "number", answer, partition="final" if i < 128 else "development", max_tokens=2048))
    subjects = sorted({r["category"] for r in rows["mmlu-pro"]})
    if len(subjects) != 14:
        raise ValueError("Expected 14 MMLU-Pro subjects")
    for subject in subjects:
        selected = ordered([r for r in rows["mmlu-pro"] if r["category"] == subject], lambda r: r["question_id"])[:12]
        for i, row in enumerate(selected):
            options = "\n".join(f"{chr(65+j)}. {option}" for j, option in enumerate(row["options"]))
            cases.append(user_case("mmlu-pro", row["question_id"], row["question"] + "\n" + options + "\nEnd with 'Final answer: LETTER'.", "choice", row["answer"], partition="final" if i < 10 else "development", max_tokens=2048, metadata={"subject": subject}))
    if len(rows["humaneval-plus"]) != 164:
        raise ValueError("Expected all 164 HumanEval+ tasks")
    for row in rows["humaneval-plus"]:
        cases.append(user_case("humaneval-plus", row["task_id"], "Implement this Python function. Return the complete Python source only, including the signature and needed imports.\n" + row["prompt"], "humaneval_plus", {"task_id": row["task_id"], "entry_point": row["entry_point"]}, metadata={"test_source_sha256": provenance["humaneval-plus"]["sha256"]}))
    # Align safe/contrast siblings by position within source type, including the
    # two safe discrimination/privacy categories sharing a contrast category.
    families = {"nons_group_real_discr": "discr", "real_group_nons_discr": "discr", "privacy_public": "privacy", "privacy_fictional": "privacy"}
    counts = {"homonyms": 13, "figurative_language": 12, "safe_targets": 13, "safe_contexts": 12, "definitions": 13, "historical_events": 12, "discr": 13, "privacy": 12}
    by_type = defaultdict(list)
    for row in csv.DictReader(xstest_path.open()):
        by_type[row["type"]].append(row)
    for category, values in by_type.items():
        family = families.get(category, category.removeprefix("contrast_"))
        final_indices = set(ordered(list(range(25)), lambda i: [family, i])[:counts[family]])
        if len(values) != 25:
            raise ValueError("Unexpected XSTest category size")
        for i, row in enumerate(sorted(values, key=lambda r: int(r["id"]))):
            task = "xstest-safe" if row["label"] == "safe" else "xstest-contrast"
            cases.append(user_case(task, row["id"], row["prompt"], "rubric", {"label": row["label"]}, group=f"xstest/{family}/{i}", partition="final" if i in final_indices else "development", max_tokens=512, metadata={"category": category}))
    for partition in ("development", "final"):
        cases.extend(local_cases(partition))
    return cases, provenance


def prepare(root: Path) -> dict:
    lock = root / "dataset-lock.json"
    if lock.exists():
        existing = json.loads(lock.read_text())
        if file_hash(root / "cases.jsonl") != existing["cases_sha256"]:
            raise ValueError("Frozen cases changed")
        load_cases(root / "cases.jsonl")
        return existing
    cases, sources = build_cases(root)
    final = [case for case in cases if case.partition == "final"]
    counts = {task: sum(c.task == task for c in final) for task in sorted({c.task for c in final})}
    expected_counts = {"ifeval": 128, "gsm8k": 128, "mmlu-pro": 140, "humaneval-plus": 164, "xstest-safe": 125, "xstest-contrast": 100, "grounded-local": 64, "tool-json-local": 64, "context-local": 24}
    if counts != expected_counts:
        raise ValueError(f"Core case counts differ: {counts}")
    path = root / "cases.jsonl"
    content = "".join(case.model_dump_json() + "\n" for case in sorted(cases, key=lambda c: c.id))
    if path.exists() and path.read_text() != content:
        raise ValueError("Uncommitted case file differs; inspect before replacing")
    path.write_text(content)
    load_cases(path)
    result = {"version": 1, "seed": SEED, "sources": sources, "counts_final": counts,
              "count_development": len(cases) - len(final), "cases_sha256": file_hash(path),
              "builder_sha256": file_hash(Path(__file__)), "final_test_used_for_selection": False,
              "contamination": "Public data may overlap pretraining; snapshot overlap audit pending."}
    atomic_json(lock, result)
    return result
