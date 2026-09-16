"""Audit frozen benchmark prompts against documents admitted to FineWeb training.

This is a high-precision lexical audit. It reports evidence and never rewrites the
already frozen benchmark selection. Prompts shorter than five words are recorded
as unassessed because lexical matching would be too ambiguous.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path

import pyarrow.parquet as pq


WORDS = re.compile(r"\w+")
ANCHOR_WORDS = 13


def normalize(text: str) -> str:
    """Exactly match the normalization used by training/prepare.py."""
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\x00", "")).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def word_tokens(text: str) -> list[str]:
    return WORDS.findall(normalize(text).casefold())


def shingles(words: list[str], width: int = ANCHOR_WORDS) -> set[tuple[str, ...]]:
    return {tuple(words[index:index + width]) for index in range(len(words) - width + 1)}


def audit(root: Path, training_run: Path) -> dict:
    cases_path = root / "cases.jsonl"
    dataset = training_run / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text())
    accepted = {}
    with (dataset / "documents.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            accepted[record["id"]] = record

    cases = []
    eligibility = collections.Counter()
    by_task = collections.Counter()
    unassessed = []
    anchors: dict[str, list[tuple[tuple[str, ...], int]]] = collections.defaultdict(list)
    with cases_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["partition"] != "final":
                continue
            query = normalize("\n".join(
                message["content"] for message in record["messages"] if message["role"] == "user"
            ))
            words = word_tokens(query)
            item = {"id": record["id"], "task": record["task"], "query": query, "words": words}
            cases.append(item)
            by_task[record["task"]] += 1
            if len(words) < 5:
                eligibility[(record["task"], "unassessed_short")] += 1
                unassessed.append({"case_id": record["id"], "task": record["task"], "query_words": len(words)})
                continue
            eligibility[(record["task"], "assessed")] += 1
            width = min(ANCHOR_WORDS, len(words))
            starts = sorted({0, (len(words) - width) // 2, len(words) - width})
            for start in starts:
                anchor = tuple(words[start:start + width])
                anchors[anchor[0]].append((anchor[1:], len(cases) - 1))

    findings: dict[int, dict] = {}
    raw_scanned = 0
    accepted_scanned = collections.Counter()
    seen_accepted: set[str] = set()
    expected_raw = manifest["stats"]["scanned_documents"]
    started = time.monotonic()
    for source_path in manifest["source"]["paths"]:
        for batch in pq.ParquetFile(source_path).iter_batches(batch_size=128, columns=["text", "url"]):
            for row in batch.to_pylist():
                if raw_scanned >= expected_raw:
                    break
                raw_scanned += 1
                text = normalize(row["text"] or "")
                document_id = hashlib.sha256(text.casefold().encode()).hexdigest()
                admitted = accepted.get(document_id)
                if admitted is None:
                    continue
                # The raw prefix contains four exact duplicates. Preparation
                # admits the first instance only, so reproduce that identity
                # boundary rather than counting duplicate source rows twice.
                if document_id in seen_accepted:
                    continue
                seen_accepted.add(document_id)
                accepted_scanned[admitted["split"]] += 1
                if admitted["split"] != "train":
                    continue
                words = word_tokens(text)
                candidates = set()
                for position, first in enumerate(words):
                    for rest, case_index in anchors.get(first, ()):
                        end = position + 1 + len(rest)
                        if end <= len(words) and tuple(words[position + 1:end]) == rest:
                            candidates.add(case_index)
                if not candidates:
                    continue
                document_shingles = None
                for case_index in candidates:
                    case = cases[case_index]
                    query_words = case["words"]
                    width = min(ANCHOR_WORDS, len(query_words))
                    if width == ANCHOR_WORDS:
                        if document_shingles is None:
                            document_shingles = shingles(words)
                        query_shingles = shingles(query_words)
                        matches = len(query_shingles & document_shingles)
                        ratio = matches / len(query_shingles)
                    else:
                        query_shingles = {tuple(query_words)}
                        matches = int(any(tuple(words[i:i + width]) == tuple(query_words)
                                          for i in range(len(words) - width + 1)))
                        ratio = float(matches)
                    previous = findings.get(case_index)
                    if previous is None or ratio > previous["shingle_coverage"]:
                        findings[case_index] = {
                            "case_id": case["id"],
                            "task": case["task"],
                            "training_document_id": document_id,
                            "training_document_url": admitted.get("url"),
                            "query_words": len(query_words),
                            "shingle_width": width,
                            "matching_unique_shingles": matches,
                            "query_unique_shingles": len(query_shingles),
                            "shingle_coverage": ratio,
                            "exact_normalized_text": case["query"].casefold() in text.casefold(),
                        }
            if raw_scanned >= expected_raw:
                break
        if raw_scanned >= expected_raw:
            break

    expected_accepted = sum(manifest["stats"][f"{split}_documents"] for split in ("train", "validation", "test"))
    if raw_scanned != expected_raw or sum(accepted_scanned.values()) != expected_accepted:
        raise RuntimeError("Raw FineWeb scan did not reproduce the frozen accepted-document set")
    suspects = sorted(
        (finding for finding in findings.values()
         if finding["exact_normalized_text"] or finding["shingle_coverage"] >= 0.5),
        key=lambda item: (-item["shingle_coverage"], item["case_id"]),
    )
    anchored_matches = sorted(
        findings.values(), key=lambda item: (-item["shingle_coverage"], item["case_id"])
    )
    assessed = sum(value for (task, state), value in eligibility.items() if state == "assessed")
    result = {
        "schema_version": 1,
        "method": {
            "name": "per-document lexical anchor and 13-word shingle overlap",
            "normalization": "FineWeb preparation whitespace/NUL normalization plus Unicode casefold for identity",
            "anchors_per_prompt": "first, middle, and last; deduplicated",
            "suspect_threshold": "exact normalized prompt containment or at least 0.5 unique 13-word shingle coverage",
            "minimum_assessed_prompt_words": 5,
            "limitations": [
                "Does not detect semantic paraphrases or contamination already present in the original Qwen pretraining corpus.",
                "Prompts shorter than five words are unassessed because lexical matches are ambiguous.",
                "Absence of a suspect is evidence against direct copying in this FineWeb subset, not proof of zero contamination.",
            ],
        },
        "implementation_sha256": sha256(Path(__file__)),
        "cases_sha256": sha256(cases_path),
        "training_dataset_snapshot": manifest["snapshot_id"],
        "fineweb_revision": manifest["source"]["revision"],
        "training_documents_scanned": accepted_scanned["train"],
        "validation_documents_scanned": accepted_scanned["validation"],
        "test_documents_scanned": accepted_scanned["test"],
        "raw_documents_scanned": raw_scanned,
        "final_cases": sum(by_task.values()),
        "assessed_cases": assessed,
        "unassessed_short_cases": sum(by_task.values()) - assessed,
        "unassessed": sorted(unassessed, key=lambda item: item["case_id"]),
        "eligibility_by_task": {
            task: {
                "total": by_task[task],
                "assessed": eligibility[(task, "assessed")],
                "unassessed_short": eligibility[(task, "unassessed_short")],
            }
            for task in sorted(by_task)
        },
        "suspect_count": len(suspects),
        "suspects": suspects,
        "anchored_match_count": len(anchored_matches),
        "anchored_matches": anchored_matches,
        "elapsed_seconds": time.monotonic() - started,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("training_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.training_run)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in (
        "final_cases", "assessed_cases", "unassessed_short_cases", "suspect_count", "elapsed_seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
