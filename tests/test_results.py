from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path

import duckdb
import pytest

from llm_lab.benchmark import (
    SampleResult,
    ScoreResult,
    ServerTimings,
    TokenUsage,
    summarize_samples,
)
from llm_lab.errors import BenchmarkError, IntegrityError, StoragePolicyError
from llm_lab.hashing import canonical_sha256
from llm_lab.results import (
    BUNDLE_FILES,
    ResultsStore,
    load_run_bundle,
    repair_run_bundle_permissions,
    verify_run_bundle,
    write_run_bundle,
)


def _sample(
    run_id: str,
    case_id: str,
    *,
    passed: bool = True,
    latency: float = 10,
    client_completion_tps: float | None = None,
    server_timings: ServerTimings | None = None,
    request_extension: Mapping[str, object] | None = None,
) -> SampleResult:
    return SampleResult(
        run_id=run_id,
        case_id=case_id,
        repetition=0,
        started_at="2026-01-01T00:00:00Z",
        latency_ms=latency,
        request={"model": "test", "messages": [], **(request_extension or {})},
        response={"choices": [{"message": {"content": "ok"}}]},
        output_text="ok",
        tool_names=(),
        usage=TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        scores=(ScoreResult("exact", passed, "ok", "ok" if passed else "bad"),),
        passed=passed,
        server_timings=server_timings,
        client_completion_tokens_per_second=client_completion_tps,
    )


def _request_contract(
    extension: Mapping[str, object] | None = None,
) -> dict[str, object]:
    requests = []
    for case_id in ("task-a", "task-b"):
        body: dict[str, object] = {
            "model": "$MODEL_UNDER_TEST",
            "messages": [],
        }
        body.update(extension or {})
        requests.append({"case_id": case_id, "body": body})
    definition = {
        "method": "POST",
        "endpoint": "/v1/chat/completions",
        "warmup_repetitions": 0,
        "repetitions": 1,
        "requests": requests,
    }
    return {
        "schema_version": 1,
        "sha256": canonical_sha256(definition),
        "definition": definition,
    }


def _write_bundle(
    path: Path,
    run_id: str,
    cases: tuple[str, ...],
    *,
    latency: float = 10,
    status: str = "completed",
    warmup_error_count: int = 0,
    request_extension: Mapping[str, object] | None = None,
) -> Path:
    samples = tuple(
        _sample(
            run_id,
            case,
            latency=latency,
            request_extension=request_extension,
        )
        for case in cases
    )
    suite_definition = {
        "id": "quality",
        "version": "1",
        "cases": [
            {"id": "task-a", "prompt": "fixed"},
            {"id": "task-b", "prompt": "fixed"},
        ],
    }
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "status": status,
        "model": {"id": f"model-{run_id}"},
        "artifact": {"id": "artifact-q4"},
        "deployment": {"id": "deployment"},
        "suite": {
            "id": "quality",
            "version": "1",
            "kind": "quality",
            "sha256": canonical_sha256(suite_definition),
            "definition": suite_definition,
        },
        "request_contract": _request_contract(request_extension),
        "runtime": {"backend": "mock"},
        "hardware": {"gpu": "test"},
        "sampling": {"repetitions": 1},
    }
    summary = summarize_samples(samples, warmup_error_count=warmup_error_count)
    telemetry = ({
        "timestamp": "2026-01-01T00:00:00Z",
        "available": False,
        "error": "no gpu",
    },)
    return write_run_bundle(path, run, summary=summary, samples=samples, telemetry=telemetry)


def test_bundle_has_exact_files_and_verifies_integrity(tmp_path: Path):
    path = _write_bundle(tmp_path / "run-a", "run-a", ("task-a", "task-b"))

    assert {item.name for item in path.iterdir()} == set(BUNDLE_FILES)
    verification = verify_run_bundle(path)
    assert verification.run_id == "run-a"
    assert verification.sample_count == 2
    loaded = load_run_bundle(path)
    assert loaded["summary"]["run_id"] == "run-a"
    assert len(loaded["samples"]) == 2

    with pytest.raises(BenchmarkError, match="already exists"):
        _write_bundle(path, "run-a", ("task-a",))


def test_results_store_rejects_database_symlink_without_touching_victim(
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim.duckdb"
    with ResultsStore(victim):
        pass
    before = victim.read_bytes()
    link = tmp_path / "results.duckdb"
    link.symlink_to(victim)

    with pytest.raises(StoragePolicyError, match="results database"):
        ResultsStore(link)

    assert victim.read_bytes() == before


def test_results_store_rejects_symlinked_parent_without_creating_database(
    tmp_path: Path,
) -> None:
    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(victim_dir, target_is_directory=True)

    with pytest.raises(StoragePolicyError, match="safe real directory"):
        ResultsStore(redirected / "results.duckdb")

    assert not (victim_dir / "results.duckdb").exists()


def test_results_store_rejects_symlinked_wal_without_touching_victim(
    tmp_path: Path,
) -> None:
    database = tmp_path / "results.duckdb"
    victim = tmp_path / "victim.wal"
    victim.write_bytes(b"do-not-change")
    Path(f"{database}.wal").symlink_to(victim)

    with pytest.raises(StoragePolicyError, match="results database WAL"):
        ResultsStore(database)

    assert victim.read_bytes() == b"do-not-change"
    assert not database.exists()


def test_results_store_rechecks_wal_after_database_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "results.duckdb"
    with ResultsStore(database):
        pass
    wal = Path(f"{database}.wal")
    victim = tmp_path / "victim.wal"
    victim.write_bytes(b"do-not-change")
    real_connect = duckdb.connect

    def connect(*args: object, **kwargs: object) -> duckdb.DuckDBPyConnection:
        connection = real_connect(*args, **kwargs)
        assert not wal.exists()
        wal.symlink_to(victim)
        return connection

    monkeypatch.setattr("llm_lab.results.duckdb.connect", connect)

    with pytest.raises(StoragePolicyError, match="results database WAL"):
        ResultsStore(database)

    assert victim.read_bytes() == b"do-not-change"


def test_first_database_publication_never_chmods_a_swapped_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "results.duckdb"
    displaced = tmp_path / "displaced.duckdb"
    victim = tmp_path / "victim"
    victim.write_bytes(b"do-not-change")
    victim.chmod(0o640)
    victim_mode = stat.S_IMODE(victim.stat().st_mode)
    real_stat = os.stat
    swapped = False

    def swap_after_publication_validation(path: object, *args: object, **kwargs: object):
        nonlocal swapped
        metadata = real_stat(path, *args, **kwargs)
        directory_fd = kwargs.get("dir_fd")
        if (
            not swapped
            and path == database.name
            and isinstance(directory_fd, int)
            and kwargs.get("follow_symlinks") is False
            and stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 2
        ):
            swapped = True
            os.rename(
                database.name,
                displaced.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            os.symlink(victim, database.name, dir_fd=directory_fd)
        return metadata

    monkeypatch.setattr("llm_lab.results.os.stat", swap_after_publication_validation)

    with pytest.raises(
        (BenchmarkError, StoragePolicyError), match="changed while sealing|results database"
    ):
        ResultsStore(database)

    assert swapped is True
    assert victim.read_bytes() == b"do-not-change"
    assert stat.S_IMODE(victim.stat().st_mode) == victim_mode
    assert not database.exists()


def test_append_rejects_wal_symlink_planted_after_connection(
    tmp_path: Path,
) -> None:
    bundle = _write_bundle(
        tmp_path / "run-a", "run-a", ("task-a", "task-b")
    )
    database = tmp_path / "results.duckdb"
    wal = Path(f"{database}.wal")
    victim = tmp_path / "victim.wal"
    victim.write_bytes(b"do-not-change")
    store = ResultsStore(database)
    try:
        store.connection.execute("CHECKPOINT")
        assert not wal.exists()
        wal.symlink_to(victim)

        with pytest.raises(StoragePolicyError, match="results database WAL"):
            store.append_bundle(bundle)

        wal.unlink()
        assert store.run_ids() == []
        assert victim.read_bytes() == b"do-not-change"
    finally:
        wal.unlink(missing_ok=True)
        store.close()


def test_bundle_tampering_is_detected(tmp_path: Path):
    path = _write_bundle(tmp_path / "run-a", "run-a", ("task-a",))
    (path / "samples.jsonl").chmod(0o644)
    with (path / "samples.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    (path / "samples.jsonl").chmod(0o444)

    with pytest.raises(IntegrityError, match="mismatch"):
        verify_run_bundle(path)


def test_run_metadata_tampering_is_detected(tmp_path: Path):
    path = _write_bundle(tmp_path / "run-a", "run-a", ("task-a",))
    run_path = path / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["model"]["id"] = "silently-changed"
    run_path.chmod(0o644)
    run_path.write_text(json.dumps(run), encoding="utf-8")
    run_path.chmod(0o444)

    with pytest.raises(IntegrityError, match="run metadata"):
        verify_run_bundle(path)


def test_bundle_verification_rejects_inconsistent_request_contract_digest(
    tmp_path: Path,
) -> None:
    source = _write_bundle(tmp_path / "source", "run-a", ("task-a", "task-b"))
    loaded = load_run_bundle(source)
    run = dict(loaded["run"])
    run.pop("bundle", None)
    run["request_contract"] = dict(run["request_contract"])
    run["request_contract"]["sha256"] = "0" * 64
    inconsistent = write_run_bundle(
        tmp_path / "inconsistent",
        run,
        summary=loaded["summary"],
        samples=loaded["samples"],
        telemetry=loaded["telemetry"],
    )

    with pytest.raises(IntegrityError, match="request contract digest mismatch"):
        verify_run_bundle(inconsistent)


def test_bundle_verification_binds_samples_to_effective_request_contract(
    tmp_path: Path,
) -> None:
    source = _write_bundle(tmp_path / "source", "run-a", ("task-a", "task-b"))
    loaded = load_run_bundle(source)
    run = dict(loaded["run"])
    run.pop("bundle", None)
    samples = list(loaded["samples"])
    samples[0] = dict(samples[0])
    samples[0]["request"] = {
        **samples[0]["request"],
        "unrecorded_extension": True,
    }
    inconsistent = write_run_bundle(
        tmp_path / "inconsistent",
        run,
        summary=loaded["summary"],
        samples=samples,
        telemetry=loaded["telemetry"],
    )

    with pytest.raises(IntegrityError, match="does not match.*request contract"):
        verify_run_bundle(inconsistent)


def test_bundle_verification_rejects_writable_metadata(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path / "run-a", "run-a", ("task-a",))
    (path / "run.json").chmod(0o644)

    with pytest.raises(IntegrityError, match="private regular file"):
        verify_run_bundle(path)


def test_writer_normalizes_old_bundle_metadata_and_rejects_summary_mismatch(
    tmp_path: Path,
) -> None:
    original = _write_bundle(tmp_path / "run-a", "run-a", ("task-a",))
    loaded = load_run_bundle(original)
    copied = write_run_bundle(
        tmp_path / "copy",
        loaded["run"],
        summary=loaded["summary"],
        samples=loaded["samples"],
        telemetry=loaded["telemetry"],
    )
    assert verify_run_bundle(copied).run_id == "run-a"

    bad_summary = dict(loaded["summary"])
    bad_summary["run_id"] = "wrong"
    with pytest.raises(BenchmarkError, match="summary run_id"):
        write_run_bundle(
            tmp_path / "bad",
            loaded["run"],
            summary=bad_summary,
            samples=loaded["samples"],
            telemetry=loaded["telemetry"],
        )


def test_published_bundle_is_read_only_and_loaded_from_one_verified_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bundle(tmp_path / "run-a", "run-a", ("task-a",))
    assert path.stat().st_mode & 0o222 == 0
    assert all((path / name).stat().st_mode & 0o222 == 0 for name in BUNDLE_FILES)

    # ``load_run_bundle`` must not verify and then reopen mutable pathnames.
    # Its verifier consumes the same in-memory bytes returned to the caller.
    monkeypatch.setattr(
        "llm_lab.results.verify_run_bundle",
        lambda *_args, **_kwargs: pytest.fail("separate verify/read seam used"),
    )
    loaded = load_run_bundle(path)
    assert loaded["run"]["run_id"] == "run-a"


def test_explicit_bundle_permission_repair_seals_verified_legacy_bundle(
    tmp_path: Path,
) -> None:
    path = _write_bundle(tmp_path / "run-a", "run-a", ("task-a",))
    path.chmod(0o755)
    for name in BUNDLE_FILES:
        (path / name).chmod(0o644)

    with pytest.raises(IntegrityError, match="writable|private regular file"):
        verify_run_bundle(path)

    repaired = repair_run_bundle_permissions(path)

    assert repaired.run_id == "run-a"
    assert path.stat().st_mode & 0o222 == 0
    assert all((path / name).stat().st_mode & 0o222 == 0 for name in BUNDLE_FILES)
    assert verify_run_bundle(path) == repaired


def test_two_results_connections_share_new_database_and_persist_both_runs(
    tmp_path: Path,
) -> None:
    first_bundle = _write_bundle(
        tmp_path / "run-a", "run-a", ("task-a", "task-b")
    )
    second_bundle = _write_bundle(
        tmp_path / "run-b", "run-b", ("task-a", "task-b")
    )
    database = tmp_path / "results.duckdb"

    first = ResultsStore(database)
    second = ResultsStore(database)
    try:
        first.append_bundle(first_bundle)
        second.append_bundle(second_bundle)
    finally:
        second.close()
        first.close()

    with ResultsStore(database) as reopened:
        assert reopened.run_ids() == ["run-a", "run-b"]


def test_duckdb_append_and_compare_suppresses_mismatched_task_composite(tmp_path: Path):
    first = _write_bundle(tmp_path / "run-a", "run-a", ("task-a", "task-b"), latency=12)
    second = _write_bundle(tmp_path / "run-b", "run-b", ("task-a", "task-b"), latency=8)
    missing = _write_bundle(tmp_path / "run-c", "run-c", ("task-a",), latency=5)

    with ResultsStore(tmp_path / "results.duckdb") as store:
        assert store.append_bundle(first) == "run-a"
        store.index_bundle(second)
        store.append_bundle(missing)
        assert store.run_ids() == ["run-a", "run-b", "run-c"]

        complete = store.compare_runs(("run-a", "run-b"))
        assert complete["composite"] is not None
        b_metrics = next(
            item for item in complete["composite"]["runs"] if item["run_id"] == "run-b"
        )
        assert b_metrics["mean_task_latency_ms"] == 8

        incomparable = store.compare_runs(("run-a", "run-c"))
        assert incomparable["composite"] is None
        assert "same task set" in incomparable["composite_unavailable_reason"]
        task_b = next(task for task in incomparable["tasks"] if task["case_id"] == "task-b")
        assert not task_b["present_in_all_runs"]

        with pytest.raises(BenchmarkError, match="already indexed"):
            store.append_bundle(first)


def test_compare_suppresses_composite_when_both_runs_omit_a_suite_task(
    tmp_path: Path,
) -> None:
    first = _write_bundle(tmp_path / "run-a", "run-a", ("task-a",))
    second = _write_bundle(tmp_path / "run-b", "run-b", ("task-a",))

    with ResultsStore(tmp_path / "results.duckdb") as store:
        store.append_bundle(first)
        store.append_bundle(second)
        comparison = store.compare_runs(("run-a", "run-b"))

    assert comparison["composite"] is None
    assert "skipped or omitted" in comparison["composite_unavailable_reason"]


def test_compare_suppresses_composite_for_non_completed_run(tmp_path: Path) -> None:
    first = _write_bundle(
        tmp_path / "run-a", "run-a", ("task-a", "task-b")
    )
    original_second = _write_bundle(
        tmp_path / "run-b-source", "run-b", ("task-a", "task-b")
    )
    loaded = load_run_bundle(original_second)
    run = dict(loaded["run"])
    run.pop("bundle", None)
    run["status"] = "failed"
    second = write_run_bundle(
        tmp_path / "run-b",
        run,
        summary=loaded["summary"],
        samples=loaded["samples"],
        telemetry=loaded["telemetry"],
    )

    with ResultsStore(tmp_path / "results.duckdb") as store:
        store.append_bundle(first)
        store.append_bundle(second)
        comparison = store.compare_runs(("run-a", "run-b"))

    assert comparison["composite"] is None
    assert "did not complete cleanly" in comparison["composite_unavailable_reason"]


def test_compare_suppresses_legacy_completed_run_with_warmup_errors(
    tmp_path: Path,
) -> None:
    legacy_buggy = _write_bundle(
        tmp_path / "run-a",
        "run-a",
        ("task-a", "task-b"),
        status="completed",
        warmup_error_count=1,
    )
    clean = _write_bundle(
        tmp_path / "run-b", "run-b", ("task-a", "task-b")
    )

    with ResultsStore(tmp_path / "results.duckdb") as store:
        store.append_bundle(legacy_buggy)
        store.append_bundle(clean)
        comparison = store.compare_runs(("run-a", "run-b"))

    assert comparison["composite"] is None
    assert "warmup errors" in comparison["composite_unavailable_reason"]


def test_compare_suppresses_legacy_run_without_warmup_accounting(
    tmp_path: Path,
) -> None:
    clean = _write_bundle(
        tmp_path / "run-a", "run-a", ("task-a", "task-b")
    )
    legacy_source = _write_bundle(
        tmp_path / "legacy-source", "run-b", ("task-a", "task-b")
    )
    loaded = load_run_bundle(legacy_source)
    run = dict(loaded["run"])
    run.pop("bundle", None)
    summary = dict(loaded["summary"])
    summary.pop("warmup_error_count")
    legacy = write_run_bundle(
        tmp_path / "run-b",
        run,
        summary=summary,
        samples=loaded["samples"],
        telemetry=loaded["telemetry"],
    )

    with ResultsStore(tmp_path / "results.duckdb") as store:
        store.append_bundle(clean)
        store.append_bundle(legacy)
        comparison = store.compare_runs(("run-a", "run-b"))

    assert comparison["composite"] is None
    assert "warmup error accounting" in comparison["composite_unavailable_reason"]


def test_compare_requires_same_effective_request_contract(tmp_path: Path) -> None:
    first = _write_bundle(
        tmp_path / "run-a",
        "run-a",
        ("task-a", "task-b"),
        request_extension={"min_p": 0.1},
    )
    second = _write_bundle(
        tmp_path / "run-b",
        "run-b",
        ("task-a", "task-b"),
        request_extension={"min_p": 0.2},
    )

    with ResultsStore(tmp_path / "results.duckdb") as store:
        store.append_bundle(first)
        store.append_bundle(second)
        comparison = store.compare_runs(("run-a", "run-b"))

    assert comparison["composite"] is None
    assert "effective request contract" in comparison["composite_unavailable_reason"]


def test_compare_safely_suppresses_bundle_without_request_contract(
    tmp_path: Path,
) -> None:
    modern = _write_bundle(
        tmp_path / "run-a", "run-a", ("task-a", "task-b")
    )
    legacy_source = _write_bundle(
        tmp_path / "legacy-source", "run-b", ("task-a", "task-b")
    )
    loaded = load_run_bundle(legacy_source)
    run = dict(loaded["run"])
    run.pop("bundle", None)
    run.pop("request_contract")
    legacy = write_run_bundle(
        tmp_path / "run-b",
        run,
        summary=loaded["summary"],
        samples=loaded["samples"],
        telemetry=loaded["telemetry"],
    )

    with ResultsStore(tmp_path / "results.duckdb") as store:
        store.append_bundle(modern)
        store.append_bundle(legacy)
        comparison = store.compare_runs(("run-a", "run-b"))

    assert comparison["composite"] is None
    assert "verifiable effective request contract" in comparison[
        "composite_unavailable_reason"
    ]


def test_compare_suppresses_same_named_suites_with_different_content(tmp_path: Path):
    first = _write_bundle(tmp_path / "run-a", "run-a", ("task-a",))
    second = _write_bundle(tmp_path / "run-b", "run-b", ("task-a",))
    loaded = load_run_bundle(second)
    run = loaded["run"]
    # ``write_run_bundle`` creates a fresh self-manifest.  Do not carry the
    # source bundle's hashes into the replacement metadata.
    run.pop("bundle", None)
    run["suite"]["definition"]["cases"][0]["prompt"] = "different"
    run["suite"]["sha256"] = canonical_sha256(run["suite"]["definition"])
    # Rebuild rather than tamper so both bundles remain individually valid.
    replacement = tmp_path / "replacement"
    write_run_bundle(
        replacement,
        run,
        summary=loaded["summary"],
        samples=loaded["samples"],
        telemetry=loaded["telemetry"],
    )

    with ResultsStore(tmp_path / "results.duckdb") as store:
        store.append_bundle(first)
        store.append_bundle(replacement)
        comparison = store.compare_runs(("run-a", "run-b"))

    assert comparison["composite"] is None
    assert "identical suite content" in comparison["composite_unavailable_reason"]


def test_duckdb_indexes_and_compares_explicit_performance_metrics(tmp_path: Path):
    run_id = "run-performance"
    sample = _sample(
        run_id,
        "task-a",
        client_completion_tps=20.0,
        server_timings=ServerTimings(
            prompt_tokens=2,
            prompt_ms=40.0,
            prompt_per_token_ms=20.0,
            prompt_tokens_per_second=50.0,
            predicted_tokens=1,
            predicted_ms=25.0,
            predicted_per_token_ms=25.0,
            predicted_tokens_per_second=40.0,
            raw={"prompt_n": 2, "predicted_n": 1, "cache_n": 1},
        ),
    )
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "status": "completed",
        "model": {"id": "model-performance"},
        "artifact": {"id": "artifact-q4"},
        "deployment": {"id": "deployment"},
        "suite": {"id": "quality", "version": "1", "kind": "performance"},
    }
    bundle = write_run_bundle(
        tmp_path / run_id,
        run,
        summary=summarize_samples((sample,)),
        samples=(sample,),
        telemetry=(),
    )

    with ResultsStore(tmp_path / "results.duckdb") as store:
        store.append_bundle(bundle)
        row = store.compare_rows((run_id,))[0]
        stored = store.connection.execute(
            """
            SELECT client_completion_tokens_per_second,
                   server_prompt_tokens_per_second,
                   server_predicted_tokens_per_second,
                   server_timings_json
            FROM samples
            """
        ).fetchone()

    assert stored[:3] == (20.0, 50.0, 40.0)
    assert json.loads(stored[3])["raw"]["cache_n"] == 1
    assert row["client_completion_tokens_per_second_sample_count"] == 1
    assert row["mean_client_completion_tokens_per_second"] == 20.0
    assert row["p50_client_completion_tokens_per_second"] == 20.0
    assert row["p95_client_completion_tokens_per_second"] == 20.0
    assert row["mean_server_prompt_ms"] == 40.0
    assert row["mean_server_prompt_tokens_per_second"] == 50.0
    assert row["mean_server_predicted_ms"] == 25.0
    assert row["mean_server_predicted_tokens_per_second"] == 40.0


def test_legacy_bundle_without_performance_fields_remains_compatible(tmp_path: Path):
    run_id = "run-legacy"
    sample = _sample(run_id, "task-a").to_dict()
    sample.pop("server_timings")
    sample.pop("client_completion_tokens_per_second")
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "status": "completed",
        "model": {"id": "legacy-model"},
        "suite": {"id": "quality", "version": "1", "kind": "quality"},
    }
    summary = summarize_samples((_sample(run_id, "task-a"),))
    summary.pop("performance")
    summary["per_case"]["task-a"].pop("performance")
    bundle = write_run_bundle(
        tmp_path / run_id,
        run,
        summary=summary,
        samples=(sample,),
        telemetry=(),
    )

    assert verify_run_bundle(bundle).sample_count == 1
    with ResultsStore(tmp_path / "results.duckdb") as store:
        store.append_bundle(bundle)
        row = store.compare_rows((run_id,))[0]

    assert row["client_completion_tokens_per_second_sample_count"] == 0
    assert row["mean_client_completion_tokens_per_second"] is None
    assert row["server_prompt_tokens_per_second_sample_count"] == 0
    assert row["mean_server_prompt_tokens_per_second"] is None


def test_existing_results_database_is_migrated_without_rewriting_rows(tmp_path: Path):
    database = tmp_path / "legacy.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute("""
        CREATE TABLE samples (
            run_id VARCHAR NOT NULL,
            case_id VARCHAR NOT NULL,
            repetition INTEGER NOT NULL,
            started_at VARCHAR,
            latency_ms DOUBLE,
            passed BOOLEAN,
            prompt_tokens BIGINT,
            completion_tokens BIGINT,
            total_tokens BIGINT,
            error_type VARCHAR,
            error_message VARCHAR,
            request_json VARCHAR NOT NULL,
            response_json VARCHAR,
            scores_json VARCHAR NOT NULL,
            PRIMARY KEY (run_id, case_id, repetition)
        )
    """)
    connection.execute(
        """INSERT INTO samples VALUES (
            'old-run', 'task-a', 0, NULL, 10.0, true, 2, 1, 3,
            NULL, NULL, '{}', NULL, '[]'
        )"""
    )
    connection.execute("CREATE INDEX samples_case_idx ON samples(case_id)")
    connection.close()

    with ResultsStore(database) as store:
        columns = {
            row[1]
            for row in store.connection.execute(
                "PRAGMA table_info('samples')"
            ).fetchall()
        }
        migrated = store.connection.execute(
            """
            SELECT client_completion_tokens_per_second,
                   server_prompt_tokens_per_second
            FROM samples WHERE run_id = 'old-run'
            """
        ).fetchone()

    assert "server_timings_json" in columns
    assert migrated == (None, None)
