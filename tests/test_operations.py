from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from llm_lab.errors import StoragePolicyError
from llm_lab.operations import (
    EventBroker,
    IdempotencyConflictError,
    InvalidOperationTransitionError,
    OperationBusyError,
    OperationConflictError,
    OperationFailure,
    OperationNotFoundError,
    OperationStatus,
    OperationStore,
    OperationStoreError,
    SafeOperationError,
    sanitize_failure,
)


class AdvancingClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
        self.lock = threading.Lock()

    def __call__(self) -> datetime:
        with self.lock:
            value = self.current
            self.current += timedelta(seconds=1)
            return value


def test_statuses_have_stable_activity_semantics() -> None:
    assert {status.value for status in OperationStatus} == {
        "queued",
        "running",
        "cancelling",
        "succeeded",
        "failed",
        "cancelled",
        "interrupted",
    }
    assert OperationStatus.QUEUED.active
    assert OperationStatus.RUNNING.active
    assert OperationStatus.CANCELLING.active
    assert OperationStatus.SUCCEEDED.terminal
    assert OperationStatus.FAILED.terminal


def test_admit_persists_a_frozen_canonical_record_and_event(tmp_path: Path) -> None:
    broker = EventBroker(stream_id="a" * 32)
    request = {"deployment_id": "local-fast", "options": {"tokens": [8, 16]}}
    with OperationStore(tmp_path / "operations.sqlite", broker=broker) as store:
        admission = store.admit(
            "runtime.activate",
            request=request,
            idempotency_key="activate-001",
        )
        record = admission.operation

        assert not admission.replayed
        assert record.status is OperationStatus.QUEUED
        assert record.mutating
        assert record.idempotency_key == "activate-001"
        assert len(record.request_sha256) == 64
        assert record.request["deployment_id"] == "local-fast"
        assert record.to_dict()["request"] == request
        assert "idempotency_key" not in record.to_dict()
        assert record.to_dict(include_idempotency_key=True)["idempotency_key"] == (
            "activate-001"
        )
        assert record.revision == 1
        assert record.started_at is None
        assert record.finished_at is None
        assert store.get(record.operation_id) == record

        with pytest.raises(TypeError):
            record.request["changed"] = True  # type: ignore[index]
        with pytest.raises(TypeError):
            record.request["options"]["tokens"] = ()  # type: ignore[index]
        with pytest.raises(FrozenInstanceError):
            record.kind = "changed"  # type: ignore[misc]

    replay = broker.replay()
    assert [event.event_type for event in replay.events] == ["operation.queued"]
    assert replay.events[0].operation_id == record.operation_id
    assert "request" not in replay.events[0].data


def test_idempotent_admission_replays_without_duplicate_event(tmp_path: Path) -> None:
    broker = EventBroker()
    with OperationStore(tmp_path / "operations.sqlite", broker=broker) as store:
        first = store.admit(
            "benchmark.run",
            request={"suite_id": "smoke", "deployment_id": "local-fast"},
            idempotency_key="same-request",
        )
        second = store.admit(
            "benchmark.run",
            request={"deployment_id": "local-fast", "suite_id": "smoke"},
            idempotency_key="same-request",
        )

        assert not first.replayed
        assert second.replayed
        assert second.operation == first.operation
        assert len(store.list()) == 1
        assert len(broker.replay().events) == 1


@pytest.mark.parametrize(
    ("kind", "payload", "mutating"),
    (
        ("runtime.stop", {"deployment_id": "local-fast"}, True),
        ("runtime.activate", {"deployment_id": "local-general"}, True),
        ("runtime.activate", {"deployment_id": "local-fast"}, False),
    ),
)
def test_idempotency_key_cannot_be_rebound(
    tmp_path: Path,
    kind: str,
    payload: dict[str, Any],
    mutating: bool,
) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        store.create(
            "runtime.activate",
            request={"deployment_id": "local-fast"},
            idempotency_key="one-key",
        )
        with pytest.raises(IdempotencyConflictError, match="another request"):
            store.create(
                kind,
                request=payload,
                idempotency_key="one-key",
                mutating=mutating,
            )


def test_single_mutation_admission_and_terminal_release(tmp_path: Path) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        active = store.create("runtime.activate", request={"deployment_id": "a"})
        with pytest.raises(OperationBusyError) as caught:
            store.create("benchmark.run", request={"suite_id": "smoke"})
        assert caught.value.active_operation == active

        # Read-only inspection jobs do not consume the single mutation slot.
        first_read = store.create("artifact.verify", mutating=False)
        second_read = store.create("storage.report", mutating=False)
        assert first_read.operation_id != second_read.operation_id

        store.start(active.operation_id)
        store.succeed(active.operation_id, result={"deployment_id": "a"})
        replacement = store.create("benchmark.run", request={"suite_id": "smoke"})
        assert replacement.status is OperationStatus.QUEUED


def test_cross_connection_mutation_admission_is_atomic(tmp_path: Path) -> None:
    database = tmp_path / "operations.sqlite"
    first = OperationStore(database)
    second = OperationStore(database)
    barrier = threading.Barrier(2)

    def admit(store: OperationStore, key: str) -> tuple[str, str]:
        barrier.wait(timeout=2)
        try:
            record = store.create(
                "runtime.activate",
                request={"deployment_id": key},
                idempotency_key=key,
            )
            return "admitted", record.operation_id
        except OperationBusyError as exc:
            return "busy", exc.active_operation.operation_id

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(
                executor.map(admit, (first, second), ("request-a", "request-b"))
            )
        assert sorted(outcome for outcome, _ in outcomes) == ["admitted", "busy"]
        assert outcomes[0][1] == outcomes[1][1]
        assert len(first.list()) == 1
    finally:
        second.close()
        first.close()


def test_cross_connection_idempotent_admission_converges(tmp_path: Path) -> None:
    database = tmp_path / "operations.sqlite"
    first = OperationStore(database)
    second = OperationStore(database)
    barrier = threading.Barrier(2)

    def admit(store: OperationStore) -> tuple[str, bool]:
        barrier.wait(timeout=2)
        result = store.admit(
            "runtime.activate",
            request={"deployment_id": "local-fast"},
            idempotency_key="shared-key",
        )
        return result.operation.operation_id, result.replayed

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(admit, (first, second)))
        assert outcomes[0][0] == outcomes[1][0]
        assert sorted(replayed for _, replayed in outcomes) == [False, True]
    finally:
        second.close()
        first.close()


def test_full_lifecycle_timestamps_progress_and_idempotent_completion(
    tmp_path: Path,
) -> None:
    clock = AdvancingClock()
    broker = EventBroker(stream_id="b" * 32)
    with OperationStore(
        tmp_path / "operations.sqlite", broker=broker, clock=clock
    ) as store:
        queued = store.create("benchmark.run", request={"suite_id": "smoke"})
        running = store.start(queued.operation_id)
        progressed = store.set_progress(
            queued.operation_id, {"phase": "warmup", "completed": 1, "total": 3}
        )
        succeeded = store.succeed(
            queued.operation_id, result={"run_id": "run-001"}
        )
        repeated = store.succeed(
            queued.operation_id, result={"run_id": "run-001"}
        )

        assert queued.created_at == datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
        assert running.started_at == datetime(2026, 9, 6, 12, 0, 1, tzinfo=UTC)
        assert progressed.progress["phase"] == "warmup"
        assert succeeded.status is OperationStatus.SUCCEEDED
        assert succeeded.result == {"run_id": "run-001"}
        assert succeeded.finished_at == datetime(2026, 9, 6, 12, 0, 3, tzinfo=UTC)
        assert [queued.revision, running.revision, progressed.revision, succeeded.revision] == [
            1,
            2,
            3,
            4,
        ]
        assert repeated == succeeded
        with pytest.raises(OperationConflictError, match="different result"):
            store.succeed(queued.operation_id, result={"run_id": "run-002"})

    assert [event.event_type for event in broker.replay().events] == [
        "operation.queued",
        "operation.running",
        "operation.progress",
        "operation.succeeded",
    ]


def test_transition_validation_and_missing_records(tmp_path: Path) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        queued = store.create("runtime.activate")
        with pytest.raises(InvalidOperationTransitionError, match="queued to succeeded"):
            store.succeed(queued.operation_id)
        with pytest.raises(InvalidOperationTransitionError, match="progress"):
            store.set_progress(queued.operation_id, {"phase": "invalid"})

        running = store.start(queued.operation_id)
        assert store.start(queued.operation_id) == running
        failed = store.fail(queued.operation_id)
        assert store.fail(queued.operation_id) == failed
        with pytest.raises(InvalidOperationTransitionError, match="failed to running"):
            store.start(queued.operation_id)

        with pytest.raises(OperationNotFoundError, match="not found"):
            store.get("op_" + "0" * 32)
        assert store.find("op_" + "0" * 32) is None
        with pytest.raises(ValueError, match="invalid operation id"):
            store.get("../../active.json")


def test_unexpected_exception_text_is_never_persisted(tmp_path: Path) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        record = store.create("runtime.activate")
        record = store.start(record.operation_id)
        record = store.fail(
            record.operation_id,
            RuntimeError(
                "password=hunter2 token=abcd /home/alice/private model failed"
            ),
        )

        assert record.error == OperationFailure(
            code="operation_failed",
            message="The operation could not be completed.",
            retryable=False,
        )
        stored = sqlite3.connect(store.path).execute(
            "SELECT error_message FROM operations WHERE operation_id = ?",
            (record.operation_id,),
        ).fetchone()[0]
        assert "hunter2" not in stored
        assert "/home/alice" not in stored


def test_explicit_public_failure_is_bounded_and_redacted(tmp_path: Path) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        first = store.create("artifact.verify", mutating=False)
        first = store.start(first.operation_id)
        failed = store.fail(
            first.operation_id,
            code="verification_failed",
            message=(
                "Cannot read /mnt/models/a.gguf; password=hunter2; "
                "Bearer abc.def; https://alice:pw@example.test/private"
            ),
            retryable=True,
        )
        assert failed.error is not None
        assert failed.error.code == "verification_failed"
        assert failed.error.retryable
        assert "hunter2" not in failed.error.message
        assert "abc.def" not in failed.error.message
        assert "/mnt/models" not in failed.error.message
        assert "alice:pw" not in failed.error.message
        assert "[redacted]" in failed.error.message
        assert "[path]" in failed.error.message

        second = store.create("storage.report", mutating=False)
        second = store.start(second.operation_id)
        safe = store.fail(
            second.operation_id,
            SafeOperationError(
                "storage_busy", "Storage verification is already running.", retryable=True
            ),
        )
        assert safe.error == OperationFailure(
            "storage_busy", "Storage verification is already running.", True
        )


def test_sanitize_failure_rejects_ambiguous_or_non_exception_input() -> None:
    with pytest.raises(ValueError, match="either error"):
        sanitize_failure(RuntimeError("boom"), code="bad")
    with pytest.raises(TypeError, match="must be an exception"):
        sanitize_failure("raw secret")  # type: ignore[arg-type]
    sanitized = sanitize_failure(
        OperationFailure("NOT VALID", "token=secret\x00\n" + "x" * 600)
    )
    assert sanitized.code == "operation_failed"
    assert "secret" not in sanitized.message
    assert "\n" not in sanitized.message
    assert len(sanitized.message) == 500


def test_cancellation_does_not_release_mutation_until_executor_confirms(
    tmp_path: Path,
) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        record = store.create("runtime.activate")
        record = store.start(record.operation_id)
        cancelling = store.request_cancel(record.operation_id)
        assert cancelling.status is OperationStatus.CANCELLING
        assert cancelling.finished_at is None
        with pytest.raises(OperationBusyError):
            store.create("runtime.stop")

        cancelled = store.mark_cancelled(record.operation_id)
        assert cancelled.status is OperationStatus.CANCELLED
        assert cancelled.finished_at is not None
        assert store.mark_cancelled(record.operation_id) == cancelled
        assert store.create("runtime.stop").status is OperationStatus.QUEUED


def test_queued_cancellation_is_immediately_terminal(tmp_path: Path) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        queued = store.create("runtime.activate")
        cancelled = store.request_cancel(queued.operation_id)
        assert cancelled.status is OperationStatus.CANCELLED
        assert cancelled.started_at is None
        assert cancelled.finished_at is not None
        assert store.request_cancel(queued.operation_id) == cancelled


def test_restart_reconciliation_is_explicit_durable_and_idempotent(
    tmp_path: Path,
) -> None:
    database = tmp_path / "operations.sqlite"
    first = OperationStore(database)
    queued_read = first.create("artifact.verify", mutating=False)
    running_mutation = first.create("runtime.activate")
    first.start(running_mutation.operation_id)
    first.close()

    broker = EventBroker()
    with OperationStore(database, broker=broker) as reopened:
        # Merely opening a query handle never declares another controller dead.
        assert reopened.get(running_mutation.operation_id).status is OperationStatus.RUNNING
        reconciled = reopened.reconcile_incomplete()
        assert {item.operation_id for item in reconciled} == {
            queued_read.operation_id,
            running_mutation.operation_id,
        }
        assert all(item.status is OperationStatus.INTERRUPTED for item in reconciled)
        assert all(item.error and item.error.code == "service_restarted" for item in reconciled)
        assert all(item.error and item.error.retryable for item in reconciled)
        assert reopened.reconcile_incomplete() == ()
        assert reopened.active_mutation() is None

    with OperationStore(database) as final:
        assert final.get(running_mutation.operation_id).status is OperationStatus.INTERRUPTED
    assert [event.event_type for event in broker.replay().events] == [
        "operation.interrupted",
        "operation.interrupted",
    ]


def test_claim_next_is_atomic_across_connections(tmp_path: Path) -> None:
    database = tmp_path / "operations.sqlite"
    first = OperationStore(database)
    jobs = [
        first.create("artifact.verify", request={"number": number}, mutating=False)
        for number in range(2)
    ]
    second = OperationStore(database)
    barrier = threading.Barrier(2)

    def claim(store: OperationStore) -> str:
        barrier.wait(timeout=2)
        record = store.claim_next(kind="artifact.verify")
        assert record is not None
        return record.operation_id

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            claimed = list(executor.map(claim, (first, second)))
        assert set(claimed) == {job.operation_id for job in jobs}
        assert first.claim_next() is None
    finally:
        second.close()
        first.close()


def test_list_filters_status_mutability_and_limit(tmp_path: Path) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        mutation = store.create("runtime.activate")
        read = store.create("artifact.verify", mutating=False)
        store.start(read.operation_id)
        store.succeed(read.operation_id)

        assert store.list(statuses=["queued"]) == (mutation,)
        assert [item.operation_id for item in store.list(mutating=False)] == [
            read.operation_id
        ]
        assert store.list(statuses=[]) == ()
        assert len(store.list(limit=1)) == 1
        with pytest.raises(ValueError, match="unknown operation status"):
            store.list(statuses=["unknown"])
        with pytest.raises(ValueError, match="between 1 and 500"):
            store.list(limit=501)


@pytest.mark.parametrize(
    "bad_kind",
    ("", "Runtime.Activate", ".activate", "runtime/activate", "a" * 101),
)
def test_operation_kinds_are_data_identifiers_not_commands(
    tmp_path: Path, bad_kind: str
) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        with pytest.raises(ValueError, match="operation kind"):
            store.create(bad_kind)


@pytest.mark.parametrize(
    "bad_key", ("", "contains a space", "line\nbreak", "x" * 201)
)
def test_idempotency_keys_are_bounded_header_values(
    tmp_path: Path, bad_key: str
) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        with pytest.raises(ValueError, match="idempotency key"):
            store.create("runtime.activate", idempotency_key=bad_key)


def test_only_bounded_strict_json_can_be_persisted(tmp_path: Path) -> None:
    with OperationStore(tmp_path / "operations.sqlite") as store:
        with pytest.raises(TypeError, match="JSON object"):
            store.create("artifact.verify", request=[])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-finite"):
            store.create("artifact.verify", request={"value": float("nan")})
        with pytest.raises(TypeError, match="only JSON"):
            store.create("artifact.verify", request={"path": Path("/tmp/model")})
        with pytest.raises(TypeError, match="keys must be strings"):
            store.create("artifact.verify", request={1: "bad"})  # type: ignore[dict-item]
        cyclic: dict[str, Any] = {}
        cyclic["self"] = cyclic
        with pytest.raises(ValueError, match="cycle"):
            store.create("artifact.verify", request=cyclic)
        deeply_nested: dict[str, Any] = {}
        cursor = deeply_nested
        for _ in range(22):
            child: dict[str, Any] = {}
            cursor["child"] = child
            cursor = child
        with pytest.raises(ValueError, match="nesting depth"):
            store.create("artifact.verify", request=deeply_nested)
        with pytest.raises(ValueError, match="encoded bytes"):
            store.create("artifact.verify", request={"payload": "x" * (256 * 1024)})


def test_large_durable_progress_uses_a_small_revision_event(tmp_path: Path) -> None:
    broker = EventBroker()
    with OperationStore(tmp_path / "operations.sqlite", broker=broker) as store:
        operation = store.create("artifact.verify", mutating=False)
        store.start(operation.operation_id)
        progress = store.set_progress(
            operation.operation_id,
            {"detail": "x" * (128 * 1024)},
        )

    assert len(progress.progress["detail"]) == 128 * 1024
    event = broker.replay().events[-1]
    assert event.event_type == "operation.progress"
    assert "progress" not in event.data


def test_clock_must_be_timezone_aware(tmp_path: Path) -> None:
    with OperationStore(
        tmp_path / "operations.sqlite",
        clock=lambda: datetime(2026, 1, 1),
    ) as store:
        with pytest.raises(ValueError, match="timezone-aware"):
            store.create("artifact.verify")


def test_event_broker_bounds_replay_and_detects_cursor_gaps() -> None:
    broker = EventBroker(max_events=2, stream_id="c" * 32)
    first = broker.publish("operation.queued", data={"number": 1})
    second = broker.publish("operation.running", data={"number": 2})
    third = broker.publish("operation.succeeded", data={"number": 3})

    initial = broker.replay()
    assert initial.events == (second, third)
    assert not initial.reset_required
    assert initial.first_available_sequence == 2
    assert initial.cursor == third.event_id

    contiguous = broker.replay(first.event_id)
    assert contiguous.events == (second, third)
    assert not contiguous.reset_required

    truncated = broker.replay(0)
    assert truncated.events == (second, third)
    assert truncated.reset_required

    incremental = broker.replay(second.event_id)
    assert incremental.events == (third,)
    assert not incremental.reset_required

    foreign = broker.replay("d" * 32 + ":99")
    assert foreign.events == (second, third)
    assert foreign.reset_required

    future = broker.replay("c" * 32 + ":999")
    assert future.events == (second, third)
    assert future.reset_required


def test_event_payloads_are_immutable_and_sse_safe() -> None:
    broker = EventBroker(stream_id="e" * 32)
    event = broker.publish("operation.progress", data={"nested": {"value": [1]}})
    with pytest.raises(TypeError):
        event.data["changed"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        event.data["nested"]["changed"] = True  # type: ignore[index]

    rendered = event.to_sse()
    assert rendered.startswith(f"id: {event.event_id}\nevent: operation.progress\n")
    assert rendered.endswith("\n\n")
    payload_line = next(line for line in rendered.splitlines() if line.startswith("data: "))
    assert json.loads(payload_line.removeprefix("data: "))["data"] == {
        "nested": {"value": [1]}
    }
    with pytest.raises(ValueError, match="event type"):
        broker.publish("operation.running\ndata: injected")


def test_event_wait_wakes_for_matching_publication_and_times_out() -> None:
    broker = EventBroker(stream_id="f" * 32)
    operation_id = "op_" + "1" * 32
    cursor = broker.cursor

    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(
            broker.wait, cursor, timeout=2.0, operation_id=operation_id
        )
        time.sleep(0.02)
        broker.publish(
            "operation.queued",
            operation_id="op_" + "2" * 32,
            data={},
        )
        time.sleep(0.02)
        expected = broker.publish(
            "operation.queued", operation_id=operation_id, data={}
        )
        replay = waiting.result(timeout=2)

    assert replay.events == (expected,)
    started = time.monotonic()
    empty = broker.wait(replay.cursor, timeout=0.02, operation_id=operation_id)
    assert empty.events == ()
    assert time.monotonic() - started >= 0.01
    with pytest.raises(ValueError, match="finite non-negative"):
        broker.wait(replay.cursor, timeout=float("inf"))


def test_database_records_survive_reopen(tmp_path: Path) -> None:
    database = tmp_path / "operations.sqlite"
    with OperationStore(database) as store:
        created = store.create(
            "benchmark.run",
            request={"suite_id": "smoke"},
            idempotency_key="durable-run",
            mutating=False,
        )
        store.start(created.operation_id)
        expected = store.succeed(created.operation_id, result={"run_id": "run-1"})

    with OperationStore(database) as reopened:
        assert reopened.get(created.operation_id) == expected
        replayed = reopened.admit(
            "benchmark.run",
            request={"suite_id": "smoke"},
            idempotency_key="durable-run",
            mutating=False,
        )
        assert replayed.replayed
        assert replayed.operation == expected


def test_request_digest_corruption_is_detected(tmp_path: Path) -> None:
    database = tmp_path / "operations.sqlite"
    with OperationStore(database) as store:
        record = store.create("artifact.verify", request={"artifact_id": "a"})
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE operations SET request_json = ? WHERE operation_id = ?",
        ('{"artifact_id":"b"}', record.operation_id),
    )
    connection.commit()
    connection.close()

    with OperationStore(database) as reopened:
        with pytest.raises(OperationStoreError, match="digest"):
            reopened.get(record.operation_id)


def test_database_symlink_and_parent_symlink_are_rejected(tmp_path: Path) -> None:
    victim = tmp_path / "victim.sqlite"
    with OperationStore(victim):
        pass
    before = victim.read_bytes()
    link = tmp_path / "operations.sqlite"
    link.symlink_to(victim)
    with pytest.raises(StoragePolicyError, match="operation database"):
        OperationStore(link)
    assert victim.read_bytes() == before

    real_directory = tmp_path / "real"
    real_directory.mkdir()
    marker = real_directory / "marker"
    marker.write_text("unchanged", encoding="utf-8")
    redirected = tmp_path / "redirected"
    redirected.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(StoragePolicyError, match="safe real directory"):
        OperationStore(redirected / "operations.sqlite")
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert not (real_directory / "operations.sqlite").exists()


@pytest.mark.parametrize("suffix", ("-journal", "-wal", "-shm"))
def test_database_sidecar_symlinks_are_rejected(
    tmp_path: Path, suffix: str
) -> None:
    database = tmp_path / "operations.sqlite"
    victim = tmp_path / f"victim{suffix}"
    victim.write_text("unchanged", encoding="utf-8")
    Path(f"{database}{suffix}").symlink_to(victim)

    with pytest.raises(StoragePolicyError, match="operation database sidecar"):
        OperationStore(database)
    assert victim.read_text(encoding="utf-8") == "unchanged"


def test_foreign_or_future_database_schema_is_rejected(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.sqlite"
    connection = sqlite3.connect(foreign)
    connection.execute("PRAGMA application_id = 12345")
    connection.commit()
    connection.close()
    with pytest.raises(OperationStoreError, match="another application"):
        OperationStore(foreign)

    future = tmp_path / "future.sqlite"
    connection = sqlite3.connect(future)
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()
    with pytest.raises(OperationStoreError, match="unsupported"):
        OperationStore(future)
