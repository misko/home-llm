"""Durable operation records and bounded management-event replay.

The web control plane uses this module to describe work; it does not persist
Python callables, shell commands, or other executable objects.  Requests,
progress, results, and event data are deliberately restricted to bounded JSON.

SQLite protects admission across threads and processes.  As elsewhere in LLM
Lab, the workstation owner's UID is the local trust boundary: no-follow opens
and private files protect against accidental/path-redirection mistakes, not a
malicious process already running as that same user.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, TypeAlias
from urllib.parse import quote

from .hashing import sha256_bytes
from .paths import open_private_regular_file, validate_private_regular_file_if_present


SCHEMA_VERSION = 1
_APPLICATION_ID = 0x4C4C4D4F  # ``LLMO``: this is a dedicated operation database.
MAX_DOCUMENT_BYTES = 256 * 1024
MAX_EVENT_BYTES = 64 * 1024
MAX_JSON_DEPTH = 20
MAX_EVENT_HISTORY = 10_000

_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_EVENT_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_OPERATION_ID_PATTERN = re.compile(r"^op_[0-9a-f]{32}$")
_IDEMPOTENCY_PATTERN = re.compile(r"^[!-~]{1,200}$")
_ACTIVE_STATUSES = ("queued", "running", "cancelling")
_GENERIC_FAILURE_MESSAGE = "The operation could not be completed."


JsonScalar: TypeAlias = None | bool | int | float | str
FrozenJson: TypeAlias = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


class OperationStatus(str, Enum):
    """Stable public states for a durable operation."""

    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"

    @property
    def active(self) -> bool:
        return self in {
            OperationStatus.QUEUED,
            OperationStatus.RUNNING,
            OperationStatus.CANCELLING,
        }

    @property
    def terminal(self) -> bool:
        return not self.active


class OperationStoreError(RuntimeError):
    """Base class for expected operation-store failures."""


class OperationNotFoundError(OperationStoreError):
    """The requested operation does not exist."""


class OperationConflictError(OperationStoreError):
    """The requested operation conflicts with durable state."""


class InvalidOperationTransitionError(OperationConflictError):
    """An operation cannot move from its current state as requested."""


class IdempotencyConflictError(OperationConflictError):
    """An idempotency key was reused for a different request."""


class OperationBusyError(OperationConflictError):
    """Another mutating operation already owns the admission slot."""

    def __init__(self, active_operation: "OperationRecord") -> None:
        self.active_operation = active_operation
        super().__init__(
            "another mutating operation is active "
            f"({active_operation.operation_id}, {active_operation.status.value})"
        )


class SafeOperationError(RuntimeError):
    """An exception whose carefully chosen message may be shown to a client."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class OperationFailure:
    """Sanitized failure information safe for management APIs and events."""

    code: str
    message: str
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


@dataclass(frozen=True, slots=True)
class OperationRecord:
    """One immutable snapshot of a durable operation."""

    operation_id: str
    kind: str
    status: OperationStatus
    mutating: bool
    idempotency_key: str | None
    request_sha256: str
    request: Mapping[str, FrozenJson]
    progress: Mapping[str, FrozenJson]
    result: Mapping[str, FrozenJson] | None
    error: OperationFailure | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    revision: int

    @property
    def active(self) -> bool:
        return self.status.active

    @property
    def terminal(self) -> bool:
        return self.status.terminal

    def to_dict(
        self,
        *,
        include_request: bool = True,
        include_idempotency_key: bool = False,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "operation_id": self.operation_id,
            "kind": self.kind,
            "status": self.status.value,
            "mutating": self.mutating,
            "request_sha256": self.request_sha256,
            "progress": _thaw_json(self.progress),
            "result": _thaw_json(self.result),
            "error": self.error.to_dict() if self.error else None,
            "created_at": _format_time(self.created_at),
            "updated_at": _format_time(self.updated_at),
            "started_at": _format_time(self.started_at) if self.started_at else None,
            "finished_at": (
                _format_time(self.finished_at) if self.finished_at else None
            ),
            "revision": self.revision,
        }
        if include_request:
            value["request"] = _thaw_json(self.request)
        if include_idempotency_key:
            value["idempotency_key"] = self.idempotency_key
        return value


@dataclass(frozen=True, slots=True)
class OperationAdmission:
    """The result of an admission request, including idempotent replay status."""

    operation: OperationRecord
    replayed: bool


@dataclass(frozen=True, slots=True)
class OperationEvent:
    """One immutable, process-local event suitable for an SSE frame."""

    stream_id: str
    sequence: int
    event_type: str
    operation_id: str | None
    occurred_at: datetime
    data: Mapping[str, FrozenJson]

    @property
    def event_id(self) -> str:
        return f"{self.stream_id}:{self.sequence}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "type": self.event_type,
            "operation_id": self.operation_id,
            "occurred_at": _format_time(self.occurred_at),
            "data": _thaw_json(self.data),
        }

    def to_sse(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            f"id: {self.event_id}\n"
            f"event: {self.event_type}\n"
            f"data: {payload}\n\n"
        )


@dataclass(frozen=True, slots=True)
class EventReplay:
    """A replay window and whether the caller's cursor fell out of that window."""

    stream_id: str
    events: tuple[OperationEvent, ...]
    first_available_sequence: int
    last_sequence: int
    reset_required: bool

    @property
    def cursor(self) -> str:
        return f"{self.stream_id}:{self.last_sequence}"


class EventBroker:
    """Thread-safe bounded event publication, replay, and long polling.

    The broker is intentionally in memory.  Operation records are authoritative
    and durable; an SSE client that receives ``reset_required`` refetches those
    records before continuing from the returned cursor.
    """

    def __init__(
        self,
        max_events: int = 512,
        *,
        clock: Callable[[], datetime] | None = None,
        stream_id: str | None = None,
    ) -> None:
        if not isinstance(max_events, int) or isinstance(max_events, bool):
            raise TypeError("max_events must be an integer")
        if not 1 <= max_events <= MAX_EVENT_HISTORY:
            raise ValueError(
                f"max_events must be between 1 and {MAX_EVENT_HISTORY}"
            )
        resolved_stream_id = stream_id or secrets.token_hex(16)
        if not re.fullmatch(r"[0-9a-f]{32}", resolved_stream_id):
            raise ValueError("stream_id must be 32 lowercase hexadecimal characters")
        self.max_events = max_events
        self.stream_id = resolved_stream_id
        self._clock = clock or _utc_now
        self._condition = threading.Condition(threading.RLock())
        self._events: deque[OperationEvent] = deque(maxlen=max_events)
        self._last_sequence = 0

    @property
    def cursor(self) -> str:
        with self._condition:
            return f"{self.stream_id}:{self._last_sequence}"

    def publish(
        self,
        event_type: str,
        *,
        operation_id: str | None = None,
        data: Mapping[str, Any] | None = None,
        occurred_at: datetime | None = None,
    ) -> OperationEvent:
        _validate_event_type(event_type)
        if operation_id is not None:
            _validate_operation_id(operation_id)
        frozen_data, _ = _document(
            {} if data is None else data,
            max_bytes=MAX_EVENT_BYTES,
            name="event",
        )
        timestamp = _coerce_time(occurred_at or self._clock())
        with self._condition:
            self._last_sequence += 1
            event = OperationEvent(
                stream_id=self.stream_id,
                sequence=self._last_sequence,
                event_type=event_type,
                operation_id=operation_id,
                occurred_at=timestamp,
                data=frozen_data,
            )
            self._events.append(event)
            self._condition.notify_all()
            return event

    def replay(
        self,
        after: str | int | None = None,
        *,
        operation_id: str | None = None,
    ) -> EventReplay:
        if operation_id is not None:
            _validate_operation_id(operation_id)
        with self._condition:
            return self._replay_locked(after, operation_id=operation_id)

    def wait(
        self,
        after: str | int | None,
        *,
        timeout: float | None = 15.0,
        operation_id: str | None = None,
    ) -> EventReplay:
        """Wait until matching events, a reset condition, or timeout.

        This blocking primitive is suitable for ``asyncio.to_thread`` in a
        FastAPI SSE generator and avoids coupling the persistence layer to an
        asynchronous framework.
        """

        if operation_id is not None:
            _validate_operation_id(operation_id)
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be a finite non-negative number or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                replay = self._replay_locked(after, operation_id=operation_id)
                if replay.events or replay.reset_required:
                    return replay
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return replay
                self._condition.wait(remaining)

    def _replay_locked(
        self,
        after: str | int | None,
        *,
        operation_id: str | None,
    ) -> EventReplay:
        foreign_stream, requested, initial = _parse_cursor(after, self.stream_id)
        first = self._events[0].sequence if self._events else self._last_sequence + 1
        reset = foreign_stream
        if initial:
            requested = first - 1
        elif requested < first - 1 or requested > self._last_sequence:
            reset = True
            requested = first - 1
        events = tuple(
            event
            for event in self._events
            if event.sequence > requested
            and (operation_id is None or event.operation_id == operation_id)
        )
        return EventReplay(
            stream_id=self.stream_id,
            events=events,
            first_available_sequence=first,
            last_sequence=self._last_sequence,
            reset_required=reset,
        )


class OperationStore:
    """SQLite-backed durable operation state with atomic mutation admission.

    Call :meth:`reconcile_incomplete` exactly once when the owning controller
    starts.  It is explicit, rather than constructor-driven, so opening another
    read/store handle cannot incorrectly interrupt work owned by a live handle.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        broker: EventBroker | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.broker = broker or EventBroker()
        self._clock = clock or _utc_now
        self._lock = threading.RLock()
        self.path, descriptor, opened = open_private_regular_file(
            path,
            os.O_RDWR | os.O_CREAT,
            purpose="operation database",
        )
        uri = f"file:{quote(str(self.path), safe='/')}?mode=rwc&nofollow=1"
        connection: sqlite3.Connection | None = None
        try:
            self._validate_sidecars()
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=30.0,
                check_same_thread=False,
            )
            named = os.lstat(self.path)
            if (
                named.st_nlink != 1
                or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                raise OperationStoreError(
                    f"operation database path changed while opening: {self.path}"
                )
            self._connection = connection
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA trusted_schema = OFF")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._validate_sidecars()
            self.initialize()
        except BaseException:
            if connection is not None:
                connection.close()
            raise
        finally:
            os.close(descriptor)

    def _validate_sidecars(self) -> None:
        for suffix in ("-journal", "-wal", "-shm"):
            validate_private_regular_file_if_present(
                Path(f"{self.path}{suffix}"),
                purpose=f"operation database sidecar {suffix}",
            )

    def initialize(self) -> None:
        """Create or validate the dedicated database schema idempotently."""

        with self._transaction() as connection:
            application_id = connection.execute("PRAGMA application_id").fetchone()[0]
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if application_id not in (0, _APPLICATION_ID):
                raise OperationStoreError(
                    "database belongs to another application and cannot store operations"
                )
            if version not in (0, SCHEMA_VERSION):
                raise OperationStoreError(
                    f"operation schema version {version} is unsupported "
                    f"(expected {SCHEMA_VERSION})"
                )
            connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'queued', 'running', 'cancelling', 'succeeded',
                            'failed', 'cancelled', 'interrupted'
                        )
                    ),
                    mutating INTEGER NOT NULL CHECK (mutating IN (0, 1)),
                    idempotency_key TEXT,
                    request_sha256 TEXT NOT NULL CHECK (length(request_sha256) = 64),
                    request_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    error_retryable INTEGER CHECK (error_retryable IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    revision INTEGER NOT NULL CHECK (revision > 0)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS operations_idempotency_key
                    ON operations(idempotency_key)
                    WHERE idempotency_key IS NOT NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS operations_one_active_mutation
                    ON operations(mutating)
                    WHERE mutating = 1
                      AND status IN ('queued', 'running', 'cancelling');
                CREATE INDEX IF NOT EXISTS operations_created
                    ON operations(created_at DESC, operation_id DESC);
                CREATE INDEX IF NOT EXISTS operations_status_created
                    ON operations(status, created_at, operation_id);
                PRAGMA application_id = {_APPLICATION_ID};
                PRAGMA user_version = {SCHEMA_VERSION};
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "OperationStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._validate_sidecars()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()
                self._validate_sidecars()

    def admit(
        self,
        kind: str,
        *,
        request: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        mutating: bool = True,
    ) -> OperationAdmission:
        """Atomically admit work or replay an identical idempotent request."""

        _validate_kind(kind)
        if type(mutating) is not bool:
            raise TypeError("mutating must be a boolean")
        if idempotency_key is not None:
            _validate_idempotency_key(idempotency_key)
        frozen_request, request_json = _document(
            {} if request is None else request,
            max_bytes=MAX_DOCUMENT_BYTES,
            name="operation request",
        )
        request_sha256 = sha256_bytes(request_json.encode("utf-8"))
        operation_id = f"op_{secrets.token_hex(16)}"
        now = _format_time(_coerce_time(self._clock()))

        with self._transaction() as connection:
            if idempotency_key is not None:
                existing = connection.execute(
                    "SELECT * FROM operations WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing is not None:
                    record = _record(existing)
                    if (
                        record.kind != kind
                        or record.mutating != mutating
                        or record.request_sha256 != request_sha256
                    ):
                        raise IdempotencyConflictError(
                            "the idempotency key is already bound to another request"
                        )
                    return OperationAdmission(record, replayed=True)

            if mutating:
                active = connection.execute(
                    "SELECT * FROM operations "
                    "WHERE mutating = 1 AND status IN ('queued','running','cancelling') "
                    "ORDER BY created_at, operation_id LIMIT 1"
                ).fetchone()
                if active is not None:
                    raise OperationBusyError(_record(active))

            try:
                connection.execute(
                    """
                    INSERT INTO operations(
                        operation_id, kind, status, mutating, idempotency_key,
                        request_sha256, request_json, progress_json, result_json,
                        error_code, error_message, error_retryable, created_at,
                        updated_at, started_at, finished_at, revision
                    ) VALUES (?, ?, 'queued', ?, ?, ?, ?, '{}', NULL,
                              NULL, NULL, NULL, ?, ?, NULL, NULL, 1)
                    """,
                    (
                        operation_id,
                        kind,
                        int(mutating),
                        idempotency_key,
                        request_sha256,
                        request_json,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                # Cross-process callers are serialized by BEGIN IMMEDIATE, but
                # retaining these translations makes the schema constraints a
                # safe final line of defence.
                if idempotency_key is not None:
                    existing = connection.execute(
                        "SELECT * FROM operations WHERE idempotency_key = ?",
                        (idempotency_key,),
                    ).fetchone()
                    if existing is not None:
                        record = _record(existing)
                        if (
                            record.kind == kind
                            and record.mutating == mutating
                            and record.request_sha256 == request_sha256
                        ):
                            return OperationAdmission(record, replayed=True)
                        raise IdempotencyConflictError(
                            "the idempotency key is already bound to another request"
                        ) from exc
                active = connection.execute(
                    "SELECT * FROM operations "
                    "WHERE mutating = 1 AND status IN ('queued','running','cancelling') "
                    "LIMIT 1"
                ).fetchone()
                if mutating and active is not None:
                    raise OperationBusyError(_record(active)) from exc
                raise OperationStoreError("could not admit operation") from exc
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            assert row is not None
            record = _record(row)

        self._emit("operation.queued", record)
        # Keep the frozen object built from SQLite authoritative.  This also
        # proves the serialized request can be decoded under the strict policy.
        assert record.request == frozen_request
        return OperationAdmission(record, replayed=False)

    def create(
        self,
        kind: str,
        *,
        request: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        mutating: bool = True,
    ) -> OperationRecord:
        """Convenience wrapper returning only the admitted/replayed record."""

        return self.admit(
            kind,
            request=request,
            idempotency_key=idempotency_key,
            mutating=mutating,
        ).operation

    def get(self, operation_id: str) -> OperationRecord:
        _validate_operation_id(operation_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        if row is None:
            raise OperationNotFoundError("operation was not found")
        return _record(row)

    def find(self, operation_id: str) -> OperationRecord | None:
        _validate_operation_id(operation_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        return _record(row) if row is not None else None

    def list(
        self,
        *,
        statuses: Iterable[OperationStatus | str] | None = None,
        mutating: bool | None = None,
        limit: int = 100,
    ) -> tuple[OperationRecord, ...]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer between 1 and 500")
        if mutating is not None and type(mutating) is not bool:
            raise TypeError("mutating must be a boolean or None")
        resolved_statuses = _coerce_statuses(statuses)
        clauses: list[str] = []
        parameters: list[Any] = []
        if resolved_statuses is not None:
            if not resolved_statuses:
                return ()
            placeholders = ",".join("?" for _ in resolved_statuses)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(status.value for status in resolved_statuses)
        if mutating is not None:
            clauses.append("mutating = ?")
            parameters.append(int(mutating))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM operations"
                + where
                + " ORDER BY created_at DESC, operation_id DESC LIMIT ?",
                parameters,
            ).fetchall()
        return tuple(_record(row) for row in rows)

    def active_mutation(self) -> OperationRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM operations "
                "WHERE mutating = 1 AND status IN ('queued','running','cancelling') "
                "ORDER BY created_at, operation_id LIMIT 1"
            ).fetchone()
        return _record(row) if row is not None else None

    def claim_next(self, *, kind: str | None = None) -> OperationRecord | None:
        """Atomically claim the oldest queued operation."""

        if kind is not None:
            _validate_kind(kind)
        with self._transaction() as connection:
            if kind is None:
                row = connection.execute(
                    "SELECT * FROM operations WHERE status = 'queued' "
                    "ORDER BY created_at, operation_id LIMIT 1"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM operations WHERE status = 'queued' AND kind = ? "
                    "ORDER BY created_at, operation_id LIMIT 1",
                    (kind,),
                ).fetchone()
            if row is None:
                return None
            record = self._start_row(connection, row)
        self._emit("operation.running", record)
        return record

    def start(self, operation_id: str) -> OperationRecord:
        _validate_operation_id(operation_id)
        with self._transaction() as connection:
            row = self._required_row(connection, operation_id)
            current = _record(row)
            if current.status is OperationStatus.RUNNING:
                return current
            if current.status is not OperationStatus.QUEUED:
                raise _invalid_transition(current, OperationStatus.RUNNING)
            record = self._start_row(connection, row)
        self._emit("operation.running", record)
        return record

    def _start_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> OperationRecord:
        now = _format_time(_coerce_time(self._clock()))
        connection.execute(
            "UPDATE operations SET status = 'running', started_at = ?, "
            "updated_at = ?, revision = revision + 1 WHERE operation_id = ?",
            (now, now, row["operation_id"]),
        )
        return self._required_record(connection, row["operation_id"])

    def set_progress(
        self,
        operation_id: str,
        progress: Mapping[str, Any],
    ) -> OperationRecord:
        _validate_operation_id(operation_id)
        _, progress_json = _document(
            progress, max_bytes=MAX_DOCUMENT_BYTES, name="operation progress"
        )
        with self._transaction() as connection:
            current = self._required_record(connection, operation_id)
            if current.status not in {
                OperationStatus.RUNNING,
                OperationStatus.CANCELLING,
            }:
                raise InvalidOperationTransitionError(
                    "progress can only be updated while an operation is running"
                )
            if _canonical_mapping(current.progress) == progress_json:
                return current
            now = _format_time(_coerce_time(self._clock()))
            connection.execute(
                "UPDATE operations SET progress_json = ?, updated_at = ?, "
                "revision = revision + 1 WHERE operation_id = ?",
                (progress_json, now, operation_id),
            )
            record = self._required_record(connection, operation_id)
        # The event is deliberately only a revision notification.  Progress
        # documents may be larger than the event window's stricter byte bound;
        # consumers fetch the authoritative record after receiving it.
        self._emit("operation.progress", record)
        return record

    def succeed(
        self,
        operation_id: str,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> OperationRecord:
        _validate_operation_id(operation_id)
        _, result_json = _document(
            {} if result is None else result,
            max_bytes=MAX_DOCUMENT_BYTES,
            name="operation result",
        )
        with self._transaction() as connection:
            current = self._required_record(connection, operation_id)
            if current.status is OperationStatus.SUCCEEDED:
                if _canonical_mapping(current.result or {}) != result_json:
                    raise OperationConflictError(
                        "operation already succeeded with a different result"
                    )
                return current
            if current.status not in {
                OperationStatus.RUNNING,
                OperationStatus.CANCELLING,
            }:
                raise _invalid_transition(current, OperationStatus.SUCCEEDED)
            now = _format_time(_coerce_time(self._clock()))
            connection.execute(
                "UPDATE operations SET status = 'succeeded', result_json = ?, "
                "error_code = NULL, error_message = NULL, error_retryable = NULL, "
                "updated_at = ?, finished_at = ?, revision = revision + 1 "
                "WHERE operation_id = ?",
                (result_json, now, now, operation_id),
            )
            record = self._required_record(connection, operation_id)
        self._emit("operation.succeeded", record)
        return record

    def fail(
        self,
        operation_id: str,
        error: BaseException | OperationFailure | None = None,
        *,
        code: str | None = None,
        message: str | None = None,
        retryable: bool = False,
    ) -> OperationRecord:
        """Mark active work failed without persisting arbitrary exception text."""

        _validate_operation_id(operation_id)
        failure = sanitize_failure(
            error, code=code, message=message, retryable=retryable
        )
        with self._transaction() as connection:
            current = self._required_record(connection, operation_id)
            if current.status is OperationStatus.FAILED:
                if current.error != failure:
                    raise OperationConflictError(
                        "operation already failed with different public error data"
                    )
                return current
            if not current.active:
                raise _invalid_transition(current, OperationStatus.FAILED)
            now = _format_time(_coerce_time(self._clock()))
            connection.execute(
                "UPDATE operations SET status = 'failed', error_code = ?, "
                "error_message = ?, error_retryable = ?, updated_at = ?, "
                "finished_at = ?, revision = revision + 1 WHERE operation_id = ?",
                (
                    failure.code,
                    failure.message,
                    int(failure.retryable),
                    now,
                    now,
                    operation_id,
                ),
            )
            record = self._required_record(connection, operation_id)
        self._emit("operation.failed", record)
        return record

    def request_cancel(self, operation_id: str) -> OperationRecord:
        """Request cancellation while keeping running work admission-blocking."""

        _validate_operation_id(operation_id)
        with self._transaction() as connection:
            current = self._required_record(connection, operation_id)
            if current.status in {
                OperationStatus.CANCELLING,
                OperationStatus.CANCELLED,
            }:
                return current
            if current.status is OperationStatus.QUEUED:
                now = _format_time(_coerce_time(self._clock()))
                connection.execute(
                    "UPDATE operations SET status = 'cancelled', updated_at = ?, "
                    "finished_at = ?, revision = revision + 1 WHERE operation_id = ?",
                    (now, now, operation_id),
                )
                record = self._required_record(connection, operation_id)
                event_type = "operation.cancelled"
            elif current.status is OperationStatus.RUNNING:
                now = _format_time(_coerce_time(self._clock()))
                connection.execute(
                    "UPDATE operations SET status = 'cancelling', updated_at = ?, "
                    "revision = revision + 1 WHERE operation_id = ?",
                    (now, operation_id),
                )
                record = self._required_record(connection, operation_id)
                event_type = "operation.cancelling"
            else:
                raise _invalid_transition(current, OperationStatus.CANCELLING)
        self._emit(event_type, record)
        return record

    def mark_cancelled(self, operation_id: str) -> OperationRecord:
        """Confirm that an executor stopped before releasing mutation admission."""

        _validate_operation_id(operation_id)
        with self._transaction() as connection:
            current = self._required_record(connection, operation_id)
            if current.status is OperationStatus.CANCELLED:
                return current
            if current.status not in {
                OperationStatus.RUNNING,
                OperationStatus.CANCELLING,
            }:
                raise _invalid_transition(current, OperationStatus.CANCELLED)
            now = _format_time(_coerce_time(self._clock()))
            connection.execute(
                "UPDATE operations SET status = 'cancelled', updated_at = ?, "
                "finished_at = ?, revision = revision + 1 WHERE operation_id = ?",
                (now, now, operation_id),
            )
            record = self._required_record(connection, operation_id)
        self._emit("operation.cancelled", record)
        return record

    def reconcile_incomplete(self) -> tuple[OperationRecord, ...]:
        """Mark nonterminal records interrupted after controller restart.

        The durable queue has no serialized executable code to resume.  The
        controller therefore reconciles every queued/running/cancelling record
        to a visible terminal state, then clients may retry idempotently with a
        new key if appropriate.
        """

        failure = OperationFailure(
            code="service_restarted",
            message="The service restarted before the operation completed.",
            retryable=True,
        )
        with self._transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM operations "
                "WHERE status IN ('queued','running','cancelling') "
                "ORDER BY created_at, operation_id"
            ).fetchall()
            if not rows:
                return ()
            now = _format_time(_coerce_time(self._clock()))
            operation_ids = [row["operation_id"] for row in rows]
            connection.executemany(
                "UPDATE operations SET status = 'interrupted', error_code = ?, "
                "error_message = ?, error_retryable = 1, updated_at = ?, "
                "finished_at = ?, revision = revision + 1 WHERE operation_id = ?",
                (
                    (failure.code, failure.message, now, now, operation_id)
                    for operation_id in operation_ids
                ),
            )
            records = tuple(
                self._required_record(connection, operation_id)
                for operation_id in operation_ids
            )
        for record in records:
            self._emit("operation.interrupted", record)
        return records

    def _required_row(
        self, connection: sqlite3.Connection, operation_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise OperationNotFoundError("operation was not found")
        return row

    def _required_record(
        self, connection: sqlite3.Connection, operation_id: str
    ) -> OperationRecord:
        return _record(self._required_row(connection, operation_id))

    def _emit(
        self,
        event_type: str,
        record: OperationRecord,
        *,
        data: Mapping[str, Any] | None = None,
    ) -> OperationEvent:
        public_data: dict[str, Any] = {
            "operation_id": record.operation_id,
            "kind": record.kind,
            "status": record.status.value,
            "mutating": record.mutating,
            "revision": record.revision,
        }
        if data:
            public_data.update(data)
        if record.error is not None:
            public_data["error"] = record.error.to_dict()
        return self.broker.publish(
            event_type,
            operation_id=record.operation_id,
            data=public_data,
            occurred_at=record.updated_at,
        )


def sanitize_failure(
    error: BaseException | OperationFailure | None = None,
    *,
    code: str | None = None,
    message: str | None = None,
    retryable: bool = False,
) -> OperationFailure:
    """Convert failure input into bounded public data.

    Unexpected exception messages are never exposed.  A caller must use
    :class:`SafeOperationError` or explicit ``code``/``message`` arguments to
    opt a known-safe explanation into the public record.
    """

    if error is not None and (code is not None or message is not None):
        raise ValueError("pass either error or explicit code/message, not both")
    if isinstance(error, OperationFailure):
        raw_code = error.code
        raw_message = error.message
        raw_retryable = error.retryable
    elif isinstance(error, SafeOperationError):
        raw_code = error.code
        raw_message = error.public_message
        raw_retryable = error.retryable
    elif error is not None:
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception or OperationFailure")
        raw_code = "operation_failed"
        raw_message = _GENERIC_FAILURE_MESSAGE
        raw_retryable = retryable
    else:
        raw_code = code or "operation_failed"
        raw_message = message or _GENERIC_FAILURE_MESSAGE
        raw_retryable = retryable
    if type(raw_retryable) is not bool:
        raise TypeError("retryable must be a boolean")
    return OperationFailure(
        code=_sanitize_error_code(raw_code),
        message=_sanitize_message(raw_message),
        retryable=raw_retryable,
    )


def _sanitize_error_code(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 64 or not _ERROR_CODE_PATTERN.fullmatch(value):
        return "operation_failed"
    return value


def _sanitize_message(value: Any) -> str:
    if not isinstance(value, str):
        return _GENERIC_FAILURE_MESSAGE
    # Remove control characters and collapse all whitespace before applying
    # conservative redaction.  Public messages should describe the action, not
    # expose the local filesystem or credentials.
    text = " ".join("".join(character if character >= " " else " " for character in value).split())
    text = re.sub(
        r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/-]+=*",
        r"\1 [redacted]",
        text,
    )
    text = re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|token|password|secret)\b"
        r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
        r"\1=[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)([a-z][a-z0-9+.-]*://)(?:[^/@\s]+)@",
        r"\1[redacted]@",
        text,
    )
    text = re.sub(r"(?<![A-Za-z0-9])(?:~|/)(?:[^\s,;:]+)", "[path]", text)
    text = re.sub(r"(?i)\b[A-Z]:\\[^\s,;:]+", "[path]", text)
    if not text:
        text = _GENERIC_FAILURE_MESSAGE
    if len(text) > 500:
        text = text[:497].rstrip() + "..."
    return text


def _document(
    value: Mapping[str, Any], *, max_bytes: int, name: str
) -> tuple[Mapping[str, FrozenJson], str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    normalized = _normalize_json(value, name=name)
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must be a JSON object")
    encoded = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > max_bytes:
        raise ValueError(f"{name} exceeds {max_bytes} encoded bytes")
    frozen = _freeze_json(normalized)
    assert isinstance(frozen, Mapping)
    return frozen, encoded


def _normalize_json(value: Any, *, name: str) -> Any:
    active: set[int] = set()

    def visit(item: Any, depth: int) -> Any:
        if depth > MAX_JSON_DEPTH:
            raise ValueError(f"{name} exceeds maximum nesting depth {MAX_JSON_DEPTH}")
        if item is None or type(item) in {bool, int, str}:
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{name} contains a non-finite number")
            return item
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ValueError(f"{name} contains a cycle")
            active.add(identity)
            try:
                output: dict[str, Any] = {}
                for key, child in item.items():
                    if type(key) is not str:
                        raise TypeError(f"{name} object keys must be strings")
                    output[key] = visit(child, depth + 1)
                return output
            finally:
                active.remove(identity)
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError(f"{name} contains a cycle")
            active.add(identity)
            try:
                return [visit(child, depth + 1) for child in item]
            finally:
                active.remove(identity)
        raise TypeError(
            f"{name} contains unsupported value type {type(item).__name__}; "
            "only JSON data is accepted"
        )

    return visit(value, 0)


def _freeze_json(value: Any) -> FrozenJson:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: FrozenJson | None) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _decode_mapping(value: str, *, name: str) -> Mapping[str, FrozenJson]:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"invalid JSON constant {constant}")

    try:
        decoded = json.loads(value, parse_constant=reject_constant)
        frozen, _ = _document(decoded, max_bytes=MAX_DOCUMENT_BYTES, name=name)
        return frozen
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationStoreError(f"operation database contains invalid {name}") from exc


def _canonical_mapping(value: Mapping[str, Any]) -> str:
    _, encoded = _document(
        value, max_bytes=MAX_DOCUMENT_BYTES, name="operation document"
    )
    return encoded


def _record(row: sqlite3.Row) -> OperationRecord:
    try:
        request = _decode_mapping(row["request_json"], name="request")
        request_sha256 = sha256_bytes(_canonical_mapping(request).encode("utf-8"))
        if request_sha256 != row["request_sha256"]:
            raise OperationStoreError("operation request digest does not match its record")
        progress = _decode_mapping(row["progress_json"], name="progress")
        result = (
            _decode_mapping(row["result_json"], name="result")
            if row["result_json"] is not None
            else None
        )
        error = None
        if row["error_code"] is not None:
            error = OperationFailure(
                code=_sanitize_error_code(row["error_code"]),
                message=_sanitize_message(row["error_message"]),
                retryable=bool(row["error_retryable"]),
            )
        return OperationRecord(
            operation_id=row["operation_id"],
            kind=row["kind"],
            status=OperationStatus(row["status"]),
            mutating=bool(row["mutating"]),
            idempotency_key=row["idempotency_key"],
            request_sha256=row["request_sha256"],
            request=request,
            progress=progress,
            result=result,
            error=error,
            created_at=_parse_time(row["created_at"]),
            updated_at=_parse_time(row["updated_at"]),
            started_at=(
                _parse_time(row["started_at"]) if row["started_at"] else None
            ),
            finished_at=(
                _parse_time(row["finished_at"]) if row["finished_at"] else None
            ),
            revision=int(row["revision"]),
        )
    except OperationStoreError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise OperationStoreError("operation database contains an invalid record") from exc


def _invalid_transition(
    current: OperationRecord, target: OperationStatus
) -> InvalidOperationTransitionError:
    return InvalidOperationTransitionError(
        f"operation cannot transition from {current.status.value} to {target.value}"
    )


def _coerce_statuses(
    values: Iterable[OperationStatus | str] | None,
) -> tuple[OperationStatus, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        values = (values,)  # type: ignore[assignment]
    output: list[OperationStatus] = []
    for value in values:
        try:
            status = value if isinstance(value, OperationStatus) else OperationStatus(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown operation status: {value!r}") from exc
        if status not in output:
            output.append(status)
    return tuple(output)


def _validate_kind(value: Any) -> None:
    if not isinstance(value, str) or len(value) > 100 or not _KIND_PATTERN.fullmatch(value):
        raise ValueError(
            "operation kind must be a lowercase dotted/dashed identifier up to 100 characters"
        )


def _validate_event_type(value: Any) -> None:
    if not isinstance(value, str) or len(value) > 100 or not _EVENT_PATTERN.fullmatch(value):
        raise ValueError(
            "event type must be a lowercase dotted/dashed identifier up to 100 characters"
        )


def _validate_operation_id(value: Any) -> None:
    if not isinstance(value, str) or not _OPERATION_ID_PATTERN.fullmatch(value):
        raise ValueError("invalid operation id")


def _validate_idempotency_key(value: Any) -> None:
    if not isinstance(value, str) or not _IDEMPOTENCY_PATTERN.fullmatch(value):
        raise ValueError(
            "idempotency key must contain 1-200 printable ASCII characters without spaces"
        )


def _parse_cursor(after: str | int | None, stream_id: str) -> tuple[bool, int, bool]:
    if after is None:
        return False, 0, True
    if isinstance(after, bool):
        raise ValueError("event cursor must be an integer or stream cursor")
    if isinstance(after, int):
        if after < 0:
            raise ValueError("event cursor cannot be negative")
        return False, after, False
    if not isinstance(after, str):
        raise TypeError("event cursor must be an integer, string, or None")
    match = re.fullmatch(r"([0-9a-f]{32}):(0|[1-9][0-9]*)", after)
    if match is None:
        raise ValueError("event cursor must have the form <stream-id>:<sequence>")
    return match.group(1) != stream_id, int(match.group(2)), False


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _coerce_time(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _format_time(value: datetime) -> str:
    return _coerce_time(value).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _coerce_time(parsed)


__all__ = [
    "EventBroker",
    "EventReplay",
    "IdempotencyConflictError",
    "InvalidOperationTransitionError",
    "OperationAdmission",
    "OperationBusyError",
    "OperationConflictError",
    "OperationEvent",
    "OperationFailure",
    "OperationNotFoundError",
    "OperationRecord",
    "OperationStatus",
    "OperationStore",
    "OperationStoreError",
    "SafeOperationError",
    "sanitize_failure",
]
