"""SQLite-backed registry for immutable artifacts and mutable aliases."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from .errors import CatalogError, IntegrityError
from .hashing import canonical_json, canonical_sha256
from .paths import open_private_regular_file, validate_private_regular_file_if_present
from .schema import ArtifactManifest


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AliasRecord:
    alias: str
    artifact_id: str
    generation: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AliasHistoryRecord:
    id: int
    alias: str
    old_artifact_id: str | None
    new_artifact_id: str
    action: str
    note: str | None
    changed_at: datetime


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    id: int
    event_type: str
    entity_type: str
    entity_id: str
    payload: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    suite_id: str
    deployment_id: str | None
    artifact_id: str | None
    status: str
    metadata: dict[str, Any]
    started_at: datetime
    finished_at: datetime | None


class Registry:
    """Persistent artifact registry.

    Construction creates the database schema if needed.  Initialization and
    all registrations are idempotent; an identifier reused for different
    immutable content is rejected rather than silently overwritten.
    """

    def __init__(self, path: str | Path) -> None:
        self.path, descriptor, opened = open_private_regular_file(
            path,
            os.O_RDWR | os.O_CREAT,
            purpose="registry database",
        )
        self._lock = threading.RLock()
        uri = f"file:{quote(str(self.path), safe='/')}?mode=rwc&nofollow=1"
        try:
            self._validate_sidecars()
            self._connection = sqlite3.connect(
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
                self._connection.close()
                raise IntegrityError(
                    f"registry database path changed while opening: {self.path}"
                )
            self._validate_sidecars()
        finally:
            os.close(descriptor)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        # WAL permits read-only status/reporting commands while a writer is
        # recording a run.  SQLite may return another mode for special files.
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._validate_sidecars()
        self.initialize()

    def _validate_sidecars(self) -> None:
        for suffix in ("-journal", "-wal", "-shm"):
            validate_private_regular_file_if_present(
                Path(f"{self.path}{suffix}"),
                purpose=f"registry database sidecar {suffix}",
            )

    def initialize(self) -> None:
        """Create or validate the registry schema; safe to call repeatedly."""

        with self._transaction() as connection:
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            if current not in (0, SCHEMA_VERSION):
                raise CatalogError(
                    f"registry schema version {current} is unsupported "
                    f"(expected {SCHEMA_VERSION})"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    manifest_sha256 TEXT NOT NULL UNIQUE,
                    manifest_json TEXT NOT NULL,
                    manifest_path TEXT,
                    created_at TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS aliases (
                    alias TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                    generation INTEGER NOT NULL CHECK (generation > 0),
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS alias_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alias TEXT NOT NULL,
                    old_artifact_id TEXT,
                    new_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                    action TEXT NOT NULL,
                    note TEXT,
                    changed_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS alias_history_lookup
                    ON alias_history(alias, id DESC);

                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS history_entity_lookup
                    ON history(entity_type, entity_id, id DESC);

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    suite_id TEXT NOT NULL,
                    deployment_id TEXT,
                    artifact_id TEXT REFERENCES artifacts(artifact_id),
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS runs_started_at
                    ON runs(started_at DESC);
                """
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "Registry":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def register_artifact(
        self,
        manifest: ArtifactManifest | Mapping[str, Any],
        *,
        manifest_path: str | Path | None = None,
    ) -> ArtifactManifest:
        """Register an immutable manifest and return its sealed form."""

        sealed = seal_manifest(manifest)
        encoded = canonical_json(sealed.model_dump(mode="json"))
        path_text = str(Path(manifest_path).resolve()) if manifest_path else None
        registered_at = _timestamp()

        with self._transaction() as connection:
            row = connection.execute(
                "SELECT manifest_sha256, manifest_json, manifest_path "
                "FROM artifacts WHERE artifact_id = ?",
                (sealed.artifact_id,),
            ).fetchone()
            if row is not None:
                if row["manifest_sha256"] != sealed.manifest_sha256:
                    existing = seal_manifest(
                        ArtifactManifest.model_validate_json(row["manifest_json"])
                    )
                    if manifest_identity_sha256(existing) != (
                        manifest_identity_sha256(sealed)
                    ):
                        raise CatalogError(
                            f"artifact id {sealed.artifact_id!r} is already registered "
                            "with different immutable content"
                        )
                if path_text and not row["manifest_path"]:
                    connection.execute(
                        "UPDATE artifacts SET manifest_path = ? WHERE artifact_id = ?",
                        (path_text, sealed.artifact_id),
                    )
                return seal_manifest(
                    ArtifactManifest.model_validate_json(row["manifest_json"])
                )

            connection.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, manifest_sha256, manifest_json, manifest_path,
                    created_at, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sealed.artifact_id,
                    sealed.manifest_sha256,
                    encoded,
                    path_text,
                    sealed.created_at.isoformat(),
                    registered_at,
                ),
            )
            self._record_history_connection(
                connection,
                event_type="artifact.registered",
                entity_type="artifact",
                entity_id=sealed.artifact_id,
                payload={"manifest_sha256": sealed.manifest_sha256},
                created_at=registered_at,
            )
        return sealed

    def find_artifact(self, artifact_id: str) -> ArtifactManifest | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT manifest_json FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        manifest = ArtifactManifest.model_validate_json(row["manifest_json"])
        return seal_manifest(manifest)

    def get_artifact(self, artifact_id: str) -> ArtifactManifest:
        manifest = self.find_artifact(artifact_id)
        if manifest is None:
            raise CatalogError(f"unknown registered artifact {artifact_id!r}")
        return manifest

    def get_manifest_path(self, artifact_id: str) -> Path | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT manifest_path FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise CatalogError(f"unknown registered artifact {artifact_id!r}")
        return Path(row["manifest_path"]) if row["manifest_path"] else None

    def list_artifacts(self) -> tuple[ArtifactManifest, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT manifest_json FROM artifacts ORDER BY artifact_id"
            ).fetchall()
        return tuple(
            seal_manifest(ArtifactManifest.model_validate_json(row["manifest_json"]))
            for row in rows
        )

    def set_alias(
        self, alias: str, artifact_id: str, *, note: str | None = None
    ) -> AliasRecord:
        """Atomically point *alias* at an artifact and append audit history."""

        alias = _validate_alias(alias)
        changed_at = _timestamp()
        with self._transaction() as connection:
            self._require_artifact_connection(connection, artifact_id)
            current = connection.execute(
                "SELECT artifact_id, generation, updated_at FROM aliases "
                "WHERE alias = ?",
                (alias,),
            ).fetchone()
            if current is not None and current["artifact_id"] == artifact_id:
                return _alias_record(alias, current)

            old_artifact = current["artifact_id"] if current else None
            generation = (current["generation"] + 1) if current else 1
            connection.execute(
                """
                INSERT INTO aliases(alias, artifact_id, generation, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(alias) DO UPDATE SET
                    artifact_id = excluded.artifact_id,
                    generation = excluded.generation,
                    updated_at = excluded.updated_at
                """,
                (alias, artifact_id, generation, changed_at),
            )
            self._insert_alias_history(
                connection,
                alias=alias,
                old_artifact_id=old_artifact,
                new_artifact_id=artifact_id,
                action="set",
                note=note,
                changed_at=changed_at,
            )
            self._record_history_connection(
                connection,
                event_type="alias.set",
                entity_type="alias",
                entity_id=alias,
                payload={
                    "old_artifact_id": old_artifact,
                    "new_artifact_id": artifact_id,
                    "generation": generation,
                    "note": note,
                },
                created_at=changed_at,
            )
        return AliasRecord(alias, artifact_id, generation, _parse_time(changed_at))

    def find_alias(self, alias: str) -> AliasRecord | None:
        alias = _validate_alias(alias)
        with self._lock:
            row = self._connection.execute(
                "SELECT artifact_id, generation, updated_at FROM aliases "
                "WHERE alias = ?",
                (alias,),
            ).fetchone()
        return _alias_record(alias, row) if row is not None else None

    def get_alias(self, alias: str) -> AliasRecord:
        record = self.find_alias(alias)
        if record is None:
            raise CatalogError(f"unknown alias {alias!r}")
        return record

    def resolve_alias(self, alias: str) -> str:
        return self.get_alias(alias).artifact_id

    def list_aliases(self) -> tuple[AliasRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT alias, artifact_id, generation, updated_at "
                "FROM aliases ORDER BY alias"
            ).fetchall()
        return tuple(_alias_record(row["alias"], row) for row in rows)

    def rollback_alias(
        self, alias: str, *, steps: int = 1, note: str | None = None
    ) -> AliasRecord:
        """Restore an alias to the state before one or more prior changes."""

        alias = _validate_alias(alias)
        if steps <= 0:
            raise ValueError("steps must be positive")
        changed_at = _timestamp()
        with self._transaction() as connection:
            current = connection.execute(
                "SELECT artifact_id, generation FROM aliases WHERE alias = ?",
                (alias,),
            ).fetchone()
            if current is None:
                raise CatalogError(f"unknown alias {alias!r}")
            target = connection.execute(
                """
                SELECT old_artifact_id
                FROM alias_history
                WHERE alias = ? AND old_artifact_id IS NOT NULL
                ORDER BY id DESC
                LIMIT 1 OFFSET ?
                """,
                (alias, steps - 1),
            ).fetchone()
            if target is None:
                raise CatalogError(
                    f"alias {alias!r} has fewer than {steps} rollback state(s)"
                )
            target_id = target["old_artifact_id"]
            self._require_artifact_connection(connection, target_id)
            generation = current["generation"] + 1
            connection.execute(
                "UPDATE aliases SET artifact_id = ?, generation = ?, "
                "updated_at = ? WHERE alias = ?",
                (target_id, generation, changed_at, alias),
            )
            self._insert_alias_history(
                connection,
                alias=alias,
                old_artifact_id=current["artifact_id"],
                new_artifact_id=target_id,
                action="rollback",
                note=note,
                changed_at=changed_at,
            )
            self._record_history_connection(
                connection,
                event_type="alias.rolled_back",
                entity_type="alias",
                entity_id=alias,
                payload={
                    "old_artifact_id": current["artifact_id"],
                    "new_artifact_id": target_id,
                    "steps": steps,
                    "generation": generation,
                    "note": note,
                },
                created_at=changed_at,
            )
        return AliasRecord(alias, target_id, generation, _parse_time(changed_at))

    def list_alias_history(
        self, alias: str, *, limit: int | None = None
    ) -> tuple[AliasHistoryRecord, ...]:
        alias = _validate_alias(alias)
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        query = (
            "SELECT id, alias, old_artifact_id, new_artifact_id, action, note, "
            "changed_at FROM alias_history WHERE alias = ? ORDER BY id DESC"
        )
        parameters: tuple[Any, ...] = (alias,)
        if limit is not None:
            query += " LIMIT ?"
            parameters += (limit,)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(
            AliasHistoryRecord(
                id=row["id"],
                alias=row["alias"],
                old_artifact_id=row["old_artifact_id"],
                new_artifact_id=row["new_artifact_id"],
                action=row["action"],
                note=row["note"],
                changed_at=_parse_time(row["changed_at"]),
            )
            for row in rows
        )

    def record_history(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        """Append a generic immutable audit event and return its row id."""

        _require_text(event_type, "event_type")
        _require_text(entity_type, "entity_type")
        _require_text(entity_id, "entity_id")
        with self._transaction() as connection:
            return self._record_history_connection(
                connection,
                event_type=event_type,
                entity_type=entity_type,
                entity_id=entity_id,
                payload=dict(payload or {}),
                created_at=_timestamp(),
            )

    def list_history(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[HistoryRecord, ...]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        clauses: list[str] = []
        parameters: list[Any] = []
        if entity_type is not None:
            clauses.append("entity_type = ?")
            parameters.append(entity_type)
        if entity_id is not None:
            clauses.append("entity_id = ?")
            parameters.append(entity_id)
        query = (
            "SELECT id, event_type, entity_type, entity_id, payload_json, "
            "created_at FROM history"
        )
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(
            HistoryRecord(
                id=row["id"],
                event_type=row["event_type"],
                entity_type=row["entity_type"],
                entity_id=row["entity_id"],
                payload=json.loads(row["payload_json"]),
                created_at=_parse_time(row["created_at"]),
            )
            for row in rows
        )

    def register_run(
        self,
        run_id: str,
        *,
        suite_id: str,
        deployment_id: str | None = None,
        artifact_id: str | None = None,
        status: str = "running",
        metadata: Mapping[str, Any] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> RunRecord:
        """Register a benchmark/runtime run without overwriting existing data."""

        _require_text(run_id, "run_id")
        _require_text(suite_id, "suite_id")
        _require_text(status, "status")
        metadata_value = dict(metadata or {})
        metadata_json = canonical_json(metadata_value)
        start_text = _format_time(started_at or datetime.now(timezone.utc))
        finish_text = _format_time(finished_at) if finished_at else None

        with self._transaction() as connection:
            if artifact_id is not None:
                self._require_artifact_connection(connection, artifact_id)
            existing = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is not None:
                same = (
                    existing["suite_id"] == suite_id
                    and existing["deployment_id"] == deployment_id
                    and existing["artifact_id"] == artifact_id
                    and existing["status"] == status
                    and existing["metadata_json"] == metadata_json
                    and (started_at is None or existing["started_at"] == start_text)
                    and (finished_at is None or existing["finished_at"] == finish_text)
                )
                if not same:
                    raise CatalogError(
                        f"run id {run_id!r} is already registered with "
                        "different content"
                    )
                return _run_record(existing)

            connection.execute(
                """
                INSERT INTO runs(
                    run_id, suite_id, deployment_id, artifact_id, status,
                    metadata_json, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    suite_id,
                    deployment_id,
                    artifact_id,
                    status,
                    metadata_json,
                    start_text,
                    finish_text,
                ),
            )
            self._record_history_connection(
                connection,
                event_type="run.registered",
                entity_type="run",
                entity_id=run_id,
                payload={
                    "suite_id": suite_id,
                    "deployment_id": deployment_id,
                    "artifact_id": artifact_id,
                    "status": status,
                },
                created_at=start_text,
            )
        return RunRecord(
            run_id=run_id,
            suite_id=suite_id,
            deployment_id=deployment_id,
            artifact_id=artifact_id,
            status=status,
            metadata=metadata_value,
            started_at=_parse_time(start_text),
            finished_at=_parse_time(finish_text) if finish_text else None,
        )

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise CatalogError(f"unknown run {run_id!r}")
        return _run_record(row)

    def list_runs(self, *, limit: int | None = None) -> tuple[RunRecord, ...]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        query = "SELECT * FROM runs ORDER BY started_at DESC, run_id"
        parameters: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return tuple(_run_record(row) for row in rows)

    def _require_artifact_connection(
        self, connection: sqlite3.Connection, artifact_id: str
    ) -> None:
        row = connection.execute(
            "SELECT 1 FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise CatalogError(f"unknown registered artifact {artifact_id!r}")

    @staticmethod
    def _insert_alias_history(
        connection: sqlite3.Connection,
        *,
        alias: str,
        old_artifact_id: str | None,
        new_artifact_id: str,
        action: str,
        note: str | None,
        changed_at: str,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO alias_history(
                alias, old_artifact_id, new_artifact_id, action, note, changed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (alias, old_artifact_id, new_artifact_id, action, note, changed_at),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _record_history_connection(
        connection: sqlite3.Connection,
        *,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: Mapping[str, Any],
        created_at: str,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO history(
                event_type, entity_type, entity_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_type,
                entity_type,
                entity_id,
                canonical_json(dict(payload)),
                created_at,
            ),
        )
        return int(cursor.lastrowid)


# A descriptive alias for callers that prefer to name the implementation.
SQLiteRegistry = Registry


def seal_manifest(
    manifest: ArtifactManifest | Mapping[str, Any],
) -> ArtifactManifest:
    """Validate a manifest and add/check its non-self-referential digest."""

    parsed = (
        manifest
        if isinstance(manifest, ArtifactManifest)
        else ArtifactManifest.model_validate(manifest)
    )
    expected = manifest_identity_sha256(parsed)
    if parsed.manifest_sha256 is not None and parsed.manifest_sha256 != expected:
        # Manifests emitted before the deterministic identity contract included
        # the local import timestamp. They remain verifiable/readable, while
        # all newly sealed manifests use the portable digest above.
        legacy_payload = parsed.model_dump(mode="json", exclude={"manifest_sha256"})
        legacy_expected = canonical_sha256(legacy_payload)
        if parsed.manifest_sha256 != legacy_expected:
            raise IntegrityError(
                f"manifest digest mismatch for artifact {parsed.artifact_id!r}: "
                f"expected {expected}, found {parsed.manifest_sha256}"
            )
        return parsed
    return parsed.model_copy(update={"manifest_sha256": expected})


def manifest_identity_sha256(
    manifest: ArtifactManifest | Mapping[str, Any],
) -> str:
    """Return the portable artifact identity, excluding observation time."""

    parsed = (
        manifest
        if isinstance(manifest, ArtifactManifest)
        else ArtifactManifest.model_validate(manifest)
    )
    payload = parsed.model_dump(
        mode="json",
        exclude={"manifest_sha256", "created_at"},
    )
    return canonical_sha256(payload)


def _validate_alias(alias: str) -> str:
    _require_text(alias, "alias")
    if len(alias) > 255:
        raise ValueError("alias must be at most 255 characters")
    if alias.startswith("/") or "\x00" in alias:
        raise ValueError("alias must be a relative, non-NUL name")
    if any(part in {"", ".", ".."} for part in alias.split("/")):
        raise ValueError("alias contains an unsafe path component")
    return alias


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _timestamp() -> str:
    return _format_time(datetime.now(timezone.utc))


def _format_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _alias_record(alias: str, row: sqlite3.Row) -> AliasRecord:
    return AliasRecord(
        alias=alias,
        artifact_id=row["artifact_id"],
        generation=row["generation"],
        updated_at=_parse_time(row["updated_at"]),
    )


def _run_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        suite_id=row["suite_id"],
        deployment_id=row["deployment_id"],
        artifact_id=row["artifact_id"],
        status=row["status"],
        metadata=json.loads(row["metadata_json"]),
        started_at=_parse_time(row["started_at"]),
        finished_at=(
            _parse_time(row["finished_at"]) if row["finished_at"] else None
        ),
    )
