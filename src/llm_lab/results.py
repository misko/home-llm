"""Immutable benchmark bundles and their append-only DuckDB index."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from .errors import BenchmarkError, IntegrityError, StoragePolicyError
from .hashing import canonical_sha256
from .paths import (
    ensure_safe_parent_directory,
    open_private_regular_file,
    open_safe_directory,
    validate_private_regular_file_if_present,
)


BUNDLE_FILES = ("run.json", "summary.json", "samples.jsonl", "telemetry.jsonl")
HASHED_BUNDLE_FILES = ("summary.json", "samples.jsonl", "telemetry.jsonl")
_REQUEST_CONTRACT_MODEL = "$MODEL_UNDER_TEST"

# Columns are appended and nullable so an existing results database can be
# upgraded without rewriting or assigning made-up values to historical rows.
_SAMPLE_PERFORMANCE_COLUMNS = (
    ("client_completion_tokens_per_second", "DOUBLE"),
    ("server_prompt_tokens", "BIGINT"),
    ("server_prompt_ms", "DOUBLE"),
    ("server_prompt_per_token_ms", "DOUBLE"),
    ("server_prompt_tokens_per_second", "DOUBLE"),
    ("server_predicted_tokens", "BIGINT"),
    ("server_predicted_ms", "DOUBLE"),
    ("server_predicted_per_token_ms", "DOUBLE"),
    ("server_predicted_tokens_per_second", "DOUBLE"),
    ("server_timings_json", "VARCHAR"),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"benchmark result is not JSON serializable: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def _jsonl_bytes(values: Iterable[Any]) -> bytes:
    chunks: list[bytes] = []
    for value in values:
        chunks.append(_json_bytes(value))
    return b"".join(chunks)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_exclusive(path: Path, content: bytes) -> None:
    # ``xb`` protects against an accidental overwrite even inside the staging
    # directory and makes the immutability contract explicit.
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def write_run_bundle(
    path: str | Path,
    run: Any,
    *,
    summary: Mapping[str, Any] | None = None,
    samples: Sequence[Any] | None = None,
    telemetry: Sequence[Any] | None = None,
) -> Path:
    """Atomically create the four-file immutable representation of a run.

    ``run`` may be a :class:`llm_lab.benchmark.BenchmarkRun` or a plain run
    mapping when the remaining payloads are passed explicitly.  Existing paths
    are never reused or overwritten.
    """

    target = Path(path).expanduser().resolve()
    if target.exists():
        raise BenchmarkError(f"run bundle already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(run, "run") and hasattr(run, "summary") and hasattr(run, "samples"):
        run_document = _jsonable(run.run)
        summary_document = _jsonable(run.summary if summary is None else summary)
        sample_values = run.samples if samples is None else samples
        telemetry_values = getattr(run, "telemetry", ()) if telemetry is None else telemetry
    else:
        run_document = _jsonable(run)
        if summary is None or samples is None:
            raise BenchmarkError("plain run mappings require summary and samples")
        summary_document = _jsonable(summary)
        sample_values = samples
        telemetry_values = telemetry or ()

    if not isinstance(run_document, Mapping):
        raise BenchmarkError("run metadata must be a JSON object")
    run_document = dict(run_document)
    # A bundle descriptor is derived output, never caller-controlled input.
    run_document.pop("bundle", None)
    run_id = run_document.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise BenchmarkError("run metadata requires a non-empty run_id")
    if not isinstance(summary_document, Mapping):
        raise BenchmarkError("summary must be a JSON object")
    summary_document = dict(summary_document)
    summary_run_id = summary_document.get("run_id")
    if summary_run_id is None:
        summary_document["run_id"] = run_id
    elif summary_run_id != run_id:
        raise BenchmarkError("summary run_id does not match bundle run_id")

    normalized_samples = [_jsonable(item) for item in sample_values]
    for item in normalized_samples:
        if not isinstance(item, Mapping):
            raise BenchmarkError("every sample must be a JSON object")
        if item.get("run_id") != run_id:
            raise BenchmarkError("sample run_id does not match bundle run_id")
    normalized_telemetry = [_jsonable(item) for item in telemetry_values]
    for item in normalized_telemetry:
        if not isinstance(item, Mapping):
            raise BenchmarkError("every telemetry sample must be a JSON object")

    payloads = {
        "summary.json": _json_bytes(summary_document),
        "samples.jsonl": _jsonl_bytes(normalized_samples),
        "telemetry.jsonl": _jsonl_bytes(normalized_telemetry),
    }
    run_metadata_sha256 = _sha256(_json_bytes(run_document))
    run_document["bundle"] = {
        "format_version": 1,
        "run_metadata_sha256": run_metadata_sha256,
        "sample_count": len(normalized_samples),
        "telemetry_sample_count": len(normalized_telemetry),
        "files": {
            name: {"sha256": _sha256(content), "size_bytes": len(content)}
            for name, content in payloads.items()
        },
    }
    payloads["run.json"] = _json_bytes(run_document)

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        for name in BUNDLE_FILES:
            _write_exclusive(staging / name, payloads[name])
            os.chmod(staging / name, 0o444, follow_symlinks=False)
        staging_fd = os.open(
            staging,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        os.chmod(staging, 0o555, follow_symlinks=False)
        # The rename is atomic on the bundle's filesystem.  It also refuses to
        # replace an existing non-empty directory if another writer won a race.
        if target.exists():
            raise BenchmarkError(f"run bundle already exists: {target}")
        os.rename(staging, target)
        parent_fd = os.open(
            target.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if staging.exists():
            os.chmod(staging, 0o700, follow_symlinks=False)
            shutil.rmtree(staging)
        raise
    return target


@dataclass(frozen=True, slots=True)
class BundleVerification:
    path: Path
    run_id: str
    sample_count: int
    telemetry_sample_count: int


def _parse_json(name: str, content: bytes) -> Any:
    try:
        return json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read {name}: {exc}") from exc


def _parse_jsonl(name: str, content: bytes) -> list[Any]:
    values: list[Any] = []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntegrityError(f"cannot read {name}: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise IntegrityError(
                f"invalid JSON in {name} at line {line_number}: {exc}"
            ) from exc
    return values


def _read_bundle_payloads(
    bundle: Path,
    *,
    require_immutable: bool = True,
) -> dict[str, bytes]:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(bundle, directory_flags)
    except OSError as exc:
        raise IntegrityError(f"run bundle is not a safe directory: {bundle}: {exc}") from exc
    try:
        directory_metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise IntegrityError(f"run bundle is not a directory: {bundle}")
        if require_immutable and stat.S_IMODE(directory_metadata.st_mode) & 0o222:
            raise IntegrityError(f"run bundle directory is writable: {bundle}")
        try:
            actual_names = set(os.listdir(directory_fd))
        except OSError as exc:
            raise IntegrityError(f"cannot list run bundle {bundle}: {exc}") from exc
        missing = set(BUNDLE_FILES) - actual_names
        extras = actual_names - set(BUNDLE_FILES)
        if missing or extras:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if extras:
                details.append(f"unexpected {sorted(extras)}")
            raise IntegrityError("invalid run bundle: " + "; ".join(details))

        payloads: dict[str, bytes] = {}
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        for name in BUNDLE_FILES:
            try:
                descriptor = os.open(name, file_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise IntegrityError(f"cannot safely open bundle member {name}: {exc}") from exc
            try:
                before = os.fstat(descriptor)
                named_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or not stat.S_ISREG(named_before.st_mode)
                    or before.st_nlink != 1
                    or (
                        require_immutable
                        and stat.S_IMODE(before.st_mode) & 0o222
                    )
                    or (before.st_dev, before.st_ino)
                    != (named_before.st_dev, named_before.st_ino)
                ):
                    raise IntegrityError(f"bundle member is not a private regular file: {name}")
                chunks: list[bytes] = []
                while block := os.read(descriptor, 1024 * 1024):
                    chunks.append(block)
                content = b"".join(chunks)
                after = os.fstat(descriptor)
                named_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                before_identity = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                if (
                    before_identity != after_identity
                    or (after.st_dev, after.st_ino)
                    != (named_after.st_dev, named_after.st_ino)
                    or len(content) != after.st_size
                ):
                    raise IntegrityError(f"bundle member changed while reading: {name}")
                payloads[name] = content
            except OSError as exc:
                raise IntegrityError(f"cannot read bundle member {name}: {exc}") from exc
            finally:
                os.close(descriptor)
        return payloads
    finally:
        os.close(directory_fd)


def _verify_bundle_payloads(
    bundle: Path, payloads: Mapping[str, bytes]
) -> tuple[BundleVerification, dict[str, Any]]:
    run = _parse_json("run.json", payloads["run.json"])
    summary = _parse_json("summary.json", payloads["summary.json"])
    samples = _parse_jsonl("samples.jsonl", payloads["samples.jsonl"])
    telemetry = _parse_jsonl("telemetry.jsonl", payloads["telemetry.jsonl"])
    if not isinstance(run, Mapping):
        raise IntegrityError("run.json must contain an object")
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise IntegrityError("run.json has no valid run_id")
    suite = run.get("suite")
    if isinstance(suite, Mapping) and (
        "sha256" in suite or "definition" in suite
    ):
        suite_digest = suite.get("sha256")
        suite_definition = suite.get("definition")
        if not isinstance(suite_digest, str) or not isinstance(
            suite_definition, Mapping
        ):
            raise IntegrityError(
                "run.json suite identity requires both sha256 and definition"
            )
        if canonical_sha256(suite_definition) != suite_digest:
            raise IntegrityError("run.json suite content digest mismatch")
    request_contract = run.get("request_contract")
    contract_requests_by_case: dict[str, Mapping[str, Any]] | None = None
    if request_contract is not None:
        if not isinstance(request_contract, Mapping):
            raise IntegrityError("run.json request contract must be an object")
        contract_digest = request_contract.get("sha256")
        contract_definition = request_contract.get("definition")
        if (
            request_contract.get("schema_version") != 1
            or not isinstance(contract_digest, str)
            or not isinstance(contract_definition, Mapping)
        ):
            raise IntegrityError(
                "run.json request contract requires schema_version, sha256 and definition"
            )
        if canonical_sha256(contract_definition) != contract_digest:
            raise IntegrityError("run.json request contract digest mismatch")
        contract_requests = contract_definition.get("requests")
        if not isinstance(contract_requests, list):
            raise IntegrityError("run.json request contract has no request list")
        contract_requests_by_case = {}
        for item in contract_requests:
            if not isinstance(item, Mapping):
                raise IntegrityError("run.json request contract entry must be an object")
            case_id = item.get("case_id")
            body = item.get("body")
            if (
                not isinstance(case_id, str)
                or not case_id
                or not isinstance(body, Mapping)
                or case_id in contract_requests_by_case
            ):
                raise IntegrityError("run.json request contract entry is invalid")
            if body.get("model") != _REQUEST_CONTRACT_MODEL:
                raise IntegrityError(
                    "run.json request contract must normalize the served model"
                )
            contract_requests_by_case[case_id] = body
    bundle_metadata = run.get("bundle")
    files = bundle_metadata.get("files") if isinstance(bundle_metadata, Mapping) else None
    if not isinstance(files, Mapping):
        raise IntegrityError("run.json has no bundle file manifest")
    if bundle_metadata.get("format_version") != 1:
        raise IntegrityError("unsupported run bundle format")
    unhashed_run = dict(run)
    unhashed_run.pop("bundle", None)
    if bundle_metadata.get("run_metadata_sha256") != _sha256(_json_bytes(unhashed_run)):
        raise IntegrityError("SHA-256 mismatch for run metadata")

    for name in HASHED_BUNDLE_FILES:
        recorded = files.get(name)
        if not isinstance(recorded, Mapping):
            raise IntegrityError(f"run.json has no manifest entry for {name}")
        content = payloads[name]
        if recorded.get("size_bytes") != len(content):
            raise IntegrityError(f"size mismatch for {name}")
        if recorded.get("sha256") != _sha256(content):
            raise IntegrityError(f"SHA-256 mismatch for {name}")

    if not isinstance(summary, Mapping):
        raise IntegrityError("summary.json must contain an object")
    if summary.get("run_id") != run_id:
        raise IntegrityError("summary run_id does not match run.json")
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("run_id") != run_id:
            raise IntegrityError("sample run_id does not match run.json")
        if contract_requests_by_case is not None:
            case_id = sample.get("case_id")
            request = sample.get("request")
            expected_request = contract_requests_by_case.get(case_id)
            if not isinstance(request, Mapping) or expected_request is None:
                raise IntegrityError(
                    "sample request is absent from the effective request contract"
                )
            normalized_request = dict(request)
            normalized_request["model"] = _REQUEST_CONTRACT_MODEL
            if canonical_sha256(normalized_request) != canonical_sha256(
                expected_request
            ):
                raise IntegrityError(
                    "sample request does not match the effective request contract"
                )
    expected_samples = bundle_metadata.get("sample_count")
    expected_telemetry = bundle_metadata.get("telemetry_sample_count")
    if expected_samples != len(samples):
        raise IntegrityError("sample count does not match run.json")
    if expected_telemetry != len(telemetry):
        raise IntegrityError("telemetry count does not match run.json")
    verification = BundleVerification(bundle, run_id, len(samples), len(telemetry))
    loaded = {
        "path": str(bundle),
        "run": run,
        "summary": summary,
        "samples": samples,
        "telemetry": telemetry,
    }
    return verification, loaded


def verify_run_bundle(path: str | Path) -> BundleVerification:
    """Verify structure and hashes from one stable read of every bundle file."""

    bundle = Path(path).expanduser().absolute()
    payloads = _read_bundle_payloads(bundle)
    verification, _ = _verify_bundle_payloads(bundle, payloads)
    return verification


def repair_run_bundle_permissions(path: str | Path) -> BundleVerification:
    """Verify a legacy writable bundle snapshot, then seal it read-only."""

    bundle = Path(path).expanduser().absolute()
    payloads = _read_bundle_payloads(bundle, require_immutable=False)
    _verify_bundle_payloads(bundle, payloads)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(bundle, directory_flags)
    try:
        for name in BUNDLE_FILES:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or (metadata.st_dev, metadata.st_ino)
                    != (named.st_dev, named.st_ino)
                ):
                    raise IntegrityError(
                        f"bundle member is not a private regular file: {name}"
                    )
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fchmod(directory_fd, 0o555)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return verify_run_bundle(bundle)


def load_run_bundle(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Load one stable snapshot of a run bundle into JSON-compatible values."""

    bundle = Path(path).expanduser().absolute()
    payloads = _read_bundle_payloads(bundle)
    if verify:
        _, loaded = _verify_bundle_payloads(bundle, payloads)
        return loaded
    return {
        "path": str(bundle),
        "run": _parse_json("run.json", payloads["run.json"]),
        "summary": _parse_json("summary.json", payloads["summary.json"]),
        "samples": _parse_jsonl("samples.jsonl", payloads["samples.jsonl"]),
        "telemetry": _parse_jsonl("telemetry.jsonl", payloads["telemetry.jsonl"]),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    )


def _nested_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("id")
        return str(candidate) if candidate is not None else None
    return None


class ResultsStore:
    """Append-only DuckDB index over immutable run bundles."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = ensure_safe_parent_directory(path, purpose="results database")
        try:
            self.connection = self._connect_safely(read_only=read_only)
        except StoragePolicyError:
            raise
        except Exception as exc:
            raise BenchmarkError(
                f"could not safely open results database {self.path}: {exc}"
            ) from exc
        if not read_only:
            self._initialize()

    def _connect_safely(self, *, read_only: bool) -> duckdb.DuckDBPyConnection:
        """Open an existing safe inode or atomically publish a new database."""

        self._validate_sidecars()
        try:
            target, descriptor, opened = open_private_regular_file(
                self.path,
                os.O_RDONLY if read_only else os.O_RDWR,
                purpose="results database",
            )
        except StoragePolicyError:
            raise
        except FileNotFoundError:
            if read_only:
                raise BenchmarkError(f"results database does not exist: {self.path}")
            return self._create_database_exclusively()

        try:
            connection = duckdb.connect(str(target), read_only=read_only)
            self._validate_sidecars()
            named = os.lstat(target)
            if (
                named.st_nlink != 1
                or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
            ):
                connection.close()
                raise IntegrityError(
                    f"results database path changed while opening: {target}"
                )
            return connection
        finally:
            os.close(descriptor)

    def _create_database_exclusively(self) -> duckdb.DuckDBPyConnection:
        """Create via an unpredictable sibling and publish without replacement."""

        candidate = self.path.parent / (
            f".{self.path.name}.new-{os.getpid()}-{secrets.token_hex(12)}"
        )
        connection: duckdb.DuckDBPyConnection | None = None
        published = False
        candidate_descriptor: int | None = None
        directory_fd: int | None = None
        try:
            # DuckDB keys its WAL/lock identity to the opened pathname.  Close
            # the candidate-named connection before publishing, then reopen the
            # final pathname so two processes can never see divergent WALs for
            # one underlying inode.
            connection = duckdb.connect(str(candidate))
            connection.close()
            connection = None
            _, candidate_descriptor, candidate_stat = open_private_regular_file(
                candidate,
                os.O_RDWR,
                purpose="results database candidate",
            )
            try:
                _, directory_fd = open_safe_directory(
                    self.path.parent,
                    purpose="results database parent",
                )
                try:
                    os.link(
                        candidate.name,
                        self.path.name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError as exc:
                    raise StoragePolicyError(
                        "refusing to replace existing results database target: "
                        f"{self.path}"
                    ) from exc
                named = os.stat(
                    self.path.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(named.st_mode)
                    or named.st_nlink != 2
                    or (named.st_dev, named.st_ino)
                    != (candidate_stat.st_dev, candidate_stat.st_ino)
                ):
                    raise IntegrityError(
                        "results database publication changed unexpectedly: "
                        f"{self.path}"
                    )
                os.unlink(candidate.name, dir_fd=directory_fd)
                published = True
                # Seal the inode already opened and validated above.  A pathname
                # chmod here would follow a replacement symlink in the interval
                # after publication validation.
                os.fchmod(candidate_descriptor, 0o600)
                os.fsync(candidate_descriptor)
                named = os.stat(
                    self.path.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(named.st_mode)
                    or named.st_nlink != 1
                    or (named.st_dev, named.st_ino)
                    != (candidate_stat.st_dev, candidate_stat.st_ino)
                ):
                    raise IntegrityError(
                        "results database path changed while sealing: "
                        f"{self.path}"
                    )
                os.fsync(directory_fd)
            finally:
                if candidate_descriptor is not None:
                    os.close(candidate_descriptor)
                    candidate_descriptor = None
                if directory_fd is not None:
                    os.close(directory_fd)
                    directory_fd = None
            target, descriptor, opened = open_private_regular_file(
                self.path,
                os.O_RDWR,
                purpose="results database",
            )
            try:
                self._validate_sidecars()
                connection = duckdb.connect(str(target))
                self._validate_sidecars()
                named = os.lstat(target)
                if (
                    named.st_nlink != 1
                    or (named.st_dev, named.st_ino)
                    != (opened.st_dev, opened.st_ino)
                ):
                    connection.close()
                    connection = None
                    raise IntegrityError(
                        f"results database path changed while opening: {target}"
                    )
            finally:
                os.close(descriptor)
            assert connection is not None
            return connection
        except BaseException:
            if connection is not None:
                connection.close()
            if published:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
            raise
        finally:
            if candidate_descriptor is not None:
                os.close(candidate_descriptor)
            if directory_fd is not None:
                os.close(directory_fd)
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    def _validate_sidecars(self) -> None:
        validate_private_regular_file_if_present(
            Path(f"{self.path}.wal"),
            purpose="results database WAL",
        )

    def __enter__(self) -> "ResultsStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self._validate_sidecars()
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                started_at VARCHAR,
                finished_at VARCHAR,
                status VARCHAR,
                model_id VARCHAR,
                artifact_id VARCHAR,
                deployment_id VARCHAR,
                suite_id VARCHAR,
                suite_version VARCHAR,
                sample_count BIGINT,
                error_count BIGINT,
                pass_rate DOUBLE,
                run_json VARCHAR NOT NULL,
                summary_json VARCHAR NOT NULL,
                bundle_path VARCHAR NOT NULL,
                indexed_at VARCHAR NOT NULL
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS samples (
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
                client_completion_tokens_per_second DOUBLE,
                server_prompt_tokens BIGINT,
                server_prompt_ms DOUBLE,
                server_prompt_per_token_ms DOUBLE,
                server_prompt_tokens_per_second DOUBLE,
                server_predicted_tokens BIGINT,
                server_predicted_ms DOUBLE,
                server_predicted_per_token_ms DOUBLE,
                server_predicted_tokens_per_second DOUBLE,
                server_timings_json VARCHAR,
                PRIMARY KEY (run_id, case_id, repetition)
            )
        """)
        sample_columns = {
            row[1]
            for row in self.connection.execute(
                "PRAGMA table_info('samples')"
            ).fetchall()
        }
        for name, sql_type in _SAMPLE_PERFORMANCE_COLUMNS:
            if name not in sample_columns:
                self.connection.execute(
                    f"ALTER TABLE samples ADD COLUMN {name} {sql_type}"
                )
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                run_id VARCHAR NOT NULL,
                timestamp VARCHAR,
                available BOOLEAN,
                gpu_index INTEGER,
                gpu_uuid VARCHAR,
                gpu_name VARCHAR,
                gpu_utilization_percent DOUBLE,
                memory_used_mib DOUBLE,
                memory_total_mib DOUBLE,
                power_draw_w DOUBLE,
                sample_json VARCHAR NOT NULL
            )
        """)
        self.connection.execute("CREATE INDEX IF NOT EXISTS runs_model_idx ON runs(model_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS runs_suite_idx ON runs(suite_id, suite_version)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS samples_case_idx ON samples(case_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS telemetry_run_idx ON telemetry(run_id)")

    def append_bundle(self, path: str | Path) -> str:
        """Verify and append one bundle; duplicate run IDs are rejected."""

        bundle = load_run_bundle(path, verify=True)
        run = bundle["run"]
        summary = bundle["summary"]
        samples = bundle["samples"]
        telemetry = bundle["telemetry"]
        run_id = str(run["run_id"])
        if self.connection.execute(
            "SELECT count(*) FROM runs WHERE run_id = ?", [run_id]
        ).fetchone()[0]:
            raise BenchmarkError(f"run is already indexed: {run_id}")

        suite = run.get("suite") if isinstance(run.get("suite"), Mapping) else {}
        transaction_started = False
        try:
            self._validate_sidecars()
            self.connection.execute("BEGIN TRANSACTION")
            transaction_started = True
            self.connection.execute(
                """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    run_id,
                    run.get("started_at"),
                    run.get("finished_at"),
                    run.get("status"),
                    _nested_id(run.get("model")),
                    _nested_id(run.get("artifact")),
                    _nested_id(run.get("deployment")),
                    suite.get("id"),
                    suite.get("version"),
                    summary.get("sample_count"),
                    summary.get("error_count"),
                    summary.get("pass_rate"),
                    _canonical_json(run),
                    _canonical_json(summary),
                    bundle["path"],
                    _utc_now(),
                ],
            )
            for sample in samples:
                usage = sample.get("usage") if isinstance(sample.get("usage"), Mapping) else {}
                error = sample.get("error") if isinstance(sample.get("error"), Mapping) else {}
                server_timings = (
                    sample.get("server_timings")
                    if isinstance(sample.get("server_timings"), Mapping)
                    else {}
                )
                self.connection.execute(
                    """
                    INSERT INTO samples (
                        run_id, case_id, repetition, started_at, latency_ms,
                        passed, prompt_tokens, completion_tokens, total_tokens,
                        error_type, error_message, request_json, response_json,
                        scores_json, client_completion_tokens_per_second,
                        server_prompt_tokens, server_prompt_ms,
                        server_prompt_per_token_ms,
                        server_prompt_tokens_per_second,
                        server_predicted_tokens, server_predicted_ms,
                        server_predicted_per_token_ms,
                        server_predicted_tokens_per_second, server_timings_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        run_id,
                        sample.get("case_id"),
                        sample.get("repetition"),
                        sample.get("started_at"),
                        sample.get("latency_ms"),
                        sample.get("passed"),
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        usage.get("total_tokens"),
                        error.get("type"),
                        error.get("message"),
                        _canonical_json(sample.get("request") or {}),
                        _canonical_json(sample.get("response")) if sample.get("response") is not None else None,
                        _canonical_json(sample.get("scores") or []),
                        sample.get("client_completion_tokens_per_second"),
                        server_timings.get("prompt_tokens"),
                        server_timings.get("prompt_ms"),
                        server_timings.get("prompt_per_token_ms"),
                        server_timings.get("prompt_tokens_per_second"),
                        server_timings.get("predicted_tokens"),
                        server_timings.get("predicted_ms"),
                        server_timings.get("predicted_per_token_ms"),
                        server_timings.get("predicted_tokens_per_second"),
                        (
                            _canonical_json(server_timings)
                            if server_timings
                            else None
                        ),
                    ],
                )
            for item in telemetry:
                self.connection.execute(
                    """INSERT INTO telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        run_id,
                        item.get("timestamp"),
                        item.get("available"),
                        item.get("index"),
                        item.get("uuid"),
                        item.get("name"),
                        item.get("gpu_utilization_percent"),
                        item.get("memory_used_mib"),
                        item.get("memory_total_mib"),
                        item.get("power_draw_w"),
                        _canonical_json(item),
                    ],
                )
            self.connection.execute("COMMIT")
            transaction_started = False
        except Exception:
            if transaction_started:
                self.connection.execute("ROLLBACK")
            raise
        return run_id

    # “Index” is the user-facing operation; “append” documents the DB policy.
    index_bundle = append_bundle

    def run_ids(self) -> list[str]:
        return [row[0] for row in self.connection.execute(
            "SELECT run_id FROM runs ORDER BY started_at, run_id"
        ).fetchall()]

    def compare_rows(self, run_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Return long-form, per-case comparison rows for selected runs."""

        if not run_ids:
            raise BenchmarkError("at least one run_id is required")
        placeholders = ",".join("?" for _ in run_ids)
        available_columns = {
            row[1]
            for row in self.connection.execute(
                "PRAGMA table_info('samples')"
            ).fetchall()
        }

        # A read-only connection may point at a database created before these
        # columns existed.  Typed NULL expressions keep comparisons working
        # while accurately representing that no measurement was recorded.
        metric_types = dict(_SAMPLE_PERFORMANCE_COLUMNS)
        numeric_metrics = tuple(
            name for name, _ in _SAMPLE_PERFORMANCE_COLUMNS
            if name != "server_timings_json"
        )
        metric_projections = ",\n                ".join(
            (
                f"s.{name} AS {name}"
                if name in available_columns
                else f"CAST(NULL AS {metric_types[name]}) AS {name}"
            )
            for name in numeric_metrics
        )
        distribution_metrics = (
            "client_completion_tokens_per_second",
            "server_prompt_ms",
            "server_prompt_per_token_ms",
            "server_prompt_tokens_per_second",
            "server_predicted_ms",
            "server_predicted_per_token_ms",
            "server_predicted_tokens_per_second",
        )
        metric_aggregates = ",\n                ".join(
            expression
            for name in distribution_metrics
            for expression in (
                f"count(s.{name}) AS {name}_sample_count",
                f"avg(s.{name}) AS mean_{name}",
                f"quantile_cont(s.{name}, 0.5) AS p50_{name}",
                f"quantile_cont(s.{name}, 0.95) AS p95_{name}",
            )
        )
        query = f"""
            WITH sample_metrics AS (
                SELECT
                    s.run_id,
                    s.case_id,
                    s.error_type,
                    s.passed,
                    s.latency_ms,
                    s.prompt_tokens,
                    s.completion_tokens,
                    s.total_tokens,
                    {metric_projections}
                FROM samples s
            )
            SELECT
                r.run_id,
                r.model_id,
                r.artifact_id,
                r.deployment_id,
                r.suite_id,
                r.suite_version,
                s.case_id,
                count(*) AS sample_count,
                sum(CASE WHEN s.error_type IS NOT NULL THEN 1 ELSE 0 END) AS error_count,
                count(s.passed) AS scoreable_sample_count,
                avg(CASE WHEN s.passed THEN 1.0 WHEN s.passed = false THEN 0.0 END) AS pass_rate,
                avg(s.latency_ms) AS mean_latency_ms,
                quantile_cont(s.latency_ms, 0.5) AS p50_latency_ms,
                quantile_cont(s.latency_ms, 0.95) AS p95_latency_ms,
                sum(s.prompt_tokens) AS prompt_tokens,
                sum(s.completion_tokens) AS completion_tokens,
                sum(s.total_tokens) AS total_tokens,
                sum(s.server_prompt_tokens) AS server_prompt_tokens,
                sum(s.server_predicted_tokens) AS server_predicted_tokens,
                {metric_aggregates}
            FROM runs r
            JOIN sample_metrics s USING (run_id)
            WHERE r.run_id IN ({placeholders})
            GROUP BY ALL
            ORDER BY s.case_id, r.run_id
        """
        cursor = self.connection.execute(query, list(run_ids))
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def compare_runs(self, run_ids: Sequence[str]) -> dict[str, Any]:
        """Compare runs per task, suppressing composites for missing tasks.

        The returned task matrix contains an explicit ``present`` marker for
        every run/task pair.  A composite is provided only when every selected
        run has exactly the same set of scoreable tasks; it is otherwise
        ``None`` rather than a misleading average over different workloads.
        """

        selected = list(dict.fromkeys(run_ids))
        if len(selected) < 2:
            raise BenchmarkError("comparison requires at least two distinct runs")
        placeholders = ",".join("?" for _ in selected)
        selected_metadata = self.connection.execute(
            f"SELECT run_id, suite_id, suite_version, run_json, summary_json, "
            f"status, error_count FROM runs "
            f"WHERE run_id IN ({placeholders})",
            selected,
        ).fetchall()
        found = {row[0] for row in selected_metadata}
        missing_runs = [run_id for run_id in selected if run_id not in found]
        if missing_runs:
            raise BenchmarkError(f"unknown run IDs: {missing_runs}")

        rows = self.compare_rows(selected)
        by_key = {(row["run_id"], row["case_id"]): row for row in rows}
        case_sets = {
            run_id: {row["case_id"] for row in rows if row["run_id"] == run_id}
            for run_id in selected
        }
        all_cases = sorted(set().union(*case_sets.values()))
        tasks: list[dict[str, Any]] = []
        for case_id in all_cases:
            task_runs: list[dict[str, Any]] = []
            for run_id in selected:
                row = by_key.get((run_id, case_id))
                task_runs.append({"run_id": run_id, "present": row is not None, **(row or {})})
            tasks.append({
                "case_id": case_id,
                "present_in_all_runs": all(item["present"] for item in task_runs),
                "runs": task_runs,
            })

        same_tasks = all(case_sets[run_id] == case_sets[selected[0]] for run_id in selected[1:])
        suite_identities: set[tuple[str | None, str | None, str | None]] = set()
        missing_suite_digest = False
        request_contract_identities: set[str] = set()
        missing_request_contract_digest = False
        complete_workloads = True
        complete_warmup_accounting = True
        warmup_errors_present = False
        for row in selected_metadata:
            try:
                run_document = json.loads(row[3])
            except (TypeError, json.JSONDecodeError):
                run_document = {}
            suite_document = (
                run_document.get("suite")
                if isinstance(run_document, Mapping)
                and isinstance(run_document.get("suite"), Mapping)
                else {}
            )
            suite_digest = suite_document.get("sha256")
            if not isinstance(suite_digest, str) or not suite_digest:
                missing_suite_digest = True
                suite_digest = None
            suite_identities.add((row[1], row[2], suite_digest))
            request_contract = (
                run_document.get("request_contract")
                if isinstance(run_document, Mapping)
                and isinstance(run_document.get("request_contract"), Mapping)
                else {}
            )
            request_contract_digest = request_contract.get("sha256")
            request_contract_definition = request_contract.get("definition")
            if (
                request_contract.get("schema_version") != 1
                or not isinstance(request_contract_digest, str)
                or not request_contract_digest
                or not isinstance(request_contract_definition, Mapping)
                or canonical_sha256(request_contract_definition)
                != request_contract_digest
            ):
                missing_request_contract_digest = True
            else:
                request_contract_identities.add(request_contract_digest)
            definition = suite_document.get("definition")
            defined_cases = (
                definition.get("cases")
                if isinstance(definition, Mapping)
                and isinstance(definition.get("cases"), list)
                else None
            )
            expected_case_ids = (
                {
                    str(case["id"])
                    for case in defined_cases
                    if isinstance(case, Mapping) and "id" in case
                }
                if defined_cases is not None
                else set()
            )
            try:
                summary_document = json.loads(row[4])
            except (TypeError, json.JSONDecodeError):
                summary_document = {}
            skipped_cases = (
                summary_document.get("skipped_cases")
                if isinstance(summary_document, Mapping)
                else None
            )
            warmup_error_count = (
                summary_document.get("warmup_error_count")
                if isinstance(summary_document, Mapping)
                else None
            )
            if (
                not isinstance(warmup_error_count, int)
                or isinstance(warmup_error_count, bool)
                or warmup_error_count < 0
            ):
                complete_warmup_accounting = False
            elif warmup_error_count:
                warmup_errors_present = True
            if (
                defined_cases is None
                or not expected_case_ids
                or case_sets[row[0]] != expected_case_ids
                or not isinstance(skipped_cases, list)
                or bool(skipped_cases)
            ):
                complete_workloads = False
        same_suite = len(suite_identities) == 1 and not missing_suite_digest
        same_request_contract = (
            len(request_contract_identities) == 1
            and not missing_request_contract_digest
        )
        all_scoreable = bool(rows) and all(row["scoreable_sample_count"] > 0 for row in rows)
        clean_runs = (
            complete_warmup_accounting
            and not warmup_errors_present
            and all(
                row[5] == "completed" and row[6] == 0
                for row in selected_metadata
            )
        )
        composite: dict[str, Any] | None = None
        reason: str | None = None
        if (
            same_suite
            and same_request_contract
            and same_tasks
            and complete_workloads
            and all_scoreable
            and clean_runs
        ):
            composite_runs: list[dict[str, Any]] = []
            for run_id in selected:
                run_rows = [row for row in rows if row["run_id"] == run_id]
                # Equal weighting by task avoids a task with more repetitions
                # silently dominating the comparison.
                composite_runs.append({
                    "run_id": run_id,
                    "mean_task_pass_rate": sum(row["pass_rate"] for row in run_rows) / len(run_rows),
                    "mean_task_latency_ms": sum(row["mean_latency_ms"] for row in run_rows) / len(run_rows),
                })
            composite = {"case_ids": all_cases, "runs": composite_runs}
        elif missing_suite_digest:
            reason = "one or more runs lack a verifiable suite content digest"
        elif not same_suite:
            reason = "selected runs do not use identical suite content"
        elif missing_request_contract_digest:
            reason = "one or more runs lack a verifiable effective request contract"
        elif not same_request_contract:
            reason = "selected runs do not use the same effective request contract"
        elif not same_tasks:
            reason = "selected runs do not contain the same task set"
        elif not complete_workloads:
            reason = "one or more runs skipped or omitted suite tasks"
        elif not complete_warmup_accounting:
            reason = "one or more runs lack trustworthy warmup error accounting"
        elif not clean_runs:
            reason = (
                "one or more runs contains request or warmup errors or did not "
                "complete cleanly"
            )
        else:
            reason = "one or more tasks have no scorer"
        return {
            "run_ids": selected,
            "tasks": tasks,
            "composite": composite,
            "composite_unavailable_reason": reason,
        }

    append = append_bundle
    compare = compare_runs


# Compatibility/readability aliases.
ResultsDB = ResultsStore
ResultsDatabase = ResultsStore
DuckDBResults = ResultsStore


def index_run_bundle(database: str | Path, bundle: str | Path) -> str:
    with ResultsStore(database) as store:
        return store.append_bundle(bundle)


def compare_runs(database: str | Path, run_ids: Sequence[str]) -> dict[str, Any]:
    with ResultsStore(database) as store:
        return store.compare_runs(run_ids)
