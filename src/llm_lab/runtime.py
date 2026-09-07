"""Runtime lifecycle management for local and external inference servers.

The module deliberately keeps process creation behind small launcher protocols.  The
real launchers never invoke a shell, while tests and callers can inject deterministic
launchers and readiness checks.  A file lock serializes all GPU/state transitions and
the active deployment is stored as an atomically replaced JSON document.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
import psutil

from .errors import DeploymentError, StoragePolicyError
from .paths import LabPaths, open_regular_file_beneath, open_safe_directory
from .schema import BackendKind, DeploymentSpec, RuntimeLockSpec


LaunchKind = Literal["process", "container", "external"]
StatePhase = Literal["starting", "ready", "stopping", "failed"]

_CODE_LOADING_ENVIRONMENT = frozenset(
    {
        "GGML_BACKEND_PATH",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
    }
)
_LOCKED_ENVIRONMENT_BASE = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": os.defpath,
    "TZ": "UTC",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _connect_host(host: str) -> str:
    """Return a locally connectable address for wildcard bind addresses."""

    if host in {"0.0.0.0", "::", "[::]", ""}:
        return "127.0.0.1"
    return host


def _base_url(spec: DeploymentSpec) -> str:
    if spec.backend == BackendKind.EXTERNAL:
        assert spec.external_base_url is not None
        return spec.external_base_url.rstrip("/")
    return f"http://{_connect_host(spec.host)}:{spec.port}"


def health_url(base_url: str, health_path: str = "/health") -> str:
    """Resolve a health endpoint, treating a leading slash as origin-relative."""

    return urljoin(f"{base_url.rstrip('/')}/", health_path)


@dataclass(frozen=True, slots=True)
class VolumeMount:
    source: str
    target: str
    read_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "read_only": self.read_only,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VolumeMount":
        return cls(
            source=str(value["source"]),
            target=str(value["target"]),
            read_only=bool(value.get("read_only", True)),
        )


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    """A fully materialized, serializable launch request."""

    kind: LaunchKind
    deployment_id: str
    command: tuple[str, ...]
    environment: dict[str, str]
    base_url: str
    health_url: str
    log_path: str | None = None
    working_directory: str | None = None
    image: str | None = None
    mounts: tuple[VolumeMount, ...] = ()
    publish_host: str | None = None
    publish_port: int | None = None
    container_port: int | None = None
    expected_executable_sha256: str | None = None
    expected_executable_version_contains: str | None = None
    expected_executable_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "deployment_id": self.deployment_id,
            "command": list(self.command),
            "environment": dict(sorted(self.environment.items())),
            "base_url": self.base_url,
            "health_url": self.health_url,
            "log_path": self.log_path,
            "working_directory": self.working_directory,
            "image": self.image,
            "mounts": [mount.to_dict() for mount in self.mounts],
            "publish_host": self.publish_host,
            "publish_port": self.publish_port,
            "container_port": self.container_port,
            "expected_executable_sha256": self.expected_executable_sha256,
            "expected_executable_version_contains": (
                self.expected_executable_version_contains
            ),
            "expected_executable_root": self.expected_executable_root,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LaunchPlan":
        return cls(
            kind=str(value["kind"]),  # type: ignore[arg-type]
            deployment_id=str(value["deployment_id"]),
            command=tuple(str(item) for item in value.get("command", ())),
            environment={
                str(key): str(item)
                for key, item in dict(value.get("environment", {})).items()
            },
            base_url=str(value["base_url"]),
            health_url=str(value["health_url"]),
            log_path=(
                None if value.get("log_path") is None else str(value["log_path"])
            ),
            working_directory=(
                None
                if value.get("working_directory") is None
                else str(value["working_directory"])
            ),
            image=None if value.get("image") is None else str(value["image"]),
            mounts=tuple(
                VolumeMount.from_dict(item) for item in value.get("mounts", ())
            ),
            publish_host=(
                None
                if value.get("publish_host") is None
                else str(value["publish_host"])
            ),
            publish_port=(
                None
                if value.get("publish_port") is None
                else int(value["publish_port"])
            ),
            container_port=(
                None
                if value.get("container_port") is None
                else int(value["container_port"])
            ),
            expected_executable_sha256=(
                None
                if value.get("expected_executable_sha256") is None
                else str(value["expected_executable_sha256"])
            ),
            expected_executable_version_contains=(
                None
                if value.get("expected_executable_version_contains") is None
                else str(value["expected_executable_version_contains"])
            ),
            expected_executable_root=(
                None
                if value.get("expected_executable_root") is None
                else str(value["expected_executable_root"])
            ),
        )


@dataclass(frozen=True, slots=True)
class LaunchRecord:
    """Resolved identity of a launched process, container, or endpoint."""

    kind: LaunchKind
    command: tuple[str, ...]
    started_at: str
    pid: int | None = None
    process_create_time: float | None = None
    resolved_executable: str | None = None
    container_id: str | None = None
    container_name: str | None = None
    requested_image: str | None = None
    resolved_image: str | None = None
    executable_sha256: str | None = None
    executable_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "command": list(self.command),
            "started_at": self.started_at,
            "pid": self.pid,
            "process_create_time": self.process_create_time,
            "resolved_executable": self.resolved_executable,
            "executable_sha256": self.executable_sha256,
            "executable_version": self.executable_version,
            "container_id": self.container_id,
            "container_name": self.container_name,
            "requested_image": self.requested_image,
            "resolved_image": self.resolved_image,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LaunchRecord":
        kind = str(value["kind"])
        if kind not in {"process", "container", "external"}:
            raise DeploymentError(f"invalid launch kind: {kind!r}")
        raw_pid = value.get("pid")
        if isinstance(raw_pid, bool):
            raise DeploymentError("process PID must be a positive integer")
        pid = None if raw_pid is None else int(raw_pid)
        if pid is not None and pid <= 0:
            raise DeploymentError("process PID must be a positive integer")
        raw_create_time = value.get("process_create_time")
        if isinstance(raw_create_time, bool):
            raise DeploymentError("process creation time must be finite and positive")
        process_create_time = (
            None if raw_create_time is None else float(raw_create_time)
        )
        if process_create_time is not None and (
            not math.isfinite(process_create_time) or process_create_time <= 0
        ):
            raise DeploymentError("process creation time must be finite and positive")
        if kind == "process" and pid is None:
            raise DeploymentError("process launch record requires a PID")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            command=tuple(str(item) for item in value.get("command", ())),
            started_at=str(value["started_at"]),
            pid=pid,
            process_create_time=process_create_time,
            resolved_executable=(
                None
                if value.get("resolved_executable") is None
                else str(value["resolved_executable"])
            ),
            executable_sha256=(
                None
                if value.get("executable_sha256") is None
                else str(value["executable_sha256"])
            ),
            executable_version=(
                None
                if value.get("executable_version") is None
                else str(value["executable_version"])
            ),
            container_id=(
                None
                if value.get("container_id") is None
                else str(value["container_id"])
            ),
            container_name=(
                None
                if value.get("container_name") is None
                else str(value["container_name"])
            ),
            requested_image=(
                None
                if value.get("requested_image") is None
                else str(value["requested_image"])
            ),
            resolved_image=(
                None
                if value.get("resolved_image") is None
                else str(value["resolved_image"])
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeState:
    schema_version: int
    phase: StatePhase
    deployment: DeploymentSpec
    artifact_path: str | None
    plan: LaunchPlan
    launch: LaunchRecord
    activated_at: str
    error: str | None = None

    @property
    def deployment_id(self) -> str:
        return self.deployment.id

    @property
    def public_alias(self) -> str:
        return self.deployment.public_alias

    @property
    def backend(self) -> BackendKind:
        return self.deployment.backend

    @property
    def base_url(self) -> str:
        return self.plan.base_url

    @property
    def health_url(self) -> str:
        return self.plan.health_url

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase,
            "deployment_id": self.deployment.id,
            "artifact_id": self.deployment.artifact_id,
            "public_alias": self.deployment.public_alias,
            "backend": self.deployment.backend.value,
            "base_url": self.base_url,
            "health_url": self.health_url,
            "artifact_path": self.artifact_path,
            "deployment": self.deployment.model_dump(mode="json"),
            "plan": self.plan.to_dict(),
            "launch": self.launch.to_dict(),
            "activated_at": self.activated_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeState":
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("unsupported active-state schema version")
        phase = str(value["phase"])
        if phase not in {"starting", "ready", "stopping", "failed"}:
            raise ValueError(f"invalid runtime phase: {phase}")
        return cls(
            schema_version=1,
            phase=phase,  # type: ignore[arg-type]
            deployment=DeploymentSpec.model_validate(value["deployment"]),
            artifact_path=(
                None
                if value.get("artifact_path") is None
                else str(value["artifact_path"])
            ),
            plan=LaunchPlan.from_dict(value["plan"]),
            launch=LaunchRecord.from_dict(value["launch"]),
            activated_at=str(value["activated_at"]),
            error=None if value.get("error") is None else str(value["error"]),
        )


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    active: bool
    ready: bool
    running: bool
    healthy: bool | None
    state: RuntimeState | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "ready": self.ready,
            "running": self.running,
            "healthy": self.healthy,
            "state": None if self.state is None else self.state.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RuntimeLockVerification:
    """Evidence that a host deployment matches its reviewed runtime lock."""

    runtime_lock_id: str
    binary: str
    binary_sha256: str
    executable_version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "runtime_lock_id": self.runtime_lock_id,
            "binary": self.binary,
            "binary_sha256": self.binary_sha256,
            "executable_version": self.executable_version,
        }


class Launcher(Protocol):
    """Dependency-injection boundary shared by process/container launchers."""

    def start(self, plan: LaunchPlan) -> LaunchRecord: ...

    def is_running(self, record: LaunchRecord) -> bool: ...

    def stop(self, record: LaunchRecord, timeout_seconds: float = 10.0) -> None: ...


class ProcessLauncher:
    """Launch and control a host process without involving a command shell."""

    def start(self, plan: LaunchPlan) -> LaunchRecord:
        if plan.kind != "process" or not plan.command:
            raise DeploymentError("process launch requires a non-empty process plan")

        executable = _resolve_executable(
            plan.command[0],
            working_directory=plan.working_directory,
        )
        command = (executable, *plan.command[1:])
        if plan.expected_executable_sha256 is not None:
            # Locked plans carry their complete, recorded environment and do
            # not inherit behavior/code-loading knobs from the caller's shell.
            environment = dict(plan.environment)
            unsafe_environment = sorted(
                key
                for key in environment
                if key in _CODE_LOADING_ENVIRONMENT or key.startswith("LLAMA_ARG_")
            )
            if unsafe_environment:
                raise DeploymentError(
                    "locked process plan contains unsafe environment variables: "
                    f"{unsafe_environment}"
                )
        else:
            environment = os.environ.copy()
            environment.update(plan.environment)
        if plan.expected_executable_sha256 is not None:
            if plan.expected_executable_root is None:
                raise DeploymentError(
                    "a locked process plan must identify its trusted executable root"
                )
            try:
                relative_executable = Path(executable).absolute().relative_to(
                    Path(plan.expected_executable_root).absolute()
                )
            except ValueError as exc:
                raise DeploymentError(
                    f"locked executable {executable} is outside trusted root "
                    f"{plan.expected_executable_root}"
                ) from exc
            _, executable_descriptor, _ = open_regular_file_beneath(
                plan.expected_executable_root,
                relative_executable,
                os.O_RDONLY | os.O_NONBLOCK,
                purpose="locked executable",
                require_immutable=True,
            )
        else:
            executable_descriptor = _open_executable_handle(executable)
        try:
            executable_sha256 = _sha256_descriptor(
                executable_descriptor, executable
            )
            if (
                plan.expected_executable_sha256 is not None
                and executable_sha256 != plan.expected_executable_sha256
            ):
                raise DeploymentError(
                    f"executable SHA-256 mismatch for {executable}: expected "
                    f"{plan.expected_executable_sha256}, found {executable_sha256}"
                )
            executable_version = _probe_executable_version(
                executable,
                working_directory=plan.working_directory,
                environment=environment,
                executable_descriptor=executable_descriptor,
            )
            if (
                plan.expected_executable_version_contains is not None
                and (
                    executable_version is None
                    or plan.expected_executable_version_contains
                    not in executable_version
                )
            ):
                raise DeploymentError(
                    f"executable version for {executable} does not contain "
                    f"{plan.expected_executable_version_contains!r}"
                )

            log_file: Any = None
            try:
                output: Any = subprocess.DEVNULL
                if plan.log_path is not None:
                    log_path = Path(plan.log_path)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    descriptor = _open_nofollow_regular(
                        log_path,
                        os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NONBLOCK,
                        purpose="process log",
                    )
                    try:
                        log_file = os.fdopen(descriptor, "ab", buffering=0)
                    except BaseException:
                        os.close(descriptor)
                        raise
                    output = log_file
                execution_path = _descriptor_execution_path(executable_descriptor)
                process = subprocess.Popen(  # noqa: S603 - command is an argv tuple
                    command,
                    executable=execution_path,
                    pass_fds=(executable_descriptor,),
                    cwd=plan.working_directory,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                raise DeploymentError(
                    f"could not start {plan.deployment_id}: {exc}"
                ) from exc
            finally:
                if log_file is not None:
                    log_file.close()
        finally:
            os.close(executable_descriptor)

        try:
            create_time = psutil.Process(process.pid).create_time()
        except (psutil.Error, OSError) as exc:
            # Never persist a PID without its birth identity: PID reuse could
            # otherwise make a later stop target an unrelated process group.
            cleaned = _cleanup_new_process_group(process, timeout_seconds=2.0)
            if not cleaned:
                provisional = LaunchRecord(
                    kind="process",
                    command=command,
                    started_at=_utc_now(),
                    pid=process.pid,
                    process_create_time=None,
                    resolved_executable=executable,
                    executable_sha256=executable_sha256,
                    executable_version=executable_version,
                )
                raise UnsafeLaunchCleanupError(
                    f"could not capture process identity for {plan.deployment_id} "
                    "and cleanup of its process group could not be confirmed",
                    provisional,
                ) from exc
            raise DeploymentError(
                f"could not capture process identity for {plan.deployment_id}: {exc}"
            ) from exc
        return LaunchRecord(
            kind="process",
            command=command,
            started_at=_utc_now(),
            pid=process.pid,
            process_create_time=create_time,
            resolved_executable=executable,
            executable_sha256=executable_sha256,
            executable_version=executable_version,
        )

    def is_running(self, record: LaunchRecord) -> bool:
        process = _recorded_process(record)
        if process is not None and process.is_running() and not _is_zombie(process):
            return True
        return record.pid is not None and _process_group_exists(record.pid)

    def stop(self, record: LaunchRecord, timeout_seconds: float = 10.0) -> None:
        if record.pid is not None and record.process_create_time is None:
            raise DeploymentError(
                f"refusing to signal legacy process PID {record.pid} without "
                "a recorded creation time; manual cleanup is required"
            )
        process = _recorded_process(record)
        if process is None or not process.is_running():
            if record.pid is not None and _process_group_exists(record.pid):
                raise DeploymentError(
                    f"process group {record.pid} still exists but its recorded "
                    "leader identity cannot be verified; manual cleanup is required"
                )
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            raise DeploymentError(f"cannot stop process {process.pid}: {exc}") from exc

        deadline = time.monotonic() + max(0.01, timeout_seconds)
        while _process_group_exists(process.pid) and time.monotonic() < deadline:
            try:
                process.wait(timeout=0)
            except (psutil.TimeoutExpired, psutil.Error):
                pass
            time.sleep(0.05)
        try:
            process.wait(timeout=0)
        except (psutil.TimeoutExpired, psutil.Error):
            pass
        if _process_group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            kill_deadline = time.monotonic() + max(
                0.01, min(timeout_seconds, 5.0)
            )
            while (
                _process_group_exists(process.pid)
                and time.monotonic() < kill_deadline
            ):
                try:
                    process.wait(timeout=0)
                except (psutil.TimeoutExpired, psutil.Error):
                    pass
                time.sleep(0.05)
            if _process_group_exists(process.pid):
                raise DeploymentError(
                    f"process group {process.pid} did not stop after SIGKILL"
                )
            try:
                process.wait(timeout=0)
            except (psutil.TimeoutExpired, psutil.Error):
                pass


class UnsafeLaunchCleanupError(DeploymentError):
    """A detached runtime may exist, but cleanup could not be confirmed."""

    def __init__(self, message: str, record: LaunchRecord) -> None:
        super().__init__(message)
        self.record = record


class DockerLauncher:
    """Launch and control detached Docker containers via argv-only subprocesses."""

    def __init__(self, executable: str = "docker") -> None:
        self.executable = executable

    def start(self, plan: LaunchPlan) -> LaunchRecord:
        if plan.kind != "container" or not plan.image:
            raise DeploymentError("container launch requires an image")
        docker = _resolve_executable(self.executable)
        launch_token = secrets.token_hex(16)
        name = f"llm-lab-{plan.deployment_id}-{launch_token[:16]}"
        command: list[str] = [
            docker,
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--label",
            f"llm-lab.launch={launch_token}",
            "--gpus",
            "all",
        ]
        if plan.publish_port is not None and plan.container_port is not None:
            bind = plan.publish_host or "127.0.0.1"
            command.extend(
                ["--publish", f"{bind}:{plan.publish_port}:{plan.container_port}"]
            )
        for key, value in sorted(plan.environment.items()):
            command.extend(["--env", f"{key}={value}"])
        for mount in plan.mounts:
            option = f"{mount.source}:{mount.target}"
            if mount.read_only:
                option += ":ro"
            command.extend(["--volume", option])
        command.append(plan.image)
        command.extend(plan.command)

        container_id: str | None = None
        try:
            completed = _run_checked(command, "start Docker container")
            container_id = completed.stdout.strip()
            if not container_id:
                raise DeploymentError("Docker did not return a container id")
            inspect = _run_checked(
                [docker, "inspect", "--format", "{{.Image}}", container_id],
                "resolve Docker image",
            )
            resolved_image = inspect.stdout.strip()
            if not resolved_image:
                raise DeploymentError("Docker did not resolve the container image")
            return LaunchRecord(
                kind="container",
                command=tuple(command),
                started_at=_utc_now(),
                resolved_executable=docker,
                container_id=container_id,
                container_name=name,
                requested_image=plan.image,
                resolved_image=resolved_image,
            )
        except BaseException as launch_error:
            try:
                _remove_detached_container(docker, name, launch_token)
            except DeploymentError as cleanup_error:
                record = LaunchRecord(
                    kind="container",
                    command=tuple(command),
                    started_at=_utc_now(),
                    resolved_executable=docker,
                    container_id=container_id,
                    container_name=name,
                    requested_image=plan.image,
                )
                raise UnsafeLaunchCleanupError(
                    f"Docker launch failed and cleanup of {name!r} could not be "
                    f"confirmed: {cleanup_error}",
                    record,
                ) from launch_error
            raise

    def is_running(self, record: LaunchRecord) -> bool:
        target = record.container_id or record.container_name
        if not target:
            return False
        docker = record.resolved_executable or self.executable
        try:
            completed = subprocess.run(  # noqa: S603
                [docker, "inspect", "--format", "{{.State.Running}}", target],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise DeploymentError(
                f"could not inspect container {target}: {exc}"
            ) from exc
        if completed.returncode != 0:
            if "No such container" in completed.stderr:
                return False
            raise DeploymentError(
                f"could not inspect container {target}: "
                f"{(completed.stderr or completed.stdout).strip()}"
            )
        running = completed.stdout.strip()
        if running not in {"true", "false"}:
            raise DeploymentError(
                f"Docker returned invalid running state for {target}: {running!r}"
            )
        return running == "true"

    def stop(self, record: LaunchRecord, timeout_seconds: float = 10.0) -> None:
        target = record.container_id or record.container_name
        if not target:
            return
        docker = record.resolved_executable or _resolve_executable(self.executable)
        completed = subprocess.run(  # noqa: S603
            [
                docker,
                "stop",
                "--time",
                str(max(1, int(timeout_seconds))),
                target,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        missing = "No such container" in completed.stderr
        if completed.returncode != 0 and not missing:
            raise DeploymentError(
                f"could not stop container {target}: "
                f"{completed.stderr.strip()}"
            )


def _run_checked(command: Sequence[str], action: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else f": {exc}"
        raise DeploymentError(f"could not {action}{suffix}") from exc


def _remove_detached_container(docker: str, name: str, launch_token: str) -> None:
    """Remove only the detached container bearing this invocation's nonce."""

    try:
        ownership = subprocess.run(  # noqa: S603 - argv values are separate
            [
                docker,
                "inspect",
                "--format",
                '{{ index .Config.Labels "llm-lab.launch" }}',
                name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if ownership.returncode != 0:
            if "No such" in ownership.stderr:
                return
            raise DeploymentError(
                f"could not verify ownership of detached container {name}: "
                f"{(ownership.stderr or ownership.stdout).strip()}"
            )
        if ownership.stdout.strip() != launch_token:
            raise DeploymentError(
                f"refusing to remove container {name}: launch nonce does not match"
            )
        completed = subprocess.run(  # noqa: S603 - argv values are separate
            [docker, "rm", "--force", name],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise DeploymentError(
            f"could not remove detached container {name}: {exc}"
        ) from exc
    if completed.returncode != 0 and "No such container" not in completed.stderr:
        raise DeploymentError(
            f"could not remove detached container {name}: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )


def _open_executable_handle(executable: str) -> int:
    """Open the executable inode that will be hashed, probed, and launched."""

    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(executable, flags)
    except OSError as exc:
        raise DeploymentError(f"could not read executable {executable}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DeploymentError(f"executable is not a regular file: {executable}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _sha256_descriptor(descriptor: int, executable: str) -> str:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return digest.hexdigest()
    except OSError as exc:
        raise DeploymentError(f"could not hash executable {executable}: {exc}") from exc


def _descriptor_execution_path(descriptor: int) -> str:
    """Return a pathname that executes the already-open descriptor."""

    for root in (Path("/proc/self/fd"), Path("/dev/fd")):
        if root.is_dir():
            return str(root / str(descriptor))
    raise DeploymentError(
        "this platform cannot bind process launch to a verified executable descriptor"
    )


def _probe_executable_version(
    executable: str,
    *,
    working_directory: str | None,
    environment: Mapping[str, str],
    executable_descriptor: int | None = None,
) -> str | None:
    """Return bounded ``--version`` evidence without making launch depend on it."""

    try:
        kwargs: dict[str, Any] = {}
        if executable_descriptor is not None:
            kwargs.update(
                executable=_descriptor_execution_path(executable_descriptor),
                pass_fds=(executable_descriptor,),
            )
        completed = subprocess.run(  # noqa: S603 - executable is a resolved path
            [executable, "--version"],
            cwd=working_directory,
            env=dict(environment),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3.0,
            **kwargs,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    version = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return version[:4096] or None


def _resolve_executable(
    value: str,
    *,
    working_directory: str | Path | None = None,
) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.is_absolute() and working_directory is not None:
            candidate = Path(working_directory) / candidate
        if not candidate.exists():
            raise DeploymentError(f"executable does not exist: {candidate}")
        # Do not dereference executable symlinks: resolving ``.venv/bin/python``
        # to the base interpreter loses pyvenv.cfg and the installed package.
        return str(candidate.absolute())
    resolved = shutil.which(value)
    if resolved is None:
        raise DeploymentError(f"executable was not found on PATH: {value}")
    return str(Path(resolved).absolute())


def _describe_unsafe_path(purpose: str, path: Path, detail: str) -> DeploymentError:
    return DeploymentError(f"unsafe {purpose} path {path}: {detail}")


def _validate_open_regular_path(
    descriptor: int,
    path: Path,
    *,
    purpose: str,
) -> os.stat_result:
    """Validate an opened inode and ensure the name still designates that inode."""

    try:
        opened = os.fstat(descriptor)
        named = os.lstat(path)
    except OSError as exc:
        raise _describe_unsafe_path(purpose, path, str(exc)) from exc
    if not stat.S_ISREG(opened.st_mode):
        raise _describe_unsafe_path(purpose, path, "not a regular file")
    if opened.st_nlink != 1:
        raise _describe_unsafe_path(purpose, path, "hard-linked files are not allowed")
    if stat.S_ISLNK(named.st_mode):
        raise _describe_unsafe_path(purpose, path, "symbolic links are not allowed")
    if not stat.S_ISREG(named.st_mode):
        raise _describe_unsafe_path(purpose, path, "not a regular file")
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
        raise _describe_unsafe_path(purpose, path, "path changed while it was opened")
    return opened


def _open_nofollow_regular(
    path: Path,
    flags: int,
    *,
    purpose: str,
    mode: int = 0o600,
) -> int:
    """Open a regular file without following its final path component."""

    secure_flags = flags | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, secure_flags, mode)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise _describe_unsafe_path(purpose, path, str(exc)) from exc
    try:
        _validate_open_regular_path(descriptor, path, purpose=purpose)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_existing_regular_path(path: Path, *, purpose: str) -> bool:
    """Reject unsafe existing targets; return false when the name is absent."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise _describe_unsafe_path(purpose, path, str(exc)) from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise _describe_unsafe_path(purpose, path, "symbolic links are not allowed")
    if not stat.S_ISREG(metadata.st_mode):
        raise _describe_unsafe_path(purpose, path, "not a regular file")
    if metadata.st_nlink != 1:
        raise _describe_unsafe_path(purpose, path, "hard-linked files are not allowed")
    return True


def _recorded_process(record: LaunchRecord) -> psutil.Process | None:
    # Legacy records without a process birth time are readable for diagnosis,
    # but never trusted for signalling: their PID may have been reused.
    if record.pid is None or record.process_create_time is None:
        return None
    try:
        process = psutil.Process(record.pid)
        if abs(process.create_time() - record.process_create_time) > 0.01:
            return None
        return process
    except (psutil.Error, OSError):
        return None


def _process_group_exists(process_group: int) -> bool:
    """Return whether a positive process group has a live member.

    ``killpg(pgid, 0)`` also succeeds when the group's only remaining members
    are zombies. Zombies cannot run or retain the GPU, so treating that result
    alone as liveness makes stop wait through SIGKILL and fail indefinitely
    until their parent reaps them. Confirm the group contains at least one
    non-zombie process while retaining the signal-zero permission/existence
    check as a fast path.
    """

    if process_group <= 0:
        return False
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass

    saw_group_member = False
    terminal_statuses = {
        psutil.STATUS_ZOMBIE,
        getattr(psutil, "STATUS_DEAD", "dead"),
    }
    try:
        processes = psutil.process_iter(attrs=("pid", "status"))
        for process in processes:
            try:
                pid = int(process.info["pid"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                member_group = os.getpgid(pid)
            except ProcessLookupError:
                continue
            except (PermissionError, OSError):
                # Membership is unknown, so this may be an unrelated protected
                # process. If no target member can be inspected, the final
                # kernel re-check below still fails closed.
                continue
            if member_group != process_group:
                continue
            try:
                status = process.info.get("status") or process.status()
            except psutil.NoSuchProcess:
                continue
            except (psutil.Error, OSError):
                # Membership was confirmed. An unreadable state must remain
                # conservatively live rather than hiding a target worker.
                return True
            saw_group_member = True
            if status not in terminal_statuses:
                return True
    except (psutil.Error, OSError):
        return True

    if saw_group_member:
        return False
    # The group changed while it was enumerated, or this process cannot see its
    # members. Re-check the kernel and fail closed if it still reports a group.
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_new_process_group(
    process: subprocess.Popen[Any],
    *,
    timeout_seconds: float,
) -> bool:
    """Stop a just-created session and confirm that no worker remains."""

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        try:
            process.wait(timeout=0)
        except (subprocess.SubprocessError, OSError):
            pass
        return not _process_group_exists(process_group)
    except (PermissionError, OSError):
        return False

    deadline = time.monotonic() + max(0.01, timeout_seconds)
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        try:
            process.wait(timeout=0)
        except (subprocess.TimeoutExpired, OSError):
            pass
        time.sleep(0.05)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except (PermissionError, OSError):
            return False
        deadline = time.monotonic() + max(0.01, timeout_seconds)
        while _process_group_exists(process_group) and time.monotonic() < deadline:
            try:
                process.wait(timeout=0)
            except (subprocess.TimeoutExpired, OSError):
                pass
            time.sleep(0.05)
    try:
        process.wait(timeout=0)
    except (subprocess.TimeoutExpired, OSError):
        pass
    return not _process_group_exists(process_group)


def _is_zombie(process: psutil.Process) -> bool:
    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.Error:
        return True


class ExclusiveGpuLock:
    """Advisory inter-process lock guarding the single-GPU active state."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 30.0,
        poll_seconds: float = 0.05,
    ) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._file: Any = None

    def __enter__(self) -> "ExclusiveGpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = _open_nofollow_regular(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_NONBLOCK,
            purpose="GPU lock",
        )
        try:
            self._file = os.fdopen(descriptor, "r+", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise
        try:
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    fcntl.flock(
                        self._file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise DeploymentError(
                            f"timed out acquiring GPU lock: {self.path}"
                        ) from exc
                    time.sleep(self.poll_seconds)
            # Recheck immediately before mutation in case the directory entry was
            # replaced while this process waited on another lock holder.
            _validate_open_regular_path(
                self._file.fileno(),
                self.path,
                purpose="GPU lock",
            )
            self._file.seek(0)
            self._file.truncate()
            self._file.write(f"pid={os.getpid()} acquired={_utc_now()}\n")
            self._file.flush()
            os.fsync(self._file.fileno())
            return self
        except BaseException:
            self._file.close()
            self._file = None
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._file is None:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None


def _open_state_directory(paths: LabPaths) -> int:
    try:
        _, descriptor = open_safe_directory(
            paths.data_root / "state",
            purpose="runtime state",
            create=False,
        )
    except StoragePolicyError as exc:
        raise DeploymentError(str(exc)) from exc
    return descriptor


def _validate_state_entry(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DeploymentError(f"could not inspect active state: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise DeploymentError(
            "unsafe active state path: expected one private regular file"
        )
    return metadata


def _atomic_write_json(paths: LabPaths, value: Mapping[str, Any]) -> None:
    directory_fd = _open_state_directory(paths)
    name = paths.active_state_path.name
    temporary_name = f".{name}.{secrets.token_hex(12)}.tmp"
    temporary_created = False
    try:
        _validate_state_entry(directory_fd, name)
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            temporary_created = True
        except OSError as exc:
            raise DeploymentError(f"could not stage active state: {exc}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
            metadata = os.fstat(file.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise DeploymentError("unsafe active state temporary file")
        _validate_state_entry(directory_fd, name)
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_created = False
        os.fsync(directory_fd)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def write_active_state(paths: LabPaths, state: RuntimeState) -> None:
    """Atomically publish an active runtime state."""

    _atomic_write_json(paths, state.to_dict())


def read_active_state(paths: LabPaths) -> RuntimeState | None:
    """Read and validate active state, returning ``None`` when inactive."""

    directory_fd = _open_state_directory(paths)
    try:
        name = paths.active_state_path.name
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | os.O_NONBLOCK
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise DeploymentError(
                f"unsafe active state path {paths.active_state_path}: {exc}"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            named = _validate_state_entry(directory_fd, name)
            assert named is not None
            if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
                raise DeploymentError("active state changed while it was opened")
            with os.fdopen(descriptor, "r", encoding="utf-8") as file:
                descriptor = -1
                raw = file.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except (OSError, UnicodeError) as exc:
        raise DeploymentError(
            f"could not read active state {paths.active_state_path}: {exc}"
        ) from exc
    finally:
        os.close(directory_fd)
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("top-level value must be an object")
        return RuntimeState.from_dict(value)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DeploymentError(
            f"active state is invalid: {paths.active_state_path}: {exc}"
        ) from exc


def clear_active_state(paths: LabPaths) -> None:
    directory_fd = _open_state_directory(paths)
    try:
        name = paths.active_state_path.name
        if _validate_state_entry(directory_fd, name) is None:
            return
        try:
            os.unlink(name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DeploymentError(
                f"could not clear active state {paths.active_state_path}: {exc}"
            ) from exc
    finally:
        os.close(directory_fd)


def wait_for_readiness(
    url: str,
    timeout_seconds: float,
    *,
    interval_seconds: float = 0.1,
    request_timeout_seconds: float = 2.0,
    client: httpx.Client | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll ``url`` until it returns a 2xx response or the deadline expires."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    owned_client = client is None
    http_client = client or httpx.Client(follow_redirects=True)
    deadline = monotonic() + timeout_seconds
    last_detail = "no response"
    try:
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            try:
                response = http_client.get(
                    url,
                    timeout=max(0.01, min(request_timeout_seconds, remaining)),
                )
                if 200 <= response.status_code < 300:
                    return
                last_detail = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_detail = str(exc)
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            sleep(min(interval_seconds, remaining))
    finally:
        if owned_client:
            http_client.close()
    raise DeploymentError(
        f"backend did not become ready at {url} within "
        f"{timeout_seconds:g}s ({last_detail})"
    )


def _image_reference(spec: DeploymentSpec) -> str | None:
    if spec.image is None:
        return None
    reference = spec.image.reference
    if spec.image.digest and "@sha256:" not in reference:
        return f"{reference}@{spec.image.digest}"
    return reference


def _expand_deployment_value(value: str, paths: LabPaths) -> str:
    """Expand the two stable lab roots without mutating process environment."""

    replacements = {
        "${LLM_LAB_DATA}": str(paths.data_root),
        "$LLM_LAB_DATA": str(paths.data_root),
        "${LLM_LAB_REPO}": str(paths.repo_root),
        "$LLM_LAB_REPO": str(paths.repo_root),
    }
    expanded = value
    for marker, replacement in replacements.items():
        expanded = expanded.replace(marker, replacement)
    return os.path.expandvars(expanded)


def _rewrite_model_arg(value: str, artifact_root: Path) -> str:
    """Translate the container's /models convention for a host process."""

    if value == "/models":
        return str(artifact_root)
    prefix = "/models/"
    if value.startswith(prefix):
        return str(artifact_root / value.removeprefix(prefix))
    return value


def _artifact_root(path: Path) -> Path:
    absolute = path.absolute()
    return absolute if absolute.is_dir() else absolute.parent


def _container_model_path(selected: Path, artifact_root: Path) -> Path:
    """Map an artifact or selected file into the conventional /models mount."""

    # Preserve a view symlink lexically.  Resolving it would move the selected
    # path into the CAS blob directory and lose its location under /models.
    selected = selected.absolute()
    artifact_root = artifact_root.absolute()
    if selected == artifact_root:
        return Path("/models")
    try:
        return Path("/models") / selected.relative_to(artifact_root)
    except ValueError as exc:
        raise DeploymentError(
            f"selected model {selected} is outside artifact root {artifact_root}"
        ) from exc


def _resolve_gguf(artifact_path: Path) -> Path:
    if artifact_path.is_file():
        if artifact_path.suffix.lower() != ".gguf":
            raise DeploymentError(f"llama.cpp artifact is not GGUF: {artifact_path}")
        return artifact_path.absolute()
    if not artifact_path.is_dir():
        raise DeploymentError(f"artifact path does not exist: {artifact_path}")
    candidates = sorted(
        path
        for path in artifact_path.rglob("*.gguf")
        if not path.name.lower().startswith(("mmproj", "draft"))
    )
    first_shards = [path for path in candidates if "-00001-of-" in path.name]
    if len(first_shards) == 1:
        return first_shards[0].absolute()
    if len(candidates) == 1:
        return candidates[0].absolute()
    if not candidates:
        raise DeploymentError(f"no GGUF weights found under {artifact_path}")
    raise DeploymentError(
        f"multiple GGUF weight candidates under {artifact_path}; pass a file path"
    )


_PROTECTED_EXTRA_OPTIONS: dict[BackendKind, frozenset[str]] = {
    BackendKind.MOCK: frozenset({"--host", "--port", "--model"}),
    BackendKind.LLAMA_CPP: frozenset(
        {
            "-a",
            "-c",
            "-ctk",
            "-ctv",
            "-dr",
            "-fa",
            "-hf",
            "-hff",
            "-hfr",
            "-m",
            "-mm",
            "-mu",
            "-ngl",
            "-np",
            "-rea",
            "--alias",
            "--cache-type-k",
            "--cache-type-v",
            "--ctx-size",
            "--docker-repo",
            "--flash-attn",
            "--gpu-layers",
            "--hf-file",
            "--hf-repo",
            "--host",
            "--mmproj",
            "--model",
            "--model-url",
            "--models-dir",
            "--models-preset",
            "--n-gpu-layers",
            "--parallel",
            "--port",
            "--reasoning",
        }
    ),
    BackendKind.VLLM: frozenset(
        {
            "--host",
            "--max-model-len",
            "--max-num-seqs",
            "--model",
            "--port",
            "--served-model-name",
        }
    ),
    BackendKind.SGLANG: frozenset(
        {
            "--context-length",
            "--host",
            "--max-running-requests",
            "--model",
            "--model-path",
            "--port",
            "--served-model-name",
        }
    ),
    BackendKind.TENSORRT_LLM: frozenset(
        {"--host", "--max_batch_size", "--max_seq_len", "--model", "--port"}
    ),
    BackendKind.EXTERNAL: frozenset(),
}
_ALLOWED_EXTRA_ARGUMENTS: dict[BackendKind, frozenset[str]] = {
    BackendKind.LLAMA_CPP: frozenset(
        {"--jinja", "--metrics", "--mlock", "--no-mmap", "--no-webui", "--slots"}
    ),
    BackendKind.MOCK: frozenset(),
    BackendKind.VLLM: frozenset(),
    BackendKind.SGLANG: frozenset(),
    BackendKind.TENSORRT_LLM: frozenset(),
    BackendKind.EXTERNAL: frozenset(),
}


def _validate_extra_args(deployment: DeploymentSpec, arguments: list[str]) -> None:
    protected = _PROTECTED_EXTRA_OPTIONS[deployment.backend]
    allowed = _ALLOWED_EXTRA_ARGUMENTS[deployment.backend]
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if argument == "--" or option in protected:
            raise DeploymentError(
                f"deployment {deployment.id!r} extra_args may not override "
                f"protected {deployment.backend.value} option {option!r}"
            )
        if argument not in allowed:
            raise DeploymentError(
                f"deployment {deployment.id!r} extra argument {argument!r} is not "
                f"in the reviewed {deployment.backend.value} allowlist; add a typed "
                "deployment field before using it"
            )


def build_backend_command(
    deployment: DeploymentSpec,
    artifact_path: str | Path | None = None,
    *,
    paths: LabPaths | None = None,
) -> LaunchPlan:
    """Build the exact launch plan for a deployment.

    An image selects Docker.  With an image, ``executable`` optionally overrides the
    command inside the container.  Without an image it is the host executable.
    """

    resolved_paths = paths or LabPaths.discover()
    base = _base_url(deployment)
    ready_url = health_url(base, deployment.health_path)
    log_path = resolved_paths.data_root / "logs" / f"{deployment.id}.log"
    explicit_environment = dict(deployment.environment)
    unsafe_code_loading = sorted(
        key for key in explicit_environment if key in _CODE_LOADING_ENVIRONMENT
    )
    if unsafe_code_loading:
        raise DeploymentError(
            f"deployment {deployment.id!r} may not set code-loading environment "
            f"variables: {unsafe_code_loading}"
        )
    environment = (
        {**_LOCKED_ENVIRONMENT_BASE, **explicit_environment}
        if deployment.runtime_lock_id is not None
        else explicit_environment
    )
    if deployment.backend == BackendKind.LLAMA_CPP:
        unsafe_environment = sorted(
            key for key in explicit_environment if key.startswith("LLAMA_ARG_")
        )
        if unsafe_environment:
            raise DeploymentError(
                f"deployment {deployment.id!r} may not set llama.cpp argument "
                f"environment variables: {unsafe_environment}"
            )

    if deployment.backend == BackendKind.EXTERNAL:
        return LaunchPlan(
            kind="external",
            deployment_id=deployment.id,
            command=(),
            environment=environment,
            base_url=base,
            health_url=ready_url,
        )

    image = _image_reference(deployment)
    configured_executable = (
        None
        if deployment.executable is None
        else _expand_deployment_value(deployment.executable, resolved_paths)
    )
    kind: LaunchKind = "container" if image else "process"
    bind_host = "0.0.0.0" if kind == "container" else deployment.host
    command: list[str]
    model_path: Path | None = None
    mount_root: Path | None = None

    if deployment.backend == BackendKind.MOCK:
        if image:
            command = [configured_executable or "python", "-m", "llm_lab.mock_backend"]
        else:
            mock_executable = (
                sys.executable
                if configured_executable in {None, "python", "python3"}
                else configured_executable
            )
            command = [mock_executable, "-m", "llm_lab.mock_backend"]
        command.extend(
            [
                "--host",
                bind_host,
                "--port",
                str(deployment.port),
                "--model",
                deployment.public_alias,
            ]
        )
    else:
        if artifact_path is None:
            raise DeploymentError(
                f"{deployment.backend.value} deployment requires an artifact path"
            )
        raw_model_path = Path(artifact_path).expanduser()
        if not raw_model_path.exists():
            raise DeploymentError(f"artifact path does not exist: {raw_model_path}")
        mount_root = _artifact_root(raw_model_path)
        if deployment.backend == BackendKind.LLAMA_CPP:
            model_path = _resolve_gguf(raw_model_path)
            model_argument = (
                _container_model_path(model_path, mount_root)
                if kind == "container"
                else model_path
            )
            prefix = [configured_executable] if configured_executable else []
            if kind == "process" and not prefix:
                raise DeploymentError("llama.cpp host deployment requires executable")
            command = [item for item in prefix if item]
            command.extend(
                [
                    "--model",
                    str(model_argument),
                    "--alias",
                    deployment.public_alias,
                    "--host",
                    bind_host,
                    "--port",
                    str(deployment.port),
                    "--ctx-size",
                    str(deployment.context_size),
                    "--parallel",
                    str(deployment.parallel),
                    "--gpu-layers",
                    str(deployment.gpu_layers),
                    "--flash-attn",
                    "on" if deployment.flash_attention else "off",
                    "--cache-type-k",
                    deployment.kv_cache_type_k,
                    "--cache-type-v",
                    deployment.kv_cache_type_v,
                    "--reasoning",
                    deployment.reasoning_mode,
                ]
            )
            if deployment.mmproj is not None:
                projector = _expand_deployment_value(deployment.mmproj, resolved_paths)
                if kind == "process":
                    projector = _rewrite_model_arg(projector, mount_root)
                    projector_path = Path(projector).absolute()
                    try:
                        projector_path.relative_to(mount_root.absolute())
                    except ValueError as exc:
                        raise DeploymentError(
                            f"mmproj path escapes artifact root: {projector_path}"
                        ) from exc
                    if not projector_path.is_file():
                        raise DeploymentError(f"mmproj does not exist: {projector_path}")
                command.extend(["--mmproj", projector])
        elif deployment.backend == BackendKind.VLLM:
            model_path = raw_model_path.resolve()
            model_argument = (
                _container_model_path(model_path, mount_root)
                if kind == "container"
                else model_path
            )
            prefix = [configured_executable] if configured_executable else ["serve"]
            if kind == "process" and not configured_executable:
                raise DeploymentError("vLLM host deployment requires executable")
            command = [item for item in prefix if item]
            if configured_executable:
                command.append("serve")
            command.extend(
                [
                    str(model_argument),
                    "--host",
                    bind_host,
                    "--port",
                    str(deployment.port),
                    "--served-model-name",
                    deployment.public_alias,
                    "--max-model-len",
                    str(deployment.context_size),
                    "--max-num-seqs",
                    str(deployment.parallel),
                ]
            )
        elif deployment.backend == BackendKind.SGLANG:
            model_path = raw_model_path.resolve()
            model_argument = (
                _container_model_path(model_path, mount_root)
                if kind == "container"
                else model_path
            )
            if configured_executable:
                command = [configured_executable, "serve"]
            elif kind == "container":
                command = ["python3", "-m", "sglang.launch_server"]
            else:
                raise DeploymentError("SGLang host deployment requires executable")
            command.extend(
                [
                    "--model-path",
                    str(model_argument),
                    "--host",
                    bind_host,
                    "--port",
                    str(deployment.port),
                    "--context-length",
                    str(deployment.context_size),
                    "--max-running-requests",
                    str(deployment.parallel),
                ]
            )
        elif deployment.backend == BackendKind.TENSORRT_LLM:
            model_path = raw_model_path.resolve()
            model_argument = (
                _container_model_path(model_path, mount_root)
                if kind == "container"
                else model_path
            )
            command = [configured_executable or "trtllm-serve"]
            command.extend(
                [
                    str(model_argument),
                    "--host",
                    bind_host,
                    "--port",
                    str(deployment.port),
                    "--max_seq_len",
                    str(deployment.context_size),
                    "--max_batch_size",
                    str(deployment.parallel),
                ]
            )
        else:  # pragma: no cover - exhaustive for future enum additions
            raise DeploymentError(f"unsupported backend: {deployment.backend.value}")

    extra_args = [
        _expand_deployment_value(argument, resolved_paths)
        for argument in deployment.extra_args
    ]
    _validate_extra_args(deployment, extra_args)
    if kind == "process" and mount_root is not None:
        extra_args = [
            _rewrite_model_arg(argument, mount_root) for argument in extra_args
        ]
    command.extend(extra_args)
    mounts: tuple[VolumeMount, ...] = ()
    if kind == "container" and model_path is not None and mount_root is not None:
        selected_mounts = [
            VolumeMount(
                source=str(mount_root),
                target="/models",
                read_only=True,
            )
        ]
        try:
            mount_root.relative_to(resolved_paths.view_root.absolute())
        except ValueError:
            pass
        else:
            # Views use ../../blobs/... relative symlinks.  Under the /models
            # bind mount those resolve to /blobs/..., so expose the immutable
            # CAS at that matching read-only location.
            selected_mounts.append(
                VolumeMount(
                    source=str(resolved_paths.blob_root.absolute()),
                    target="/blobs",
                    read_only=True,
                )
            )
        mounts = tuple(selected_mounts)
    return LaunchPlan(
        kind=kind,
        deployment_id=deployment.id,
        command=tuple(command),
        environment=environment,
        base_url=base,
        health_url=ready_url,
        log_path=str(log_path),
        working_directory=str(resolved_paths.repo_root),
        image=image,
        mounts=mounts,
        publish_host=deployment.host if kind == "container" else None,
        publish_port=deployment.port if kind == "container" else None,
        container_port=deployment.port if kind == "container" else None,
    )


def verify_runtime_lock(
    paths: LabPaths,
    deployment: DeploymentSpec,
    runtime_lock: RuntimeLockSpec,
    artifact_path: str | Path | None,
) -> RuntimeLockVerification:
    """Verify the exact host binary before a deployment can replace the GPU owner.

    The catalog lock identifies both the reviewed source/build recipe and the
    concrete binary produced for this installation.  Container deployments use
    their immutable image digest instead and are therefore outside this check.
    """

    if deployment.runtime_lock_id != runtime_lock.id:
        raise DeploymentError(
            f"deployment {deployment.id!r} does not reference runtime lock "
            f"{runtime_lock.id!r}"
        )
    plan = build_backend_command(deployment, artifact_path, paths=paths)
    if plan.kind != "process" or not plan.command:
        raise DeploymentError(
            f"runtime lock {runtime_lock.id!r} requires a host-process deployment"
        )

    expected = (paths.data_root / runtime_lock.binary).absolute()
    try:
        expected.relative_to(paths.data_root.absolute())
    except ValueError as exc:  # defensive; the schema already rejects ``..``.
        raise DeploymentError(
            f"runtime lock binary escapes the data root: {runtime_lock.binary}"
        ) from exc
    actual = Path(
        _resolve_executable(
            plan.command[0], working_directory=plan.working_directory
        )
    ).absolute()
    if actual != expected:
        raise DeploymentError(
            f"deployment executable {actual} does not match locked binary {expected}"
        )

    _, descriptor, _ = open_regular_file_beneath(
        paths.data_root,
        runtime_lock.binary,
        os.O_RDONLY | os.O_NONBLOCK,
        purpose="locked runtime binary",
        require_immutable=True,
    )
    try:
        binary_sha256 = _sha256_descriptor(descriptor, str(actual))
        if binary_sha256 != runtime_lock.binary_sha256:
            raise DeploymentError(
                f"locked runtime binary SHA-256 mismatch for {actual}: expected "
                f"{runtime_lock.binary_sha256}, found {binary_sha256}"
            )

        environment = dict(plan.environment)
        unsafe_environment = sorted(
            key
            for key in environment
            if key in _CODE_LOADING_ENVIRONMENT or key.startswith("LLAMA_ARG_")
        )
        if unsafe_environment:
            raise DeploymentError(
                "locked process plan contains unsafe environment variables: "
                f"{unsafe_environment}"
            )
        version = _probe_executable_version(
            str(actual),
            working_directory=plan.working_directory,
            environment=environment,
            executable_descriptor=descriptor,
        )
    finally:
        os.close(descriptor)
    if version is None or runtime_lock.version_contains not in version:
        raise DeploymentError(
            f"locked runtime version evidence for {actual} does not contain "
            f"{runtime_lock.version_contains!r}"
        )
    return RuntimeLockVerification(
        runtime_lock_id=runtime_lock.id,
        binary=str(actual),
        binary_sha256=binary_sha256,
        executable_version=version,
    )


def _assert_launch_matches_runtime_lock(
    launch: LaunchRecord, verification: RuntimeLockVerification
) -> None:
    if launch.kind != "process":
        raise DeploymentError("a host runtime lock produced a non-process launch")
    if launch.resolved_executable != verification.binary:
        raise DeploymentError(
            "launched executable path changed after runtime-lock verification"
        )
    if launch.executable_sha256 != verification.binary_sha256:
        raise DeploymentError(
            "launched executable bytes changed after runtime-lock verification"
        )
    if (
        launch.executable_version is None
        or verification.executable_version != launch.executable_version
    ):
        raise DeploymentError(
            "launched executable version changed after runtime-lock verification"
        )


def _assert_rollback_identity(previous: LaunchRecord, restored: LaunchRecord) -> None:
    """Do not call changed executable/container bytes a successful rollback."""

    if previous.kind == "container":
        if (
            restored.kind != "container"
            or restored.resolved_image != previous.resolved_image
            or restored.requested_image != previous.resolved_image
        ):
            raise DeploymentError(
                "rollback container identity differs from the previously active runtime"
            )
        return
    if previous.kind != "process" or previous.executable_sha256 is None:
        return
    if (
        restored.kind != "process"
        or restored.resolved_executable != previous.resolved_executable
        or restored.executable_sha256 != previous.executable_sha256
    ):
        raise DeploymentError(
            "rollback executable identity differs from the previously active runtime"
        )


HealthChecker = Callable[[str, float], None]


class RuntimeManager:
    """Activate exactly one deployment, with health-gated rollback semantics."""

    def __init__(
        self,
        paths: LabPaths | None = None,
        *,
        process_launcher: Launcher | None = None,
        container_launcher: Launcher | None = None,
        health_checker: HealthChecker | None = None,
        gpu_lock_timeout_seconds: float = 30.0,
        stop_timeout_seconds: float = 10.0,
        status_timeout_seconds: float = 1.0,
    ) -> None:
        self.paths = paths or LabPaths.discover()
        self.process_launcher = process_launcher or ProcessLauncher()
        self.container_launcher = container_launcher or DockerLauncher()
        self.health_checker = health_checker or wait_for_readiness
        self.gpu_lock_timeout_seconds = gpu_lock_timeout_seconds
        self.stop_timeout_seconds = stop_timeout_seconds
        self.status_timeout_seconds = status_timeout_seconds

    def activate(
        self,
        deployment: DeploymentSpec,
        artifact_path: str | Path | None = None,
        *,
        runtime_lock: RuntimeLockSpec | None = None,
    ) -> RuntimeState:
        """Activate a deployment and restore the prior one if readiness fails."""

        self.paths.initialize()
        artifact = (
            None if artifact_path is None else str(Path(artifact_path).expanduser().resolve())
        )
        plan = build_backend_command(deployment, artifact, paths=self.paths)
        if deployment.runtime_lock_id is not None and runtime_lock is None:
            raise DeploymentError(
                f"deployment {deployment.id!r} requires runtime lock "
                f"{deployment.runtime_lock_id!r}"
            )
        lock_verification = (
            None
            if runtime_lock is None
            else verify_runtime_lock(
                self.paths, deployment, runtime_lock, artifact
            )
        )
        if lock_verification is not None:
            plan = replace(
                plan,
                expected_executable_sha256=lock_verification.binary_sha256,
                expected_executable_version_contains=runtime_lock.version_contains,
                expected_executable_root=str(self.paths.data_root),
            )
        with self._lock():
            previous = read_active_state(self.paths)
            if previous is not None:
                self._verify_saved_artifact(previous)
                self._stop_state(previous)
                clear_active_state(self.paths)

            candidate: RuntimeState | None = None
            try:
                candidate = self._start_state(deployment, artifact, plan)
                if lock_verification is not None:
                    _assert_launch_matches_runtime_lock(
                        candidate.launch, lock_verification
                    )
                write_active_state(self.paths, candidate)
                self._confirm_ready(
                    candidate,
                    timeout_seconds=deployment.startup_timeout_seconds,
                )
                ready = replace(candidate, phase="ready", error=None)
                write_active_state(self.paths, ready)
                return ready
            except Exception as activation_error:
                if isinstance(activation_error, UnsafeLaunchCleanupError):
                    failed = RuntimeState(
                        schema_version=1,
                        phase="failed",
                        deployment=deployment,
                        artifact_path=artifact,
                        plan=plan,
                        launch=activation_error.record,
                        activated_at=_utc_now(),
                        error=str(activation_error),
                    )
                    write_active_state(self.paths, failed)
                    raise DeploymentError(
                        f"activation of {deployment.id} left a detached runtime "
                        "whose cleanup could not be confirmed; rollback was not "
                        "attempted"
                    ) from activation_error
                unsafe_candidate_stop: Exception | None = None
                if candidate is not None:
                    try:
                        self._stop_state(candidate)
                    except Exception as exc:
                        try:
                            still_running = self._is_running(candidate)
                        except Exception:
                            still_running = True
                        if still_running:
                            unsafe_candidate_stop = exc
                if unsafe_candidate_stop is not None and candidate is not None:
                    failed = replace(
                        candidate,
                        phase="failed",
                        error=(
                            f"activation failed: {activation_error}; cleanup failed: "
                            f"{unsafe_candidate_stop}"
                        ),
                    )
                    write_active_state(self.paths, failed)
                    raise DeploymentError(
                        f"activation of {deployment.id} failed and candidate may "
                        f"still own the GPU: {unsafe_candidate_stop}; rollback was "
                        "not attempted"
                    ) from activation_error
                clear_active_state(self.paths)
                rollback_error: Exception | None = None
                restored: RuntimeState | None = None
                if previous is not None:
                    try:
                        self._verify_saved_artifact(previous)
                        rollback_plan = self._resolved_rollback_plan(previous)
                        restored = self._start_state(
                            previous.deployment,
                            previous.artifact_path,
                            rollback_plan,
                        )
                        _assert_rollback_identity(previous.launch, restored.launch)
                        write_active_state(self.paths, restored)
                        self._confirm_ready(
                            restored,
                            timeout_seconds=(
                                previous.deployment.startup_timeout_seconds
                            ),
                        )
                        restored = replace(restored, phase="ready", error=None)
                        write_active_state(self.paths, restored)
                    except Exception as exc:
                        rollback_error = exc
                        unsafe_restored_stop: Exception | None = None
                        if restored is not None:
                            try:
                                self._stop_state(restored)
                            except Exception as stop_exc:
                                try:
                                    still_running = self._is_running(restored)
                                except Exception:
                                    still_running = True
                                if still_running:
                                    unsafe_restored_stop = stop_exc
                        if unsafe_restored_stop is not None and restored is not None:
                            rollback_error = DeploymentError(
                                f"{exc}; restored process cleanup failed and may "
                                f"still own the GPU: {unsafe_restored_stop}"
                            )
                            write_active_state(
                                self.paths,
                                replace(
                                    restored,
                                    phase="failed",
                                    error=str(rollback_error),
                                ),
                            )
                        else:
                            clear_active_state(self.paths)

                message = f"activation of {deployment.id} failed: {activation_error}"
                if previous is not None and rollback_error is None:
                    message += f"; restored {previous.deployment.id}"
                elif rollback_error is not None:
                    message += f"; rollback also failed: {rollback_error}"
                raise DeploymentError(message) from activation_error

    def status(self, *, check_health: bool = True) -> RuntimeStatus:
        self.paths.initialize()
        with self._lock():
            return self._status_unlocked(check_health=check_health)

    @contextmanager
    def benchmark_lease(
        self,
        *,
        check_health: bool = True,
    ) -> Iterator[RuntimeStatus]:
        """Hold the GPU lifecycle lock for one uninterrupted benchmark run."""

        self.paths.initialize()
        with self._lock():
            yield self._status_unlocked(check_health=check_health)

    def _status_unlocked(self, *, check_health: bool) -> RuntimeStatus:
        state = read_active_state(self.paths)
        if state is None:
            return RuntimeStatus(
                active=False,
                ready=False,
                running=False,
                healthy=None,
                state=None,
            )
        return self.inspect_state(state, check_health=check_health)

    def inspect_state(
        self,
        state: RuntimeState,
        *,
        check_health: bool = True,
    ) -> RuntimeStatus:
        """Inspect one state snapshot without acquiring the lifecycle lock."""

        running = self._is_running(state)
        healthy: bool | None = None
        if check_health and running:
            try:
                self.health_checker(state.health_url, self.status_timeout_seconds)
                healthy = True
            except Exception:
                healthy = False
        ready = state.phase == "ready" and running and healthy is not False
        return RuntimeStatus(
            active=True,
            ready=ready,
            running=running,
            healthy=healthy,
            state=state,
        )

    def stop(self) -> RuntimeState | None:
        self.paths.initialize()
        with self._lock():
            state = read_active_state(self.paths)
            if state is None:
                return None
            stopping = replace(state, phase="stopping")
            write_active_state(self.paths, stopping)
            try:
                self._stop_state(stopping)
            except Exception as exc:
                failed = replace(stopping, phase="failed", error=str(exc))
                write_active_state(self.paths, failed)
                raise DeploymentError(
                    f"could not stop {state.deployment.id}: {exc}"
                ) from exc
            clear_active_state(self.paths)
            return state

    def _lock(self) -> ExclusiveGpuLock:
        return ExclusiveGpuLock(
            self.paths.gpu_lock_path,
            timeout_seconds=self.gpu_lock_timeout_seconds,
        )

    def _start_state(
        self,
        deployment: DeploymentSpec,
        artifact_path: str | None,
        plan: LaunchPlan,
    ) -> RuntimeState:
        if plan.kind == "external":
            launch = LaunchRecord(
                kind="external",
                command=(),
                started_at=_utc_now(),
            )
        else:
            launcher = self._launcher(plan.kind)
            launch = launcher.start(plan)
        return RuntimeState(
            schema_version=1,
            phase="starting",
            deployment=deployment,
            artifact_path=artifact_path,
            plan=plan,
            launch=launch,
            activated_at=_utc_now(),
        )

    def _launcher(self, kind: LaunchKind) -> Launcher:
        if kind == "process":
            return self.process_launcher
        if kind == "container":
            return self.container_launcher
        raise DeploymentError(f"external endpoints do not have a launcher: {kind}")

    def _is_running(self, state: RuntimeState) -> bool:
        if state.launch.kind == "external":
            return True
        return self._launcher(state.launch.kind).is_running(state.launch)

    def _confirm_ready(
        self,
        state: RuntimeState,
        *,
        timeout_seconds: float,
    ) -> None:
        self.health_checker(state.health_url, timeout_seconds)
        if not self._is_running(state):
            raise DeploymentError(
                f"{state.deployment.id} exited before readiness was committed; "
                "the health response may belong to another service on the port"
            )

    def _stop_state(self, state: RuntimeState) -> None:
        if state.launch.kind == "external":
            return
        self._launcher(state.launch.kind).stop(
            state.launch,
            timeout_seconds=self.stop_timeout_seconds,
        )

    def _verify_saved_artifact(self, state: RuntimeState) -> None:
        if state.deployment.backend in {BackendKind.EXTERNAL, BackendKind.MOCK}:
            return
        if state.artifact_path is None:
            raise DeploymentError(
                f"saved deployment {state.deployment.id!r} has no artifact path"
            )
        from .storage import ArtifactStore

        with ArtifactStore(self.paths) as store:
            expected_view = store.view_path(state.deployment.artifact_id).absolute()
            recorded = Path(state.artifact_path).absolute()
            if recorded != expected_view:
                raise DeploymentError(
                    f"saved artifact path {recorded} does not match registered view "
                    f"{expected_view}"
                )
            store.verify(state.deployment.artifact_id, verify_view=True)

    @staticmethod
    def _resolved_rollback_plan(state: RuntimeState) -> LaunchPlan:
        plan = state.plan
        if state.launch.kind == "process" and state.launch.resolved_executable:
            command = plan.command
            if command:
                updates: dict[str, Any] = {
                    "command": (state.launch.resolved_executable, *command[1:])
                }
                if state.deployment.runtime_lock_id is not None:
                    updates.update(
                        expected_executable_sha256=state.launch.executable_sha256,
                        expected_executable_version_contains=(
                            state.launch.executable_version
                        ),
                        expected_executable_root=(
                            plan.expected_executable_root
                            or str(Path(state.launch.resolved_executable).anchor)
                        ),
                    )
                plan = replace(plan, **updates)
        elif state.launch.kind == "container" and state.launch.resolved_image:
            plan = replace(plan, image=state.launch.resolved_image)
        return plan


def origin_url(value: str) -> str:
    """Return only the scheme and authority of a URL (useful to health probes)."""

    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
