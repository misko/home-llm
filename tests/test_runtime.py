from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import psutil

import llm_lab.runtime as runtime_module

from llm_lab.errors import DeploymentError, StoragePolicyError
from llm_lab.paths import LabPaths
from llm_lab.runtime import (
    DockerLauncher,
    ExclusiveGpuLock,
    LaunchPlan,
    LaunchRecord,
    ProcessLauncher,
    RuntimeManager,
    clear_active_state,
    build_backend_command,
    read_active_state,
    verify_runtime_lock,
    wait_for_readiness,
    write_active_state,
)
from llm_lab.schema import (
    BackendKind,
    DeploymentSpec,
    RuntimeImage,
    RuntimeLockSpec,
)


class FakeLauncher:
    def __init__(self) -> None:
        self.started: list[LaunchPlan] = []
        self.stopped: list[LaunchRecord] = []
        self.running: set[int | str] = set()
        self._counter = 1000

    def start(self, plan: LaunchPlan) -> LaunchRecord:
        self._counter += 1
        self.started.append(plan)
        if plan.kind == "container":
            identity: int | str = f"container-{self._counter}"
            self.running.add(identity)
            return LaunchRecord(
                kind="container",
                command=("docker", "run", *plan.command),
                started_at="2026-09-03T00:00:00+00:00",
                container_id=str(identity),
                container_name=f"test-{plan.deployment_id}",
                requested_image=plan.image,
                resolved_image="sha256:" + "1" * 64,
                resolved_executable="/usr/bin/docker",
            )
        identity = self._counter
        self.running.add(identity)
        return LaunchRecord(
            kind="process",
            command=("/resolved/mock", *plan.command[1:]),
            started_at="2026-09-03T00:00:00+00:00",
            pid=identity,
            process_create_time=float(identity),
            resolved_executable="/resolved/mock",
        )

    def is_running(self, record: LaunchRecord) -> bool:
        identity: int | str | None = (
            record.pid if record.kind == "process" else record.container_id
        )
        return identity in self.running

    def stop(self, record: LaunchRecord, timeout_seconds: float = 10.0) -> None:
        self.stopped.append(record)
        identity: int | str | None = (
            record.pid if record.kind == "process" else record.container_id
        )
        self.running.discard(identity)


def _paths(tmp_path: Path) -> LabPaths:
    repo = tmp_path / "repo"
    repo.mkdir()
    return LabPaths.discover(repo_root=repo, data_root=tmp_path / "data")


def _deployment(
    deployment_id: str,
    *,
    backend: BackendKind = BackendKind.MOCK,
    port: int = 18089,
    executable: str | None = None,
    image: RuntimeImage | None = None,
    external_base_url: str | None = None,
    mmproj: str | None = None,
    extra_args: tuple[str, ...] = (),
    reasoning_mode: str = "auto",
    speculative_mode: str = "none",
    speculative_draft_tokens: int = 2,
) -> DeploymentSpec:
    return DeploymentSpec(
        id=deployment_id,
        artifact_id="artifact",
        public_alias=f"alias-{deployment_id}",
        backend=backend,
        executable=executable,
        image=image,
        external_base_url=external_base_url,
        port=port,
        startup_timeout_seconds=0.1,
        mmproj=mmproj,
        extra_args=extra_args,
        reasoning_mode=reasoning_mode,
        speculative_mode=speculative_mode,
        speculative_draft_tokens=speculative_draft_tokens,
    )


def test_activate_status_stop_publishes_resolved_state(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    launcher = FakeLauncher()
    probes: list[tuple[str, float]] = []

    def healthy(url: str, timeout: float) -> None:
        probes.append((url, timeout))

    manager = RuntimeManager(
        paths,
        process_launcher=launcher,
        container_launcher=launcher,
        health_checker=healthy,
    )
    state = manager.activate(_deployment("mock-one"), artifact_path=None)

    assert state.phase == "ready"
    assert state.launch.pid == 1001
    assert state.launch.resolved_executable == "/resolved/mock"
    assert probes == [("http://127.0.0.1:18089/health", 0.1)]
    on_disk = json.loads(paths.active_state_path.read_text())
    assert on_disk["launch"]["command"][0] == "/resolved/mock"
    assert on_disk["plan"]["command"][:3] == [
        sys.executable,
        "-m",
        "llm_lab.mock_backend",
    ]
    assert not list(paths.active_state_path.parent.glob("*.tmp"))

    status = manager.status()
    assert status.active and status.ready and status.running
    assert status.healthy is True
    stopped = manager.stop()
    assert stopped is not None and stopped.deployment_id == "mock-one"
    assert read_active_state(paths) is None
    assert manager.stop() is None


def test_process_stop_treats_zombie_only_group_as_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_group = 4242
    signals: list[tuple[int, int]] = []

    class ZombieLeader:
        pid = process_group
        info = {"pid": process_group, "status": psutil.STATUS_ZOMBIE}

        def is_running(self) -> bool:
            return True

        def status(self) -> str:
            return psutil.STATUS_ZOMBIE

        def wait(self, timeout: float = 0) -> int:
            assert timeout == 0
            return 0

    class InaccessibleUnrelatedProcess:
        info = {"pid": process_group - 1, "status": None}

    leader = ZombieLeader()
    record = LaunchRecord(
        kind="process",
        command=("/reviewed/server",),
        started_at="2026-09-06T00:00:00+00:00",
        pid=process_group,
        process_create_time=123.0,
    )
    monkeypatch.setattr(runtime_module, "_recorded_process", lambda _: leader)
    monkeypatch.setattr(
        runtime_module.psutil,
        "process_iter",
        lambda **_: iter((InaccessibleUnrelatedProcess(), leader)),
    )

    def process_group_for(pid: int) -> int:
        if pid == process_group - 1:
            raise PermissionError("unrelated protected process")
        return process_group

    monkeypatch.setattr(runtime_module.os, "getpgid", process_group_for)
    monkeypatch.setattr(
        runtime_module.os,
        "killpg",
        lambda pgid, sent_signal: signals.append((pgid, sent_signal)),
    )

    launcher = ProcessLauncher()
    assert launcher.is_running(record) is False
    launcher.stop(record, timeout_seconds=0.01)

    # Signal-zero deliberately succeeds for the zombie group. Liveness comes
    # from member state, so stop does not wait or escalate to SIGKILL.
    assert (process_group, 0) in signals
    assert (process_group, signal.SIGTERM) in signals
    assert (process_group, signal.SIGKILL) not in signals


def test_zombie_group_leader_does_not_hide_live_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_group = 4343

    class Member:
        def __init__(self, pid: int, status: str) -> None:
            self.info = {"pid": pid, "status": status}

        def status(self) -> str:
            return str(self.info["status"])

    members = (
        Member(process_group, psutil.STATUS_ZOMBIE),
        Member(process_group + 1, psutil.STATUS_SLEEPING),
    )
    monkeypatch.setattr(runtime_module.os, "killpg", lambda *_: None)
    monkeypatch.setattr(runtime_module.os, "getpgid", lambda _: process_group)
    monkeypatch.setattr(
        runtime_module.psutil,
        "process_iter",
        lambda **_: iter(members),
    )

    assert runtime_module._process_group_exists(process_group) is True


def test_verify_runtime_lock_binds_recipe_path_bytes_and_version(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.initialize()
    binary = paths.data_root / "cache/runtimes/runtime/bin/server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 'server build commit abcdef0'\n", encoding="utf-8")
    binary.chmod(0o555)
    artifact = paths.view_root / "artifact"
    artifact.mkdir()
    (artifact / "model.gguf").write_bytes(b"fixture")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    runtime_lock = RuntimeLockSpec(
        id="runtime-lock",
        source="https://example.invalid/runtime.git",
        commit="abcdef0" + "0" * 33,
        build={"generator": "fixture"},
        binary="cache/runtimes/runtime/bin/server",
        binary_sha256=digest,
        version_contains="commit abcdef0",
    )
    deployment = DeploymentSpec(
        id="locked",
        artifact_id="artifact",
        public_alias="locked",
        backend=BackendKind.LLAMA_CPP,
        executable="${LLM_LAB_DATA}/cache/runtimes/runtime/bin/server",
        runtime_lock_id="runtime-lock",
        port=19001,
    )

    report = verify_runtime_lock(paths, deployment, runtime_lock, artifact)
    assert report.binary_sha256 == digest
    assert report.binary == str(binary)
    assert "commit abcdef0" in report.executable_version

    binary.chmod(0o755)
    binary.write_text("#!/bin/sh\necho tampered\n", encoding="utf-8")
    binary.chmod(0o555)
    with pytest.raises(DeploymentError, match="SHA-256 mismatch"):
        verify_runtime_lock(paths, deployment, runtime_lock, artifact)


def test_manager_cannot_skip_a_declared_runtime_lock(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    launcher = FakeLauncher()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.gguf").write_bytes(b"model")
    payload = _deployment(
        "locked-host",
        backend=BackendKind.LLAMA_CPP,
        executable="llama-server",
    ).model_dump(mode="python")
    payload["runtime_lock_id"] = "required-lock"
    deployment = DeploymentSpec.model_validate(payload)
    manager = RuntimeManager(
        paths,
        process_launcher=launcher,
        container_launcher=launcher,
        health_checker=lambda *_: None,
    )

    with pytest.raises(DeploymentError, match="requires runtime lock"):
        manager.activate(deployment, artifact)

    assert launcher.started == []


def test_process_launch_executes_the_same_inode_it_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "server"
    replacement = tmp_path / "replacement"
    marker = tmp_path / "executed"
    original_bytes = (
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = --version ]; then echo original-version; exit 0; fi\n"
        f"printf original > '{marker}'\n"
    ).encode()
    replacement_bytes = (
        "#!/bin/sh\n"
        f"printf replacement > '{marker}'\n"
    ).encode()
    executable.write_bytes(original_bytes)
    replacement.write_bytes(replacement_bytes)
    executable.chmod(0o755)
    replacement.chmod(0o755)

    def replace_during_probe(*_args, **_kwargs) -> str:
        os.replace(replacement, executable)
        return "original-version"

    monkeypatch.setattr(
        "llm_lab.runtime._probe_executable_version", replace_during_probe
    )
    record = ProcessLauncher().start(
        LaunchPlan(
            kind="process",
            deployment_id="inode-bound",
            command=(str(executable),),
            environment={},
            base_url="http://127.0.0.1:1",
            health_url="http://127.0.0.1:1/health",
        )
    )
    deadline = time.monotonic() + 3
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert marker.read_text() == "original"
    assert executable.read_bytes() == replacement_bytes
    assert record.executable_sha256 == hashlib.sha256(original_bytes).hexdigest()


def test_locked_process_launch_executes_reviewed_inode_after_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_root = tmp_path / "data"
    executable = trusted_root / "cache/runtime/server"
    executable.parent.mkdir(parents=True)
    replacement = tmp_path / "replacement"
    marker = tmp_path / "executed"
    original_bytes = (
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = --version ]; then echo locked-version; exit 0; fi\n"
        f"printf reviewed > '{marker}'\n"
    ).encode()
    replacement_bytes = (
        "#!/bin/sh\n"
        f"printf substituted > '{marker}'\n"
    ).encode()
    executable.write_bytes(original_bytes)
    replacement.write_bytes(replacement_bytes)
    executable.chmod(0o555)
    replacement.chmod(0o555)

    def replace_during_probe(*_args, **_kwargs) -> str:
        os.replace(replacement, executable)
        return "locked-version"

    monkeypatch.setattr(
        "llm_lab.runtime._probe_executable_version", replace_during_probe
    )
    record = ProcessLauncher().start(
        LaunchPlan(
            kind="process",
            deployment_id="locked-inode-bound",
            command=(str(executable),),
            environment={},
            base_url="http://127.0.0.1:1",
            health_url="http://127.0.0.1:1/health",
            expected_executable_sha256=hashlib.sha256(original_bytes).hexdigest(),
            expected_executable_version_contains="locked-version",
            expected_executable_root=str(trusted_root),
        )
    )
    deadline = time.monotonic() + 3
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert marker.read_text(encoding="utf-8") == "reviewed"
    assert executable.read_bytes() == replacement_bytes
    assert record.executable_sha256 == hashlib.sha256(original_bytes).hexdigest()


def test_locked_process_launch_rejects_symlinked_executable_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted_root = tmp_path / "data"
    (trusted_root / "cache").mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    executable = victim / "server"
    executable.write_text("#!/bin/sh\necho should-not-run\n", encoding="utf-8")
    executable.chmod(0o555)
    (trusted_root / "cache/runtime").symlink_to(victim, target_is_directory=True)

    def forbidden_popen(*_args, **_kwargs):
        pytest.fail("a locked binary beneath a symlink must never execute")

    monkeypatch.setattr("llm_lab.runtime.subprocess.Popen", forbidden_popen)
    with pytest.raises(StoragePolicyError, match="safe real directory"):
        ProcessLauncher().start(
            LaunchPlan(
                kind="process",
                deployment_id="symlinked-runtime",
                command=(str(trusted_root / "cache/runtime/server"),),
                environment={},
                base_url="http://127.0.0.1:1",
                health_url="http://127.0.0.1:1/health",
                expected_executable_sha256=hashlib.sha256(
                    executable.read_bytes()
                ).hexdigest(),
                expected_executable_version_contains="should-not-run",
                expected_executable_root=str(trusted_root),
            )
        )


def test_locked_process_launch_rejects_loader_and_llama_argument_environment(
    tmp_path: Path,
) -> None:
    trusted_root = tmp_path / "data"
    executable = trusted_root / "cache/runtime/server"
    executable.parent.mkdir(parents=True)
    marker = tmp_path / "environment-result"
    executable_bytes = (
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = --version ]; then echo locked-version; exit 0; fi\n"
        "if [ -n \"${LD_PRELOAD:-}\" ] || [ -n \"${LLAMA_ARG_MODEL:-}\" ]; then\n"
        f"  printf unsafe > '{marker}'\n"
        "else\n"
        f"  printf clean > '{marker}'\n"
        "fi\n"
    ).encode()
    executable.write_bytes(executable_bytes)
    executable.chmod(0o555)

    with pytest.raises(DeploymentError, match="unsafe environment variables"):
        ProcessLauncher().start(
            LaunchPlan(
                kind="process",
                deployment_id="locked-environment",
                command=(str(executable),),
                environment={
                    "LD_PRELOAD": "/tmp/unreviewed.so",
                    "LLAMA_ARG_MODEL": "/tmp/unreviewed.gguf",
                },
                base_url="http://127.0.0.1:1",
                health_url="http://127.0.0.1:1/health",
                expected_executable_sha256=hashlib.sha256(
                    executable_bytes
                ).hexdigest(),
                expected_executable_version_contains="locked-version",
                expected_executable_root=str(trusted_root),
            )
        )

    assert not marker.exists()


def test_failed_health_restores_previous_deployment(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    launcher = FakeLauncher()

    def selective_health(url: str, _: float) -> None:
        if ":18090/" in url:
            raise DeploymentError("deliberate readiness failure")

    manager = RuntimeManager(
        paths,
        process_launcher=launcher,
        container_launcher=launcher,
        health_checker=selective_health,
    )
    manager.activate(_deployment("previous", port=18089))

    with pytest.raises(DeploymentError, match=r"restored previous"):
        manager.activate(_deployment("candidate", port=18090))

    restored = read_active_state(paths)
    assert restored is not None
    assert restored.deployment_id == "previous"
    assert restored.phase == "ready"
    assert [plan.deployment_id for plan in launcher.started] == [
        "previous",
        "candidate",
        "previous",
    ]
    assert len(launcher.stopped) == 2
    assert restored.launch.pid != 1001


def test_benchmark_lease_blocks_runtime_lifecycle_changes(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    benchmark_manager = RuntimeManager(
        paths,
        gpu_lock_timeout_seconds=0.05,
    )
    competing_manager = RuntimeManager(
        paths,
        gpu_lock_timeout_seconds=0.05,
    )

    with benchmark_manager.benchmark_lease(check_health=False) as status:
        assert status.active is False
        with pytest.raises(DeploymentError, match="timed out acquiring GPU lock"):
            competing_manager.stop()


def test_failed_candidate_cleanup_never_starts_second_gpu_owner(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    class CleanupFailureLauncher(FakeLauncher):
        def stop(self, record: LaunchRecord, timeout_seconds: float = 10.0) -> None:
            if record.pid == 1002:
                self.stopped.append(record)
                raise DeploymentError("candidate refused to stop")
            super().stop(record, timeout_seconds)

    launcher = CleanupFailureLauncher()

    def selective_health(url: str, _: float) -> None:
        if ":18090/" in url:
            raise DeploymentError("deliberate readiness failure")

    manager = RuntimeManager(
        paths,
        process_launcher=launcher,
        container_launcher=launcher,
        health_checker=selective_health,
    )
    manager.activate(_deployment("previous", port=18089))

    with pytest.raises(DeploymentError, match="rollback was not attempted"):
        manager.activate(_deployment("candidate", port=18090))

    failed = read_active_state(paths)
    assert failed is not None
    assert failed.deployment_id == "candidate"
    assert failed.phase == "failed"
    assert [plan.deployment_id for plan in launcher.started] == [
        "previous",
        "candidate",
    ]
    assert launcher.running == {1002}


def test_failed_initial_activation_leaves_no_active_state(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    launcher = FakeLauncher()

    def unhealthy(_: str, __: float) -> None:
        raise DeploymentError("not ready")

    manager = RuntimeManager(
        paths,
        process_launcher=launcher,
        health_checker=unhealthy,
    )
    with pytest.raises(DeploymentError, match="activation of broken failed"):
        manager.activate(_deployment("broken"))

    assert read_active_state(paths) is None
    assert len(launcher.stopped) == 1


def test_health_response_does_not_mask_candidate_process_exit(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    class ExitedLauncher(FakeLauncher):
        def is_running(self, record: LaunchRecord) -> bool:
            return False

    launcher = ExitedLauncher()
    manager = RuntimeManager(
        paths,
        process_launcher=launcher,
        health_checker=lambda _url, _timeout: None,
    )

    with pytest.raises(DeploymentError, match="exited before readiness"):
        manager.activate(_deployment("port-conflict"))

    assert read_active_state(paths) is None
    assert len(launcher.stopped) == 1


def test_external_activation_needs_no_process_and_preserves_v1_base(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    launcher = FakeLauncher()
    manager = RuntimeManager(
        paths,
        process_launcher=launcher,
        health_checker=lambda _url, _timeout: None,
    )
    state = manager.activate(
        _deployment(
            "remote",
            backend=BackendKind.EXTERNAL,
            external_base_url="https://model.invalid/v1/",
        )
    )

    assert state.base_url == "https://model.invalid/v1"
    assert state.health_url == "https://model.invalid/health"
    assert state.launch.kind == "external"
    assert launcher.started == []
    assert manager.status(check_health=False).running is True


def test_llama_host_and_container_commands_map_artifact_paths(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    weights = artifact / "model-Q4_K_M.gguf"
    projector = artifact / "mmproj-model-f16.gguf"
    weights.write_bytes(b"weights")
    projector.write_bytes(b"projector")

    host = _deployment(
        "llama-host",
        backend=BackendKind.LLAMA_CPP,
        executable="${LLM_LAB_DATA}/cache/llama.cpp/llama-server",
        mmproj="/models/mmproj-model-f16.gguf",
        reasoning_mode="off",
        speculative_mode="mtp",
        speculative_draft_tokens=2,
    )
    host_plan = build_backend_command(host, artifact, paths=paths)
    assert host_plan.kind == "process"
    assert host_plan.command[0] == str(
        paths.data_root / "cache/llama.cpp/llama-server"
    )
    assert str(weights.resolve()) in host_plan.command
    assert str(projector.resolve()) in host_plan.command
    reasoning_index = host_plan.command.index("--reasoning")
    assert host_plan.command[reasoning_index + 1] == "off"
    assert host_plan.command[
        host_plan.command.index("--spec-type") + 1
    ] == "draft-mtp"
    assert host_plan.command[
        host_plan.command.index("--spec-draft-n-max") + 1
    ] == "2"

    container = _deployment(
        "llama-container",
        backend=BackendKind.LLAMA_CPP,
        image=RuntimeImage(
            reference="example.invalid/llama:server",
            digest="sha256:" + "a" * 64,
        ),
        mmproj="/models/mmproj-model-f16.gguf",
    )
    container_plan = build_backend_command(container, artifact, paths=paths)
    assert container_plan.kind == "container"
    assert container_plan.image == (
        "example.invalid/llama:server@sha256:" + "a" * 64
    )
    assert "/models/model-Q4_K_M.gguf" in container_plan.command
    assert "/models/mmproj-model-f16.gguf" in container_plan.command
    assert container_plan.command[container_plan.command.index("--reasoning") + 1] == (
        "auto"
    )
    assert container_plan.mounts[0].source == str(artifact.resolve())
    assert container_plan.mounts[0].target == "/models"


@pytest.mark.parametrize(
    "extra_args",
    (
        ("--model", "/tmp/unregistered.gguf"),
        ("--model=/tmp/unregistered.gguf",),
        ("--port", "9999"),
        ("--ctx-size=1",),
        ("--mmproj", "/tmp/unregistered.gguf"),
        ("-mm", "/tmp/unregistered.gguf"),
        ("--n-gpu-layers", "0"),
        ("-rea", "on"),
        ("--hf-repo", "unreviewed/model:Q4"),
        ("-mu", "https://example.invalid/unreviewed.gguf"),
        ("--models-dir", "/tmp/unregistered"),
        ("--", "--model", "/tmp/unregistered.gguf"),
    ),
)
def test_llama_extra_args_cannot_override_identity_or_runtime_contract(
    tmp_path: Path,
    extra_args: tuple[str, ...],
) -> None:
    paths = _paths(tmp_path)
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.gguf").write_bytes(b"model")
    deployment = _deployment(
        "protected",
        backend=BackendKind.LLAMA_CPP,
        executable="llama-server",
        extra_args=extra_args,
    )

    with pytest.raises(DeploymentError, match="may not override protected"):
        build_backend_command(deployment, artifact, paths=paths)


def test_llama_container_mounts_cas_for_relative_view_symlinks(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    paths.initialize()
    digest = "b" * 64
    blob = paths.blob_root / digest
    blob.write_bytes(b"weights")
    view = paths.view_root / "artifact"
    view.mkdir()
    (view / "model.gguf").symlink_to(Path("../../blobs/sha256") / digest)
    deployment = _deployment(
        "llama-symlink-view",
        backend=BackendKind.LLAMA_CPP,
        image=RuntimeImage(reference="example.invalid/llama:server"),
    )

    plan = build_backend_command(deployment, view, paths=paths)

    assert "/models/model.gguf" in plan.command
    assert [(mount.source, mount.target) for mount in plan.mounts] == [
        (str(view.absolute()), "/models"),
        (str(paths.blob_root.absolute()), "/blobs"),
    ]


@pytest.mark.parametrize(
    ("backend", "executable", "expected"),
    [
        (BackendKind.VLLM, "bin/vllm", ("serve", "--max-model-len")),
        (BackendKind.SGLANG, "bin/sglang", ("serve", "--context-length")),
        (
            BackendKind.TENSORRT_LLM,
            "bin/trtllm-serve",
            ("--max_seq_len", "--max_batch_size"),
        ),
    ],
)
def test_generic_backend_profiles(
    tmp_path: Path,
    backend: BackendKind,
    executable: str,
    expected: tuple[str, str],
) -> None:
    paths = _paths(tmp_path)
    artifact = tmp_path / "hf-model"
    artifact.mkdir()
    deployment = _deployment(
        f"profile-{backend.value.replace('_', '-')}",
        backend=backend,
        executable=executable,
    )

    plan = build_backend_command(deployment, artifact, paths=paths)

    assert plan.kind == "process"
    assert plan.command[0] == executable
    assert all(value in plan.command for value in expected)
    assert str(artifact.resolve()) in plan.command


def test_process_launcher_resolves_repo_relative_binary_without_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "bin" / "server"
    executable.parent.mkdir()
    executable_bytes = (
        b'#!/bin/sh\nif [ "$1" = "--version" ]; then '
        b'printf "test-server 1.2.3\\n"; fi\n'
    )
    executable.write_bytes(executable_bytes)
    executable.chmod(0o755)
    observed: dict[str, object] = {}

    class Started:
        pid = os.getpid()

    class VersionProcess:
        def __init__(self, command) -> None:
            self.args = command
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def communicate(self, _input=None, timeout=None):
            return ("test-server 1.2.3\n", "")

        def poll(self) -> int:
            return self.returncode

    def fake_popen(command, **kwargs):
        if list(command) == [str(executable.resolve()), "--version"]:
            return VersionProcess(command)
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Started()

    monkeypatch.setattr("llm_lab.runtime.subprocess.Popen", fake_popen)
    plan = LaunchPlan(
        kind="process",
        deployment_id="relative",
        command=("bin/server", "--flag"),
        environment={},
        base_url="http://127.0.0.1:1",
        health_url="http://127.0.0.1:1/health",
        log_path=str(tmp_path / "server.log"),
        working_directory=str(tmp_path),
    )

    record = ProcessLauncher().start(plan)

    assert record.resolved_executable == str(executable.resolve())
    assert record.executable_sha256 == hashlib.sha256(executable_bytes).hexdigest()
    assert record.executable_version == "test-server 1.2.3"
    assert observed["command"] == (str(executable.resolve()), "--flag")
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert "shell" not in kwargs
    assert kwargs["start_new_session"] is True

    legacy = record.to_dict()
    legacy.pop("executable_sha256")
    legacy.pop("executable_version")
    restored = LaunchRecord.from_dict(legacy)
    assert restored.executable_sha256 is None
    assert restored.executable_version is None


@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory"])
def test_process_launcher_rejects_unsafe_log_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    executable = tmp_path / "server"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    log_path = tmp_path / "server.log"
    sentinel = tmp_path / "outside.log"
    if unsafe_kind == "symlink":
        sentinel.write_text("do-not-change")
        log_path.symlink_to(sentinel)
    else:
        log_path.mkdir()

    monkeypatch.setattr(
        "llm_lab.runtime._probe_executable_version",
        lambda *_args, **_kwargs: None,
    )

    def forbidden_popen(*_args, **_kwargs):
        pytest.fail("an unsafe log path must be rejected before process creation")

    monkeypatch.setattr("llm_lab.runtime.subprocess.Popen", forbidden_popen)
    plan = LaunchPlan(
        kind="process",
        deployment_id="unsafe-log",
        command=(str(executable),),
        environment={},
        base_url="http://127.0.0.1:1",
        health_url="http://127.0.0.1:1/health",
        log_path=str(log_path),
        working_directory=str(tmp_path),
    )

    with pytest.raises(DeploymentError, match="unsafe process log path"):
        ProcessLauncher().start(plan)
    if unsafe_kind == "symlink":
        assert sentinel.read_text() == "do-not-change"


@pytest.mark.parametrize("failure_stage", ["start", "inspect", "record"])
def test_docker_launcher_cleans_up_after_detached_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    checked: list[tuple[list[str], str]] = []
    cleanup: list[tuple[list[str], dict[str, object]]] = []
    docker = str(Path(sys.executable).absolute())

    def fake_checked(command, action):
        checked.append((list(command), action))
        if action == "start Docker container":
            if failure_stage == "start":
                raise DeploymentError("start failed")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="a" * 64 + "\n",
                stderr="",
            )
        if failure_stage == "inspect":
            raise DeploymentError("inspect failed")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="sha256:" + "c" * 64 + "\n",
            stderr="",
        )

    def fake_run(command, **kwargs):
        cleanup.append((list(command), kwargs))
        stdout = "deadbeef\n" if "inspect" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("llm_lab.runtime.secrets.token_hex", lambda _size: "deadbeef")
    monkeypatch.setattr("llm_lab.runtime._run_checked", fake_checked)
    monkeypatch.setattr("llm_lab.runtime.subprocess.run", fake_run)
    if failure_stage == "record":

        def fail_record(**_kwargs):
            raise RuntimeError("record failed")

        monkeypatch.setattr("llm_lab.runtime.LaunchRecord", fail_record)

    plan = LaunchPlan(
        kind="container",
        deployment_id="candidate",
        command=("serve",),
        environment={},
        base_url="http://127.0.0.1:1",
        health_url="http://127.0.0.1:1/health",
        image="example.invalid/server@sha256:" + "b" * 64,
    )

    with pytest.raises(
        (DeploymentError, RuntimeError),
        match=f"{failure_stage} failed",
    ):
        DockerLauncher(executable=docker).start(plan)

    expected_actions = ["start Docker container"]
    if failure_stage != "start":
        expected_actions.append("resolve Docker image")
    assert [action for _, action in checked] == expected_actions
    assert cleanup == [
        (
            [
                docker,
                "inspect",
                "--format",
                '{{ index .Config.Labels "llm-lab.launch" }}',
                "llm-lab-candidate-deadbeef",
            ],
            {"check": False, "capture_output": True, "text": True},
        ),
        (
            [docker, "rm", "--force", "llm-lab-candidate-deadbeef"],
            {"check": False, "capture_output": True, "text": True},
        )
    ]


def test_readiness_retries_until_success() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503 if attempts == 1 else 200)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        wait_for_readiness(
            "https://backend.invalid/health",
            1.0,
            interval_seconds=0.0,
            client=client,
        )
    assert attempts == 2


def test_gpu_lock_is_exclusive(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    with ExclusiveGpuLock(lock_path):
        with pytest.raises(DeploymentError, match="timed out acquiring GPU lock"):
            with ExclusiveGpuLock(
                lock_path,
                timeout_seconds=0.02,
                poll_seconds=0.005,
            ):
                pass


@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory"])
def test_gpu_lock_rejects_unsafe_target(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    lock_path = tmp_path / "gpu.lock"
    sentinel = tmp_path / "outside.lock"
    if unsafe_kind == "symlink":
        sentinel.write_text("do-not-truncate")
        lock_path.symlink_to(sentinel)
    else:
        lock_path.mkdir()

    with pytest.raises(DeploymentError, match="unsafe GPU lock path"):
        with ExclusiveGpuLock(lock_path):
            pass
    if unsafe_kind == "symlink":
        assert sentinel.read_text() == "do-not-truncate"


@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory"])
def test_active_state_operations_reject_unsafe_target(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    paths = _paths(tmp_path)
    launcher = FakeLauncher()
    state = RuntimeManager(
        paths,
        process_launcher=launcher,
        health_checker=lambda _url, _timeout: None,
    ).activate(_deployment("state-fixture"))
    paths.active_state_path.unlink()

    sentinel = tmp_path / "outside-state.json"
    if unsafe_kind == "symlink":
        sentinel.write_text("do-not-change")
        paths.active_state_path.symlink_to(sentinel)
    else:
        paths.active_state_path.mkdir()

    operations = (
        lambda: read_active_state(paths),
        lambda: write_active_state(paths, state),
        lambda: clear_active_state(paths),
    )
    for operation in operations:
        with pytest.raises(DeploymentError, match="unsafe active state path"):
            operation()
    if unsafe_kind == "symlink":
        assert sentinel.read_text() == "do-not-change"
