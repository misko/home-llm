from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from llm_lab.paths import LabPaths


REPOSITORY_ROOT = Path(__file__).parents[1]


def _executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _write_runtime_lock(
    path: Path,
    *,
    source: str,
    commit: str,
    binary_sha256: str,
    shared_libraries: bool = False,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "fixture-runtime",
                "source": source,
                "commit": commit,
                "build": {
                    "options": {
                        "CMAKE_CUDA_ARCHITECTURES": "89",
                        "BUILD_SHARED_LIBS": shared_libraries,
                    }
                },
                "binary": "cache/runtimes/fixture/llama-server",
                "binary_sha256": binary_sha256,
                "version_contains": "reviewed-version",
            }
        ),
        encoding="utf-8",
    )


def _run_build_script(
    *,
    data: Path,
    checkout: Path,
    build: Path,
    lock: Path | None = None,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (
        "LLM_LAB_ALLOW_UNLOCKED_BUILD",
        "LLAMA_CPP_SOURCE",
        "LLAMA_CPP_REF",
        "LLAMA_CPP_CUDA_ARCH",
    ):
        environment.pop(name, None)
    environment.update(
        LLM_LAB_DATA=str(data),
        LLAMA_CPP_DIR=str(checkout),
        LLAMA_CPP_BUILD_DIR=str(build),
    )
    if lock is not None:
        environment["LLAMA_CPP_LOCK_FILE"] = str(lock)
    if extra_environment is not None:
        environment.update(extra_environment)
    return subprocess.run(
        ["bash", "scripts/build_llama_cpp.sh"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_mismatched_candidate_never_executes_or_replaces_production(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    data = tmp_path / "data"
    checkout = data / "cache/llama.cpp"
    (checkout / ".git").mkdir(parents=True)
    candidate_build = data / "work/verify/candidate"
    marker = tmp_path / "candidate-executed"
    source = "https://example.invalid/llama.cpp.git"
    commit = "a" * 40
    lock = tmp_path / "runtime-lock.yaml"
    lock.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "fixture-runtime",
                "source": source,
                "commit": commit,
                "build": {
                    "options": {
                        "CMAKE_CUDA_ARCHITECTURES": "89",
                        "BUILD_SHARED_LIBS": False,
                    }
                },
                "binary": "cache/runtimes/fixture/llama-server",
                "binary_sha256": "0" * 64,
                "version_contains": "reviewed-version",
            }
        ),
        encoding="utf-8",
    )
    _executable(
        fake_bin / "git",
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        f"  *'remote get-url origin'*) echo '{source}' ;;\n"
        f"  *'rev-parse HEAD'*) echo '{commit}' ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
    )
    _executable(
        fake_bin / "cmake",
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = --build ]; then\n"
        "  mkdir -p \"${LLAMA_CPP_BUILD_DIR}/bin\"\n"
        "  printf '#!/bin/sh\\nprintf candidate-ran > \"${TEST_MARKER}\"\\necho unreviewed-version\\n' "
        "> \"${LLAMA_CPP_BUILD_DIR}/bin/llama-server\"\n"
        "  chmod 755 \"${LLAMA_CPP_BUILD_DIR}/bin/llama-server\"\n"
        "fi\n",
    )
    production = data / "cache/runtimes/fixture/llama-server"
    production.parent.mkdir(parents=True)
    production.write_bytes(b"trusted-production")

    environment = os.environ.copy()
    environment.update(
        PATH=f"{fake_bin}:{environment['PATH']}",
        LLM_LAB_DATA=str(data),
        LLAMA_CPP_DIR=str(checkout),
        LLAMA_CPP_BUILD_DIR=str(candidate_build),
        LLAMA_CPP_LOCK_FILE=str(lock),
        TEST_MARKER=str(marker),
    )
    completed = subprocess.run(
        ["bash", "scripts/build_llama_cpp.sh"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 3
    assert "production binary was not touched" in completed.stderr
    assert production.read_bytes() == b"trusted-production"
    assert not marker.exists()


def test_reviewed_static_candidate_is_installed_immutable(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    data = tmp_path / "data"
    checkout = data / "cache/llama.cpp"
    (checkout / ".git").mkdir(parents=True)
    candidate_build = data / "work/verify/candidate"
    candidate_source = tmp_path / "reviewed-server"
    candidate_source.write_bytes(
        b"#!/bin/sh\n"
        b"if [ \"${1:-}\" = --version ]; then echo reviewed-version; fi\n"
    )
    candidate_sha256 = hashlib.sha256(candidate_source.read_bytes()).hexdigest()
    source = "https://example.invalid/llama.cpp.git"
    commit = "a" * 40
    lock = tmp_path / "runtime-lock.yaml"
    _write_runtime_lock(
        lock,
        source=source,
        commit=commit,
        binary_sha256=candidate_sha256,
    )
    cmake_log = tmp_path / "cmake.log"
    _executable(
        fake_bin / "git",
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        f"  *'remote get-url origin'*) echo '{source}' ;;\n"
        f"  *'rev-parse HEAD'*) echo '{commit}' ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
    )
    _executable(
        fake_bin / "cmake",
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"${TEST_CMAKE_LOG}\"\n"
        "if [ \"${1:-}\" = --build ]; then\n"
        "  mkdir -p \"${LLAMA_CPP_BUILD_DIR}/bin\"\n"
        "  cp \"${TEST_CANDIDATE_SOURCE}\" "
        "\"${LLAMA_CPP_BUILD_DIR}/bin/llama-server\"\n"
        "  chmod 755 \"${LLAMA_CPP_BUILD_DIR}/bin/llama-server\"\n"
        "fi\n",
    )
    _executable(fake_bin / "ldd", "#!/bin/sh\necho linux-vdso.so.1\n")

    completed = _run_build_script(
        data=data,
        checkout=checkout,
        build=candidate_build,
        lock=lock,
        extra_environment={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "TEST_CANDIDATE_SOURCE": str(candidate_source),
            "TEST_CMAKE_LOG": str(cmake_log),
        },
    )

    production = data / "cache/runtimes/fixture/llama-server"
    assert completed.returncode == 0, completed.stderr
    assert production.read_bytes() == candidate_source.read_bytes()
    assert stat.S_IMODE(production.stat().st_mode) == 0o555
    assert "-DBUILD_SHARED_LIBS=OFF" in cmake_log.read_text(encoding="utf-8")


@pytest.mark.parametrize("escaped", ("checkout", "build"))
def test_build_script_rejects_paths_outside_managed_subtrees(
    tmp_path: Path,
    escaped: str,
) -> None:
    data = tmp_path / "data"
    checkout = data / "cache/llama.cpp"
    build = data / "work/verify/candidate"
    outside = tmp_path / "outside"
    if escaped == "checkout":
        checkout = outside
    else:
        build = outside

    completed = _run_build_script(data=data, checkout=checkout, build=build)

    assert completed.returncode != 0
    if escaped == "checkout":
        assert "checkout must remain below the managed cache" in completed.stderr
    else:
        assert "build candidate must remain below managed work/verify" in completed.stderr
    assert not outside.exists()


@pytest.mark.parametrize("attacked", ("checkout", "build"))
def test_build_script_rejects_symlinked_candidate_paths(
    tmp_path: Path,
    attacked: str,
) -> None:
    data = tmp_path / "data"
    paths = LabPaths(repo_root=REPOSITORY_ROOT, data_root=data)
    paths.initialize()
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "marker"
    marker.write_text("unchanged", encoding="utf-8")
    checkout = data / "cache/llama.cpp"
    build = data / "work/verify/candidate"
    if attacked == "checkout":
        checkout.symlink_to(victim, target_is_directory=True)
    else:
        link = data / "work/verify/redirected"
        link.symlink_to(victim, target_is_directory=True)
        build = link / "candidate"

    completed = _run_build_script(data=data, checkout=checkout, build=build)

    assert completed.returncode != 0
    assert "safe real directory" in completed.stderr
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert list(victim.iterdir()) == [marker]


def test_build_script_rejects_shared_library_runtime_lock_before_build(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    lock = tmp_path / "runtime-lock.yaml"
    _write_runtime_lock(
        lock,
        source="https://example.invalid/llama.cpp.git",
        commit="a" * 40,
        binary_sha256="0" * 64,
        shared_libraries=True,
    )

    completed = _run_build_script(
        data=data,
        checkout=data / "cache/llama.cpp",
        build=data / "work/verify/candidate",
        lock=lock,
    )

    assert completed.returncode == 2
    assert "self-contained static build" in completed.stderr
    assert not data.exists()
