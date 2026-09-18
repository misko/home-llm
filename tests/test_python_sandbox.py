from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

from llm_lab.tooling.python_sandbox import (
    DEFAULT_IMAGE,
    PythonSandboxProvider,
    PythonSandboxSettings,
)


def _docker_image_available() -> bool:
    docker = shutil.which("docker")
    return bool(docker and subprocess.run(
        [docker, "image", "inspect", DEFAULT_IMAGE],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_python_sandbox_is_disposable_and_stages_output() -> None:
    if not _docker_image_available():
        pytest.skip("pinned Python sandbox image is not installed")
    provider = PythonSandboxProvider(PythonSandboxSettings(enabled=True))
    result = await provider.run({
        "code": (
            "from pathlib import Path\n"
            "Path('artifact.txt').write_text('staged output')\n"
            "print(Path('/srv/llm-lab-workspaces').exists())\n"
        ),
    })
    assert result["exit_code"] == 0
    assert result["stdout"] == "False\n"
    assert result["output_files"] == [{
        "path": "artifact.txt",
        "sha256": "ea3a85c34d19fd707a40811e3584f85f51293bb62fbbdf3f5ef1ee23d53c9576",
        "bytes": 13,
    }]
