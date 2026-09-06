# Acceptance checklist

Run this checklist from the repository root. The baseline is offline and does
not need a GPU, Docker daemon, or model download. Optional hardware/model checks
are isolated at the end.

## Baseline acceptance

- [ ] The locked development environment installs.

```bash
uv sync --dev --locked
```

- [ ] Source compiles, patches have no whitespace errors, and the complete test
  suite passes.

```bash
uv run python -m compileall -q src
git diff --check
uv run pytest -q
```

- [ ] The current CLI initializes an isolated data plane and exposes the
  catalog/storage read paths. No model data is downloaded.

```bash
uv run llmctl --help >/dev/null
uv run python - <<'PY'
import subprocess
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory(prefix="llm-lab-cli-acceptance-") as temporary:
    common = [
        sys.executable,
        "-m",
        "llm_lab.cli",
        "--repo",
        str(Path.cwd()),
        "--data",
        str(Path(temporary) / "data"),
        "--json",
    ]
    subprocess.run([*common, "init"], check=True, capture_output=True, text=True)
    marker = Path(temporary) / "data" / ".llm-lab-root"
    assert marker.is_file() and not marker.is_symlink()
    subprocess.run(
        [*common, "catalog", "validate"], check=True, capture_output=True, text=True
    )
    subprocess.run(
        [*common, "storage", "report"], check=True, capture_output=True, text=True
    )
print("CLI acceptance: OK")
PY
```

- [ ] Heavy model formats are not tracked by Git.

```bash
test -z "$(git ls-files '*.gguf' '*.safetensors' '*.bin' '*.pth')"
```

- [ ] The checked-in catalog has the expected four-model portfolio and five
  declarative identity categories, pinned revisions, unique deployment aliases,
  one reviewed runtime lock, the exact starter storage total, and no remote-code
  artifact.

```bash
uv run python - <<'PY'
import re
from pathlib import Path

from llm_lab.catalog import Catalog

catalog = Catalog.load(Path("catalog"))
expected_models = {
    "qwen3.8-27b",
    "muse-glimmer-30b",
    "ling-3.0-tiny",
    "devstral-small-2-24b",
}
expected_bytes = 56_893_598_323
capacity = 2_700_000_000_000
reserve = 540_000_000_000
commit = re.compile(r"^[0-9a-f]{40,64}$")

assert set(catalog.models) == expected_models
assert len(catalog.artifacts) == 4
assert len(catalog.deployments) == 6  # four primary, one reasoning, plus mock
assert set(catalog.suites) == {"smoke", "perf-4090"}
assert set(catalog.runtime_locks) == {"llama-cpp-cuda-4090"}
assert all(commit.fullmatch(model.upstream.revision) for model in catalog.models.values())
assert all(commit.fullmatch(item.source.revision) for item in catalog.artifacts.values())
assert all(not item.requires_remote_code for item in catalog.artifacts.values())
assert len({item.public_alias for item in catalog.deployments.values()}) == 6
assert all(
    item.runtime_lock_id == "llama-cpp-cuda-4090"
    for item in catalog.deployments.values()
    if item.id != "mock-canary"
)
runtime_lock = catalog.get_runtime_lock("llama-cpp-cuda-4090")
assert commit.fullmatch(runtime_lock.commit)
assert re.fullmatch(r"[0-9a-f]{64}", runtime_lock.binary_sha256)
assert sum(item.expected_size_bytes or 0 for item in catalog.artifacts.values()) == expected_bytes
assert expected_bytes * 2 + reserve < capacity
print("catalog acceptance: OK")
PY
```

- [ ] The offline end-to-end harness passes. It writes only to a fresh temporary
  directory and removes it on exit.

```bash
uv run python - <<'PY'
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import httpx

from llm_lab.benchmark import BenchmarkRunner
from llm_lab.catalog import Catalog
from llm_lab.hashing import canonical_sha256
from llm_lab.paths import DATA_DIRECTORIES, DATA_ROOT_SENTINEL, LabPaths
from llm_lab.results import ResultsStore, verify_run_bundle
from llm_lab.runtime import build_backend_command
from llm_lab.schema import (
    ArtifactFileSelector,
    ArtifactFormat,
    ArtifactSpec,
    BenchmarkCase,
    BenchmarkSuite,
    ChatMessage,
    Expectation,
    FileRole,
    RepositorySource,
)
from llm_lab.storage import ArtifactStore
from llm_lab.telemetry import NvidiaTelemetrySampler


async def benchmark_check(paths: LabPaths) -> None:
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "model": "local-canary",
                "choices": [{"message": {"role": "assistant", "content": "READY"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    suite = BenchmarkSuite(
        id="acceptance",
        version="1.0.0",
        description="offline protocol acceptance",
        kind="smoke",
        cases=(
            BenchmarkCase(
                id="chat",
                messages=(ChatMessage(role="user", content="Reply READY"),),
                expectations=(Expectation(kind="exact", value="READY"),),
            ),
            BenchmarkCase(
                id="tool-capability",
                messages=(ChatMessage(role="user", content="Capability gate"),),
                required_capabilities=("tools",),
                expectations=(Expectation(kind="nonempty"),),
            ),
        ),
    )
    transport = httpx.MockTransport(handler)
    async with BenchmarkRunner(
        "https://acceptance.invalid",
        served_model="local-canary",
        transport=transport,
    ) as runner:
        full = await runner.run(
            suite,
            model={"id": "fixture-model", "capabilities": ["chat", "tools"]},
            artifact="fixture-a",
            deployment="mock-canary",
            available_capabilities=("chat", "tools"),
            collect_telemetry=False,
            run_id="acceptance-full",
            bundle_dir=paths.data_root / "runs" / "acceptance-full",
        )
        partial = await runner.run(
            suite,
            model={"id": "fixture-model", "capabilities": ["chat"]},
            artifact="fixture-a",
            deployment="mock-canary",
            available_capabilities=("chat",),
            collect_telemetry=False,
            run_id="acceptance-partial",
            bundle_dir=paths.data_root / "runs" / "acceptance-partial",
        )

    assert calls == 3
    assert full.summary["sample_count"] == 2
    assert full.summary["error_count"] == 0
    assert partial.summary["sample_count"] == 1
    assert partial.summary["pass_rate"] is None
    assert partial.summary["skipped_cases"][0]["case_id"] == "tool-capability"

    full_path = paths.data_root / "runs" / "acceptance-full"
    partial_path = paths.data_root / "runs" / "acceptance-partial"
    assert verify_run_bundle(full_path).sample_count == 2
    assert verify_run_bundle(partial_path).sample_count == 1
    run_document = json.loads((full_path / "run.json").read_text())
    assert run_document["suite"]["definition"]["id"] == "acceptance"
    assert run_document["suite"]["sha256"] == canonical_sha256(
        run_document["suite"]["definition"]
    )
    summary_document = json.loads((full_path / "summary.json").read_text())
    assert summary_document["performance"][
        "client_completion_tokens_per_second"
    ]["count"] == 2
    with ResultsStore(paths.results_db_path) as results:
        results.append_bundle(full_path)
        results.append_bundle(partial_path)
        comparison = results.compare_runs(("acceptance-full", "acceptance-partial"))
        assert comparison["composite"] is None
        assert "same task set" in comparison["composite_unavailable_reason"]


with tempfile.TemporaryDirectory(prefix="llm-lab-acceptance-") as temporary:
    root = Path(temporary)
    paths = LabPaths(repo_root=Path.cwd(), data_root=root / "data")
    paths.initialize()
    marker = paths.data_root / DATA_ROOT_SENTINEL
    assert marker.is_file() and not marker.is_symlink()
    assert all((paths.data_root / relative).is_dir() for relative in DATA_DIRECTORIES)

    source_a = root / "source-a"
    source_b = root / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "model.gguf").write_bytes(b"fixture artifact A\n")
    (source_b / "model.gguf").write_bytes(b"fixture artifact B\n")

    def fixture_artifact(identifier: str, source: Path) -> ArtifactSpec:
        weight = source / "model.gguf"
        return ArtifactSpec(
            id=identifier,
            model_id="fixture-model",
            source=RepositorySource(provider="local", local_path=str(source)),
            format=ArtifactFormat.GGUF,
            quantization="fixture",
            expected_size_bytes=weight.stat().st_size,
            files=(ArtifactFileSelector(pattern="model.gguf", role=FileRole.WEIGHTS),),
        )

    with ArtifactStore(paths) as store:
        first = store.promote(fixture_artifact("fixture-a", source_a), source_a, "fixture-a", alias="candidate")
        second = store.promote(fixture_artifact("fixture-b", source_b), source_b, "fixture-b", alias="candidate")
        assert first.new_blob_count == 1
        assert second.new_blob_count == 1
        assert store.verify("fixture-a").view_verified
        assert store.verify("fixture-b").view_verified
        assert store.registry.resolve_alias("candidate") == "fixture-b"
        restored = store.registry.rollback_alias("candidate", note="acceptance rollback")
        assert restored.artifact_id == "fixture-a"
        assert store.gc().candidate_count == 0

    catalog = Catalog.load(Path("catalog"))
    plan = build_backend_command(catalog.get_deployment("mock-canary"), paths=paths)
    assert plan.kind == "process"
    assert plan.base_url == "http://127.0.0.1:18089"
    assert "local-canary" in plan.command

    unavailable = NvidiaTelemetrySampler(
        command_runner=lambda _: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi"))
    ).sample_once()
    assert len(unavailable) == 1 and not unavailable[0].available

    asyncio.run(benchmark_check(paths))

print("offline end-to-end acceptance: OK")
PY
```

## Optional RTX 4090 and container acceptance

These checks are environment-specific and are not part of the offline baseline.

- [ ] The expected GPU and driver are visible.

```bash
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,compute_cap \
  --format=csv,noheader
```

- [ ] Docker and NVIDIA Container Toolkit can expose the GPU to a container.
Choose a reviewed CUDA image digest appropriate for the installed driver; do
not copy an unreviewed floating tag into automation.

```bash
docker version
docker info --format '{{json .Runtimes}}'
```

- [ ] The pinned llama.cpp host build completes, if host serving is part of the
  accepted deployment path. This is a network and compilation operation.

```bash
scripts/build_llama_cpp.sh
```

## Optional real-model acceptance

Complete these steps separately for each deployed artifact after reviewing
license terms and capacity. The exact current Python procedures are in
[Operations](operations.md).

- [ ] `llmctl artifact pull <artifact> --reserve-gb 540` pulls and promotes the
  pinned artifact.
- [ ] `llmctl artifact verify <artifact>` validates manifest, CAS blobs, and view.
- [ ] The runtime image has a reviewed digest, or the pinned host runtime lock's
  binary path, SHA-256, and version evidence verify.
- [ ] `llmctl serve activate <deployment>` reaches the cataloged health endpoint
  under the GPU lock.
- [ ] `POST /v1/chat/completions` accepts the deployment `public_alias`.
- [ ] `llmctl benchmark run smoke` creates a verified four-file bundle with zero
  API errors.
- [ ] `perf-4090@1.1.0` is compared only with runs having identical suite and
  effective-request-contract SHA-256 values, complete task coverage, and zero
  measured or warmup errors; reported client performance is understood as
  end-to-end latency/completion throughput, not TTFT.
- [ ] A deliberately invalid candidate in a controlled test fails readiness and
  restores the prior deployment.
- [ ] Stop clears active state, and a second stop is harmless.

## Acceptance evidence

For a release or workstation handoff, retain:

- repository commit and `uv.lock` hash;
- output of the baseline commands;
- catalog diff and license review;
- artifact manifest hashes and verification reports;
- runtime image digest or host-runtime lock plus verified binary SHA-256/version;
- GPU/driver identity;
- verified smoke and performance run bundles; and
- per-task comparison output, including any suppressed-composite reason.
