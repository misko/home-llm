from __future__ import annotations

import json
import socket
import textwrap
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from llm_lab.cli import _safe_bundle_path, app
from llm_lab.errors import LabError
from llm_lab.paths import LabPaths
from llm_lab.results import ResultsStore


runner = CliRunner()


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _write_minimal_catalog(repo: Path, port: int) -> Path:
    catalog = repo / "catalog"
    for directory in ("models", "artifacts", "deployments", "suites"):
        (catalog / directory).mkdir(parents=True, exist_ok=True)
    (catalog / "models/test.yaml").write_text(
        textwrap.dedent(
            """
            schema_version: 1
            id: test-model
            display_name: Test Model
            family: test
            description: Tiny fixture used by the CLI lifecycle test.
            total_params_b: 1
            active_params_b: 1
            native_context: 1024
            modalities: [text]
            capabilities: [chat]
            license: {name: MIT, osi_approved: true}
            upstream:
              provider: local
              local_path: /fixture/source
              revision: fixture-revision
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (catalog / "artifacts/test.yaml").write_text(
        textwrap.dedent(
            """
            schema_version: 1
            id: test-artifact
            model_id: test-model
            source:
              provider: local
              local_path: /fixture/source
              revision: fixture-revision
            format: gguf
            quantization: fixture
            expected_size_bytes: 5
            files:
              - {pattern: weights.gguf, role: weights}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (catalog / "deployments/mock.yaml").write_text(
        textwrap.dedent(
            f"""
            schema_version: 1
            id: test-mock
            artifact_id: test-artifact
            public_alias: local-test
            backend: mock
            executable: python
            host: 127.0.0.1
            port: {port}
            context_size: 1024
            startup_timeout_seconds: 10
            reasoning_mode: "off"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (catalog / "suites/smoke.yaml").write_text(
        textwrap.dedent(
            """
            schema_version: 1
            id: cli-smoke
            version: "1"
            description: CLI integration smoke.
            kind: smoke
            repetitions: 1
            cases:
              - id: echo
                messages:
                  - {role: user, content: hello from the CLI}
                expectations:
                  - {kind: contains, value: hello from the CLI}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return catalog


def _base(repo: Path, data: Path) -> list[str]:
    return ["--repo", str(repo), "--data", str(data), "--json"]


def test_catalog_validate_and_init_json(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    _write_minimal_catalog(repo, _free_port())

    validated = runner.invoke(app, [*_base(repo, data), "catalog", "validate"])
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.stdout) == {
        "valid": True,
        "counts": {
            "models": 1,
            "artifacts": 1,
            "deployments": 1,
            "suites": 1,
            "runtime_locks": 0,
        },
    }

    initialized = runner.invoke(app, [*_base(repo, data), "init"])
    assert initialized.exit_code == 0, initialized.output
    assert json.loads(initialized.stdout)["status"] == "initialized"
    assert (data / "registry/catalog.sqlite").is_file()
    assert (data / "results/results.duckdb").is_file()


def test_run_bundle_path_has_containment_defense(tmp_path: Path) -> None:
    paths = LabPaths.discover(tmp_path / "repo", tmp_path / "data")
    paths.initialize()

    with pytest.raises(LabError, match="unsafe benchmark run path"):
        _safe_bundle_path(paths, "safe/../../../escaped")


def test_metadata_commands_reject_symlinked_managed_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    victim = tmp_path / "victim"
    victim.mkdir()
    data.mkdir()
    (data / "registry").symlink_to(victim, target_is_directory=True)

    result = runner.invoke(app, [*_base(repo, data), "artifact", "list"])

    assert result.exit_code == 1
    assert "safe real directory" in result.output
    assert not (victim / "catalog.sqlite").exists()


def test_init_rejects_final_results_database_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    _write_minimal_catalog(repo, _free_port())
    paths = LabPaths.discover(repo, data)
    paths.initialize()
    victim = tmp_path / "victim.duckdb"
    with ResultsStore(victim):
        pass
    before = victim.read_bytes()
    paths.results_db_path.symlink_to(victim)

    result = runner.invoke(app, [*_base(repo, data), "init"])

    assert result.exit_code == 1
    assert "results database" in result.output
    assert victim.read_bytes() == before


@pytest.mark.integration
def test_cli_artifact_runtime_and_benchmark_lifecycle(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.gguf").write_bytes(b"model")
    port = _free_port()
    _write_minimal_catalog(repo, port)
    base = _base(repo, data)

    assert runner.invoke(app, [*base, "init"]).exit_code == 0
    promoted = runner.invoke(
        app,
        [*base, "artifact", "promote", "test-artifact", str(source), "--alias", "test"],
    )
    assert promoted.exit_code == 0, promoted.output
    assert json.loads(promoted.stdout)["manifest"]["artifact_id"] == "test-artifact"

    verified = runner.invoke(app, [*base, "artifact", "verify", "test-artifact"])
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.stdout)["view_verified"] is True

    activated = runner.invoke(app, [*base, "serve", "activate", "test-mock"])
    assert activated.exit_code == 0, activated.output
    try:
        assert json.loads(activated.stdout)["phase"] == "ready"
        response = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "local-test",
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=5,
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "mock response: hello"

        benchmark = runner.invoke(
            app,
            [*base, "benchmark", "run", "cli-smoke", "--no-telemetry"],
        )
        assert benchmark.exit_code == 0, benchmark.output
        payload = json.loads(benchmark.stdout)
        assert payload["summary"]["pass_rate"] == 1.0
        assert payload["run"]["runtime"]["benchmark_endpoint"] == {
            "kind": "direct_active_backend",
            "effective_base_url": f"http://127.0.0.1:{port}",
            "deployment_id": "test-mock",
        }
        assert Path(payload["bundle_path"], "run.json").is_file()
        assert (data / "results/results.duckdb").is_file()

        unverified = runner.invoke(
            app,
            [
                *base,
                "benchmark",
                "run",
                "cli-smoke",
                "--base-url",
                "http://127.0.0.1:1",
                "--no-telemetry",
            ],
        )
        assert unverified.exit_code == 1
        assert "not the configured LLM Lab gateway" in unverified.output
    finally:
        stopped = runner.invoke(app, [*base, "serve", "stop"])
        assert stopped.exit_code == 0, stopped.output

    status = runner.invoke(app, [*base, "serve", "status"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.stdout)["active"] is False


@pytest.mark.integration
def test_activation_rejects_registered_bytes_after_catalog_identity_changes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.gguf").write_bytes(b"model")
    _write_minimal_catalog(repo, _free_port())
    base = _base(repo, data)
    assert runner.invoke(app, [*base, "init"]).exit_code == 0
    assert runner.invoke(
        app,
        [*base, "artifact", "promote", "test-artifact", str(source)],
    ).exit_code == 0

    artifact_path = repo / "catalog/artifacts/test.yaml"
    artifact_path.write_text(
        artifact_path.read_text(encoding="utf-8").replace(
            "quantization: fixture", "quantization: changed-after-promotion"
        ),
        encoding="utf-8",
    )
    rejected = runner.invoke(app, [*base, "serve", "activate", "test-mock"])

    assert rejected.exit_code == 1
    assert "no longer matches its catalog spec" in rejected.output
    assert not (data / "state/active.json").exists()
