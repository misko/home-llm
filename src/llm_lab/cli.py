"""Operator CLI for the model catalog, artifact store, runtime and evals."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import typer

from .attestation import (
    configured_gateway_origin,
    normalize_gateway_origin,
    verify_gateway_attestation,
)
from .benchmark import BenchmarkRunner
from .catalog import Catalog
from .errors import LabError
from .hashing import canonical_sha256, sha256_file
from .paths import LabPaths
from .registry import Registry
from .results import ResultsStore, repair_run_bundle_permissions
from .runtime import RuntimeManager, origin_url, verify_runtime_lock
from .schema import ArtifactManifest, ArtifactSpec
from .storage import ArtifactStore, artifact_path_matches


app = typer.Typer(
    name="llmctl",
    help="Reproducible local LLM storage, serving, and benchmarking.",
    no_args_is_help=True,
)
catalog_app = typer.Typer(help="Validate and inspect the declarative catalog.")
artifact_app = typer.Typer(help="Download, promote, verify, and list artifacts.")
serve_app = typer.Typer(help="Manage the exclusive single-GPU model runtime.")
benchmark_app = typer.Typer(help="Run and compare reproducible benchmark suites.")
registry_app = typer.Typer(help="Inspect and change human-friendly artifact aliases.")
storage_app = typer.Typer(help="Inspect capacity and safely collect orphaned blobs.")

app.add_typer(catalog_app, name="catalog")
app.add_typer(artifact_app, name="artifact")
app.add_typer(serve_app, name="serve")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(registry_app, name="registry")
app.add_typer(storage_app, name="storage")


@dataclass(slots=True)
class CliState:
    paths: LabPaths
    json_output: bool = False
    _catalog: Catalog | None = None

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            self._catalog = Catalog.load(self.paths.catalog_root)
        return self._catalog


def _state(ctx: typer.Context) -> CliState:
    value = ctx.find_root().obj
    if not isinstance(value, CliState):  # pragma: no cover - Typer owns this path
        raise RuntimeError("CLI context was not initialized")
    return value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    return value


def _emit(ctx: typer.Context, value: Any, *, summary: str | None = None) -> None:
    state = _state(ctx)
    if state.json_output or summary is None:
        typer.echo(json.dumps(_jsonable(value), indent=2, sort_keys=True))
    else:
        typer.echo(summary)


def _fail(exc: Exception) -> None:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1) from exc


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024 or unit == "PiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PiB"  # pragma: no cover


def _assert_artifact_matches_catalog(
    spec: ArtifactSpec, manifest: ArtifactManifest
) -> None:
    """Reject a stale registered artifact before it can be served or measured."""

    mismatches: list[str] = []
    for field in ("artifact_id", "model_id", "source", "format", "quantization"):
        expected = spec.id if field == "artifact_id" else getattr(spec, field)
        if getattr(manifest, field) != expected:
            mismatches.append(field)
    if spec.effective_bpw != manifest.effective_bpw:
        mismatches.append("effective_bpw")
    if (
        spec.expected_size_bytes is not None
        and manifest.total_logical_bytes != spec.expected_size_bytes
    ):
        mismatches.append("expected_size_bytes")
    if len(spec.source.revision) >= 40 and all(
        character in "0123456789abcdef" for character in spec.source.revision
    ) and manifest.resolved_revision != spec.source.revision:
        mismatches.append("resolved_revision")

    for locked in manifest.files:
        selectors = [
            selector
            for selector in spec.files
            if artifact_path_matches(locked.logical_path, selector.pattern)
        ]
        if not selectors or all(selector.role != locked.role for selector in selectors):
            mismatches.append(f"file:{locked.logical_path}")
    for selector in spec.files:
        if selector.required and not any(
            artifact_path_matches(locked.logical_path, selector.pattern)
            and locked.role == selector.role
            for locked in manifest.files
        ):
            mismatches.append(f"selector:{selector.pattern}")

    if mismatches:
        rendered = ", ".join(sorted(set(mismatches)))
        raise LabError(
            f"registered artifact {spec.id!r} no longer matches its catalog spec: "
            f"{rendered}; pull/promote a new artifact ID"
        )


def _safe_bundle_path(paths: LabPaths, run_id: str) -> Path:
    runs_root = (paths.data_root / "runs").resolve()
    candidate = (runs_root / run_id).resolve()
    if candidate.parent != runs_root:
        raise LabError(f"unsafe benchmark run path for run id {run_id!r}")
    return candidate


async def _attest_benchmark_endpoint(
    endpoint: str,
    *,
    paths: LabPaths,
    deployment_id: str,
    public_alias: str,
    active_base_url: str,
) -> dict[str, Any]:
    if endpoint.rstrip("/") == active_base_url.rstrip("/"):
        return {
            "kind": "direct_active_backend",
            "effective_base_url": endpoint,
            "deployment_id": deployment_id,
        }

    endpoint_origin = normalize_gateway_origin(origin_url(endpoint))
    expected_origin = configured_gateway_origin()
    if endpoint_origin != expected_origin:
        raise LabError(
            f"--base-url origin {endpoint_origin!r} is not the configured LLM "
            f"Lab gateway {expected_origin!r}"
        )
    probe_url = f"{endpoint_origin}/health"
    challenge = secrets.token_hex(32)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(probe_url, params={"challenge": challenge})
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise LabError(
            f"--base-url is not an attested LLM Lab gateway: {probe_url}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise LabError("--base-url gateway health response is not an object")
    try:
        attestation = verify_gateway_attestation(
            paths,
            payload.get("attestation"),
            challenge=challenge,
            origin=endpoint_origin,
            deployment=deployment_id,
            model=public_alias,
        )
    except LabError as exc:
        raise LabError(
            f"--base-url gateway attestation failed: {exc}"
        ) from exc
    return {
        "kind": "verified_llm_lab_gateway",
        "effective_base_url": endpoint,
        "health_url": probe_url,
        "deployment_id": deployment_id,
        "public_alias": public_alias,
        "challenge_response": attestation,
    }


@app.callback()
def root(
    ctx: typer.Context,
    repo: Path | None = typer.Option(
        None, "--repo", envvar="LLM_LAB_REPO", help="Git control-plane root."
    ),
    data: Path | None = typer.Option(
        None, "--data", envvar="LLM_LAB_DATA", help="Large data-plane root."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON for every command."
    ),
) -> None:
    ctx.obj = CliState(LabPaths.discover(repo, data), json_output=json_output)


@app.command("init")
def initialize(ctx: typer.Context) -> None:
    """Initialize data directories and both metadata databases."""

    state = _state(ctx)
    try:
        state.catalog
        state.paths.initialize()
        with Registry(state.paths.registry_path):
            pass
        with ResultsStore(state.paths.results_db_path):
            pass
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    payload = {
        "status": "initialized",
        "repo_root": state.paths.repo_root,
        "data_root": state.paths.data_root,
        "catalog": state.paths.catalog_root,
        "registry": state.paths.registry_path,
        "results": state.paths.results_db_path,
    }
    _emit(ctx, payload, summary=f"Initialized LLM Lab at {state.paths.data_root}")


@catalog_app.command("validate")
def catalog_validate(ctx: typer.Context) -> None:
    """Load strict schemas and verify all cross-references."""

    try:
        loaded = _state(ctx).catalog
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    counts = {
        "models": len(loaded.models),
        "artifacts": len(loaded.artifacts),
        "deployments": len(loaded.deployments),
        "suites": len(loaded.suites),
        "runtime_locks": len(loaded.runtime_locks),
    }
    _emit(ctx, {"valid": True, "counts": counts}, summary=f"Catalog valid: {counts}")


@catalog_app.command("list")
def catalog_list(
    ctx: typer.Context,
    kind: str = typer.Argument(
        "all",
        help="One of all, models, artifacts, deployments, suites, runtime_locks.",
    ),
) -> None:
    try:
        loaded = _state(ctx).catalog
        collections = {
            "models": loaded.models,
            "artifacts": loaded.artifacts,
            "deployments": loaded.deployments,
            "suites": loaded.suites,
            "runtime_locks": loaded.runtime_locks,
        }
        if kind != "all" and kind not in collections:
            raise ValueError(f"unknown catalog kind {kind!r}")
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    selected = collections if kind == "all" else {kind: collections[kind]}
    payload = {
        name: [_jsonable(item) for item in values.values()]
        for name, values in selected.items()
    }
    _emit(ctx, payload)


@artifact_app.command("pull")
def artifact_pull(
    ctx: typer.Context,
    artifact_id: str,
    alias: str | None = typer.Option(None, help="Optional registry alias to update."),
    offline: bool = typer.Option(False, help="Use only an already-populated HF cache."),
    reserve_gb: float = typer.Option(
        540.0, min=0.0, help="Free-space reserve that ingestion may not consume."
    ),
) -> None:
    """Download selected files at the pinned commit and promote them to CAS."""

    state = _state(ctx)
    try:
        spec = state.catalog.get_artifact(artifact_id)
        with ArtifactStore(
            state.paths, free_reserve_bytes=int(reserve_gb * 1_000_000_000)
        ) as store:
            result = store.pull_and_promote(
                spec, alias=alias, local_files_only=offline
            )
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(
        ctx,
        result,
        summary=(
            f"Installed {artifact_id}: {_format_bytes(result.manifest.total_logical_bytes)}; "
            f"{result.new_blob_count} new blob(s), {result.reused_blob_count} reused"
        ),
    )


@artifact_app.command("promote")
def artifact_promote(
    ctx: typer.Context,
    artifact_id: str,
    source_tree: Path,
    resolved_revision: str | None = typer.Option(None),
    alias: str | None = typer.Option(None),
    enforce_expected_size: bool = typer.Option(True, "--enforce-size/--allow-size-drift"),
) -> None:
    """Promote an already-resolved local tree into the immutable CAS."""

    state = _state(ctx)
    try:
        spec = state.catalog.get_artifact(artifact_id)
        with ArtifactStore(state.paths) as store:
            result = store.promote(
                spec,
                source_tree,
                resolved_revision,
                alias=alias,
                enforce_expected_size=enforce_expected_size,
            )
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(ctx, result, summary=f"Promoted {artifact_id} to {result.view_path}")


@artifact_app.command("verify")
def artifact_verify(
    ctx: typer.Context,
    artifact_id: str,
    verify_view: bool = typer.Option(True, "--view/--no-view"),
) -> None:
    """Re-hash the immutable manifest, blobs, and loader-ready view."""

    state = _state(ctx)
    try:
        with ArtifactStore(state.paths) as store:
            report = store.verify(artifact_id, verify_view=verify_view)
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(
        ctx,
        report,
        summary=(
            f"Verified {artifact_id}: {report.file_count} files, "
            f"{_format_bytes(report.total_logical_bytes)}"
        ),
    )


@artifact_app.command("list")
def artifact_list(ctx: typer.Context) -> None:
    state = _state(ctx)
    try:
        state.paths.initialize()
        with Registry(state.paths.registry_path) as registry:
            artifacts = registry.list_artifacts()
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(ctx, {"artifacts": artifacts, "count": len(artifacts)})


@artifact_app.command("repair-view")
def artifact_repair_view(ctx: typer.Context, artifact_id: str) -> None:
    """Freeze a byte-identical legacy view after verifying its CAS manifest."""

    state = _state(ctx)
    try:
        with ArtifactStore(state.paths) as store:
            report = store.repair_view_permissions(artifact_id)
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(
        ctx,
        report,
        summary=f"Repaired and verified immutable view for {artifact_id}",
    )


@serve_app.command("activate")
def serve_activate(ctx: typer.Context, deployment_id: str) -> None:
    """Health-gate a model activation; restore the previous model on failure."""

    state = _state(ctx)
    try:
        deployment = state.catalog.get_deployment(deployment_id)
        artifact_spec = state.catalog.get_artifact(deployment.artifact_id)
        with ArtifactStore(state.paths) as store:
            manifest = store.registry.get_artifact(deployment.artifact_id)
            _assert_artifact_matches_catalog(artifact_spec, manifest)
            store.verify(manifest, verify_view=True)
        artifact_path = state.paths.view_root / deployment.artifact_id
        runtime_lock = (
            None
            if deployment.runtime_lock_id is None
            else state.catalog.get_runtime_lock(deployment.runtime_lock_id)
        )
        active = RuntimeManager(state.paths).activate(
            deployment, artifact_path, runtime_lock=runtime_lock
        )
        with Registry(state.paths.registry_path) as registry:
            registry.record_history(
                "deployment.activated",
                "deployment",
                deployment.id,
                active.to_dict(),
            )
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(
        ctx,
        active,
        summary=f"Ready: {active.public_alias} at {active.base_url}",
    )


@serve_app.command("status")
def serve_status(
    ctx: typer.Context,
    check_health: bool = typer.Option(True, "--health/--no-health"),
) -> None:
    state = _state(ctx)
    try:
        status = RuntimeManager(state.paths).status(check_health=check_health)
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    summary = "No active model"
    if status.state is not None:
        summary = (
            f"{status.state.public_alias}: phase={status.state.phase}, "
            f"running={status.running}, healthy={status.healthy}"
        )
    _emit(ctx, status, summary=summary)


@serve_app.command("stop")
def serve_stop(ctx: typer.Context) -> None:
    state = _state(ctx)
    try:
        stopped = RuntimeManager(state.paths).stop()
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    summary = "No active model" if stopped is None else f"Stopped {stopped.public_alias}"
    _emit(ctx, {"stopped": stopped}, summary=summary)


@serve_app.command("gateway")
def serve_gateway(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(14000, min=1, max=65535),
    api_key: str | None = typer.Option(
        None,
        envvar=["LLM_LAB_GATEWAY_API_KEY", "LLM_LAB_API_KEY"],
    ),
) -> None:
    """Run the stable OpenAI-compatible gateway in the foreground."""

    import uvicorn

    try:
        from .gateway import create_app
    except ImportError as exc:  # pragma: no cover - development-only state
        _fail(exc)
    uvicorn.run(create_app(paths=_state(ctx).paths, api_key=api_key), host=host, port=port)


def _new_run_id(suite_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{suite_id}-{uuid.uuid4().hex[:10]}"


def _control_plane_provenance(paths: LabPaths) -> dict[str, Any]:
    candidates = sorted((paths.repo_root / "src/llm_lab").rglob("*.py"))
    candidates.extend(
        path
        for path in (paths.repo_root / "pyproject.toml", paths.repo_root / "uv.lock")
        if path.is_file()
    )
    files = [
        {
            "path": path.relative_to(paths.repo_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(candidates))
        if path.is_file()
    ]
    commit: str | None = None
    dirty: bool | None = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(paths.repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip() or None
        status = subprocess.run(
            ["git", "-C", str(paths.repo_root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if status.returncode == 0:
            dirty = bool(status.stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    lock_path = paths.repo_root / "uv.lock"
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_tree_sha256": canonical_sha256(files),
        "uv_lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "file_count": len(files),
    }


async def _execute_benchmark(
    state: CliState,
    suite_id: str,
    deployment_id: str | None,
    base_url: str | None,
    collect_telemetry: bool,
    api_key: str | None,
) -> tuple[Any, Path]:
    state.paths.initialize()
    catalog = state.catalog
    runtime_manager = RuntimeManager(state.paths)
    with runtime_manager.benchmark_lease(check_health=True) as active:
        return await _execute_benchmark_locked(
            state,
            suite_id,
            deployment_id,
            base_url,
            collect_telemetry,
            api_key,
            catalog,
            active,
        )


async def _execute_benchmark_locked(
    state: CliState,
    suite_id: str,
    deployment_id: str | None,
    base_url: str | None,
    collect_telemetry: bool,
    api_key: str | None,
    catalog: Catalog,
    active: Any,
) -> tuple[Any, Path]:
    """Execute while the caller holds RuntimeManager's exclusive GPU lease."""

    suite = catalog.get_suite(suite_id)

    if active.state is None or not active.ready:
        raise LabError("no ready deployment; activate one before benchmarking")
    selected_deployment_id = deployment_id or active.state.deployment.id
    deployment = catalog.get_deployment(selected_deployment_id)
    if active.state.deployment.id != deployment.id:
        raise LabError(
            f"deployment {deployment.id!r} is not the ready active deployment"
        )
    if active.state.deployment != deployment:
        raise LabError(
            f"active deployment {deployment.id!r} was launched from a stale "
            "catalog definition; reactivate it before benchmarking"
        )

    endpoint = base_url or active.state.base_url
    endpoint_attestation = await _attest_benchmark_endpoint(
        endpoint,
        paths=state.paths,
        deployment_id=deployment.id,
        public_alias=deployment.public_alias,
        active_base_url=active.state.base_url,
    )
    artifact_spec = catalog.get_artifact(deployment.artifact_id)
    model_spec = catalog.get_model(artifact_spec.model_id)
    with ArtifactStore(state.paths) as store:
        manifest = store.registry.get_artifact(artifact_spec.id)
        _assert_artifact_matches_catalog(artifact_spec, manifest)
        store.verify(manifest, verify_view=True)

    runtime_lock_metadata: dict[str, Any] | None = None
    if deployment.runtime_lock_id is not None:
        runtime_lock = catalog.get_runtime_lock(deployment.runtime_lock_id)
        runtime_verification = verify_runtime_lock(
            state.paths,
            deployment,
            runtime_lock,
            state.paths.view_root / deployment.artifact_id,
        )
        if (
            active.state.launch.resolved_executable != runtime_verification.binary
            or active.state.launch.executable_sha256
            != runtime_verification.binary_sha256
            or active.state.launch.executable_version
            != runtime_verification.executable_version
        ):
            raise LabError(
                "active runtime identity does not match its reviewed runtime lock"
            )
        runtime_lock_metadata = {
            "spec": runtime_lock.model_dump(mode="json"),
            "verification": runtime_verification.to_dict(),
        }

    run_id = _new_run_id(suite.id)
    bundle_path = _safe_bundle_path(state.paths, run_id)
    runtime_metadata = active.state.to_dict()
    runtime_metadata["control_plane"] = _control_plane_provenance(state.paths)
    runtime_metadata["benchmark_endpoint"] = endpoint_attestation
    if runtime_lock_metadata is not None:
        runtime_metadata["runtime_lock"] = runtime_lock_metadata
    runner_api_key = (
        api_key
        if endpoint_attestation.get("kind") == "verified_llm_lab_gateway"
        else None
    )
    runner = BenchmarkRunner(
        endpoint,
        deployment.public_alias,
        api_key=runner_api_key,
    )
    try:
        run = await runner.run(
            suite,
            model=model_spec.model_dump(mode="json"),
            artifact=manifest.model_dump(mode="json"),
            deployment=deployment.model_dump(mode="json"),
            runtime=runtime_metadata,
            available_capabilities=model_spec.capabilities,
            run_id=run_id,
            collect_telemetry=collect_telemetry,
            bundle_dir=bundle_path,
        )
    finally:
        await runner.aclose()

    with ResultsStore(state.paths.results_db_path) as results:
        results.append_bundle(bundle_path)
    with Registry(state.paths.registry_path) as registry:
        registry.register_run(
            run_id,
            suite_id=suite.id,
            deployment_id=deployment.id,
            artifact_id=artifact_spec.id,
            status=run.run["status"],
            metadata={"bundle_path": str(bundle_path), "summary": run.summary},
            started_at=datetime.fromisoformat(run.run["started_at"].replace("Z", "+00:00")),
            finished_at=datetime.fromisoformat(run.run["finished_at"].replace("Z", "+00:00")),
        )
    return run, bundle_path


@benchmark_app.command("run")
def benchmark_run(
    ctx: typer.Context,
    suite_id: str,
    deployment: str | None = typer.Option(None, help="Expected active deployment id."),
    base_url: str | None = typer.Option(
        None, help="Route through an attested LLM Lab gateway endpoint."
    ),
    telemetry: bool = typer.Option(True, "--telemetry/--no-telemetry"),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar=["LLM_LAB_GATEWAY_API_KEY", "LLM_LAB_API_KEY"],
        help="Bearer token for an authenticated LLM Lab gateway.",
    ),
) -> None:
    """Run a catalog suite and atomically index its immutable result bundle."""

    try:
        run, path = asyncio.run(
            _execute_benchmark(
                _state(ctx), suite_id, deployment, base_url, telemetry, api_key
            )
        )
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    payload = {"run": run.run, "summary": run.summary, "bundle_path": path}
    _emit(
        ctx,
        payload,
        summary=(
            f"Run {run.run_id}: status={run.run['status']}, "
            f"pass_rate={run.summary['pass_rate']}, bundle={path}"
        ),
    )


@benchmark_app.command("compare")
def benchmark_compare(ctx: typer.Context, run_ids: list[str]) -> None:
    """Compare identical tasks without averaging over missing metrics."""

    state = _state(ctx)
    try:
        state.paths.initialize()
        with ResultsStore(state.paths.results_db_path) as results:
            comparison = results.compare_runs(run_ids)
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(ctx, comparison)


@benchmark_app.command("repair-bundle")
def benchmark_repair_bundle(ctx: typer.Context, run_id: str) -> None:
    """Seal a verified legacy bundle as read-only without changing its bytes."""

    state = _state(ctx)
    try:
        state.paths.initialize()
        bundle = _safe_bundle_path(state.paths, run_id)
        report = repair_run_bundle_permissions(bundle)
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(ctx, report, summary=f"Repaired immutable benchmark bundle {run_id}")


@registry_app.command("aliases")
def registry_aliases(ctx: typer.Context) -> None:
    state = _state(ctx)
    try:
        state.paths.initialize()
        with Registry(state.paths.registry_path) as registry:
            aliases = registry.list_aliases()
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(ctx, {"aliases": aliases, "count": len(aliases)})


@registry_app.command("set-alias")
def registry_set_alias(
    ctx: typer.Context,
    alias: str,
    artifact_id: str,
    note: str | None = typer.Option(None),
) -> None:
    state = _state(ctx)
    try:
        state.paths.initialize()
        with Registry(state.paths.registry_path) as registry:
            record = registry.set_alias(alias, artifact_id, note=note)
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(ctx, record, summary=f"{alias} -> {artifact_id} (generation {record.generation})")


@registry_app.command("rollback-alias")
def registry_rollback_alias(
    ctx: typer.Context,
    alias: str,
    steps: int = typer.Option(1, min=1),
    note: str | None = typer.Option(None),
) -> None:
    state = _state(ctx)
    try:
        state.paths.initialize()
        with Registry(state.paths.registry_path) as registry:
            record = registry.rollback_alias(alias, steps=steps, note=note)
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    _emit(
        ctx,
        record,
        summary=f"Rolled {alias} back to {record.artifact_id} (generation {record.generation})",
    )


@storage_app.command("report")
def storage_report(ctx: typer.Context) -> None:
    state = _state(ctx)
    state.paths.initialize()
    usage = shutil.disk_usage(state.paths.data_root)
    try:
        with Registry(state.paths.registry_path) as registry:
            manifests = registry.list_artifacts()
        logical = sum(item.total_logical_bytes for item in manifests)
        digests: dict[str, int] = {}
        for manifest in manifests:
            for item in manifest.files:
                digests[item.sha256] = item.size_bytes
        physical_registered = sum(digests.values())
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    payload = {
        "data_root": state.paths.data_root,
        "filesystem": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_human": _format_bytes(usage.free),
        },
        "registered": {
            "artifact_count": len(manifests),
            "logical_bytes": logical,
            "unique_blob_bytes": physical_registered,
            "deduplicated_bytes": logical - physical_registered,
        },
    }
    _emit(ctx, payload)


@storage_app.command("gc")
def storage_gc(
    ctx: typer.Context,
    apply: bool = typer.Option(
        False, "--apply", help="Actually remove candidates; default is a dry run."
    ),
) -> None:
    state = _state(ctx)
    try:
        with ArtifactStore(state.paths) as store:
            report = store.gc(dry_run=not apply)
    except (LabError, OSError, ValueError) as exc:
        _fail(exc)
    verb = "Removed" if apply else "Would remove"
    _emit(
        ctx,
        report,
        summary=f"{verb} {report.candidate_count} orphan blob(s), {_format_bytes(report.candidate_bytes)}",
    )


if __name__ == "__main__":  # pragma: no cover
    app()
