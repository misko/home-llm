# Operations

This runbook covers a single Linux workstation with one NVIDIA GPU. The shown
`llmctl` commands are present in the current CLI; Python examples expose lower-
level interfaces where an operator needs finer control or inspection.

## Command status

The implemented top-level groups are `catalog`, `artifact`, `serve`,
`benchmark`, `registry`, and `storage`; `init` initializes the data plane and
both databases. Always use the installed version's help as authority:

```bash
uv run llmctl --help
uv run llmctl artifact --help
```

The following capabilities are **Planned**, with no promised CLI syntax yet:
model conversion, a global 2.7 TB quota daemon, Hub/Xet or Docker-cache pruning,
alias-history display, quarantine workflow, and results export. Use the current
Python APIs or documented external tools where applicable; do not invent
`llmctl` commands for them.

## 1. Install and choose paths

Required:

- Python 3.11 or newer;
- `uv`;
- enough local filesystem capacity for the 2.7 TB data-plane policy; and
- for real GPU serving, a compatible NVIDIA driver plus either Docker with
  NVIDIA Container Toolkit or a supported host backend executable.

From the repository root:

```bash
uv sync --dev --locked
uv run pytest -q
uv run llmctl --help
```

Select an absolute data path. Avoid a home directory, the Git checkout, a
filesystem root, or a path shared with unrelated data. The example path is only
a placeholder and must be adapted to the workstation.

```bash
export LLM_LAB_REPO="$PWD"
export LLM_LAB_DATA="/srv/llm-lab-data"
```

Initialize the data-plane directories, validate the catalog, and create both
metadata databases:

```bash
uv run llmctl init
uv run llmctl --json storage report
```

Initialization claims the directory by creating the private regular-file
sentinel `.llm-lab-root`. On first use, the directory must be empty or contain
only recognized LLM Lab top-level directories. The data root and every managed
directory component must be real directories, not symbolic links; initialization
also rejects a linked, hard-linked, or modified sentinel. Do not replace
`blobs/`, `manifests/`, `state/`, `logs/`, or another managed path with a link.

To inspect the resolved locations directly:

```bash
uv run python - <<'PY'
from llm_lab.paths import LabPaths

paths = LabPaths.discover()
paths.initialize()
print("repo:", paths.repo_root)
print("data:", paths.data_root)
print("registry:", paths.registry_path)
print("results:", paths.results_db_path)
PY
```

`LLM_LAB_DATA` is not automatically a dedicated mount. Confirm that the
resolved path is on the intended filesystem before downloading:

```bash
findmnt --target "$LLM_LAB_DATA"
df -h "$LLM_LAB_DATA"
du -sx --block-size=1 "$LLM_LAB_DATA"
```

## 2. Validate the control plane

Catalog loading rejects unknown fields, duplicate YAML keys, duplicate IDs,
missing references, duplicate public aliases, host deployments without a
runtime lock, container deployments without an immutable image digest, and
other schema violations.

```bash
uv run llmctl catalog validate
uv run llmctl --json catalog list
```

The checked-in catalog currently validates as four models, four artifacts, six
deployments (five real profiles and the mock canary), two suites, and one
runtime lock. `catalog list runtime_locks` exposes the reviewed lock record.

Validation proves internal consistency, not that a remote repository remains
available or that its license terms have been accepted. Inspect every model's
license and every artifact's source before acquisition.

## 3. Enforce the 2.7 TB policy

Use decimal bytes for admission decisions:

- operating envelope: `2_700_000_000_000` bytes;
- warning watermark: `1_890_000_000_000` bytes (70%);
- registry-aware retention/GC watermark: `2_025_000_000_000` bytes (75%);
- ordinary-ingestion ceiling: `2_160_000_000_000` bytes (80%); and
- free reserve: `540_000_000_000` bytes (20%).

Before a pull, budget at least twice `expected_size_bytes` for separate upstream
and CAS copies, plus any conversion scratch. CAS promotion copies by default so
the acquisition cache and immutable bytes cannot share a mutable inode. The
lower-level hard-link mode is explicit opt-in and must not be assumed by
admission calculations.

This read-only preflight prints the current starter-set estimate and filesystem
headroom:

```bash
uv run python - <<'PY'
import shutil
from llm_lab.catalog import Catalog
from llm_lab.paths import LabPaths

CAPACITY = 2_700_000_000_000
CEILING = 2_160_000_000_000
RESERVE = 540_000_000_000

paths = LabPaths.discover()
catalog = Catalog.load(paths.catalog_root)
selected = sum(item.expected_size_bytes or 0 for item in catalog.artifacts.values())
free = shutil.disk_usage(paths.data_root).free
print(f"selected artifact bytes: {selected:,}")
print(f"two-tier estimate:       {selected * 2:,}")
print(f"filesystem free:         {free:,}")
print(f"policy capacity:         {CAPACITY:,}")
print(f"managed-use ceiling:     {CEILING:,}")
if free < selected * 2 + RESERVE:
    raise SystemExit("insufficient free space for starter set plus reserve")
PY
```

This free-space check complements, but does not calculate, physical use below
the data root. Use `du` (which understands hard links on the local filesystem)
and reject an acquisition if projected managed use crosses the ceiling.

## 4. Pull and promote an artifact

This step performs a large network transfer. Run one artifact at a time. The
example uses the smallest starter artifact; change the ID only after reviewing
its source, expected size, and license.

```bash
uv run llmctl artifact pull ling-3.0-tiny-q4-k-m --reserve-gb 540
uv run llmctl artifact list
```

The CLI's default reserve is 540 GB, matching this repository's 2.7 TB policy.
`artifact promote` handles an already-resolved local
tree but does not currently expose the reserve setting, so complete the manual
capacity preflight before using it.

Authentication, when required, should use Hugging Face's supported token
mechanism or a process environment secret. Do not put tokens in catalog YAML,
shell history, run metadata, or committed files.

Promotion is safe to retry with identical bytes. Reusing an artifact ID for
different bytes is rejected. Required file selectors, exact selected size,
source stability during hashing, CAS hashes, tree hash, and manifest hash are
checked before publication.

## 5. Verify before serving

Always verify the manifest, every CAS blob, and the materialized view after a
download, before a deployment change, and after suspected disk or filesystem
problems:

```bash
uv run llmctl artifact verify ling-3.0-tiny-q4-k-m
```

Do not “fix” a mismatched blob in place. Stop affected deployments, preserve the
error and logs, isolate the exact bad path if incident policy requires it, then
rehydrate through the pinned source and promotion flow.

## 6. Lock the runtime

The checked-in `catalog/runtime-locks/llama-cpp-cuda.yaml` is the authoritative
`RuntimeLockSpec`. It pins the llama.cpp source and Git commit, build recipe and
CUDA architecture, installed binary path, binary SHA-256, and required version
evidence. The reproducible host build helper is:

```bash
scripts/build_llama_cpp.sh
```

It clones/fetches source and builds CUDA code, so it is a deliberate network and
compute operation. The script reads its source, commit, architecture, output
path, and expected hash from the lock; it refuses to switch a modified checkout
or accept disagreeing overrides. `LLM_LAB_ALLOW_UNLOCKED_BUILD=1` exists only
for an explicitly unlocked experiment; any output whose installed path, hash,
or version differs from the lock is refused by a locked activation.

All five real checked-in deployment profiles use this lock and the pinned host
binary at `${LLM_LAB_DATA}/cache/runtimes/llama-cpp-cuda-4090/llama-server`. Before an
activation can replace the GPU owner, LLM Lab verifies that exact path, hashes
the no-follow regular file, and checks that `--version` contains the lock's
required evidence. The launch record captures the resolved executable,
SHA-256, and complete version output and must still match the pre-launch
verification. Benchmarking re-verifies the lock and active launch identity.

Container profiles are also supported, but a mutable image tag is not a
complete runtime identity: use an embedded digest or populate a matching
`RuntimeImage.digest` in a reviewed deployment revision.

Check the local GPU/runtime before activation:

```bash
nvidia-smi
# Only when using a container deployment:
docker version && docker info --format '{{json .Runtimes}}'
```

## 7. Activate and probe a deployment

`RuntimeManager` allows one active deployment. Activation stops the previous
backend, starts the candidate, waits for health, and automatically attempts to
restore the previous resolved launch if the candidate fails readiness.

After the artifact in this example has been promoted:

```bash
uv run llmctl serve activate ling-3.0-tiny-4090-8k
uv run llmctl --json serve status
```

Start the stable gateway in a second terminal. Supplying a key enables bearer
authentication for all `/v1/*` routes:

```bash
LLM_LAB_API_KEY="replace-with-a-secret" \
  uv run llmctl serve gateway --host 127.0.0.1 --port 14000
```

Probe the gateway through the cataloged alias:

```bash
curl --fail-with-body http://127.0.0.1:14000/health
curl --fail-with-body http://127.0.0.1:14000/v1/chat/completions \
  -H 'Authorization: Bearer replace-with-a-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-fast",
    "messages": [{"role": "user", "content": "Reply with READY."}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

The gateway also passes through `/v1/models`, `/v1/completions`, and
`/v1/responses`, including streaming SSE. Its `/health` route is intentionally
unauthenticated. The default ports bind locally and the gateway does not provide
TLS; do not use a routable host without a reviewed TLS proxy and firewall
policy. Direct backend ports bypass gateway authentication.

### Browser console

The same single-worker gateway serves the compiled operator console at
`http://127.0.0.1:14000/ui/`. Keep it on loopback and use an SSH tunnel for a
remote browser:

```bash
ssh -N -L 14000:127.0.0.1:14000 user@kalman
```

Then open `http://127.0.0.1:14000/ui/` locally. Use exactly one gateway worker;
durable console operations are serialized in-process and multi-worker serving
is not supported. Do not expose the Vite development server or unauthenticated
control API outside a trusted, firewalled network. When gateway bearer auth is
enabled, use the CLI/API until an authenticated browser session proxy is in
place; the console has no token-entry flow.

The Models page verifies the selected catalog deployment before switching and
shows durable operation progress. Closing a page does not cancel admitted work.
On activation failure, the runtime manager attempts to restore the prior ready
deployment. Inspect `llmctl serve status` and the operator journal if status
polling is interrupted.

## 8. Run and index a benchmark

The benchmark runner records the five applicable declarative identities,
runtime, endpoint attestation, hardware, sampling, latency, usage, errors,
scorer details, and best-effort GPU telemetry. The following runs
`smoke@1.0.0` against the active deployment and creates a new write-once bundle:

```bash
uv run llmctl benchmark run smoke
```

The command requires a ready active deployment whose state exactly matches the
current catalog. It reconciles the registered artifact with its catalog spec,
verifies the manifest/CAS/view, and re-verifies the runtime lock before writing
the bundle under `runs/<run-id>/`, verifying/appending it to DuckDB, and
recording the run in SQLite. Use `--no-telemetry` only when GPU telemetry is
intentionally outside the measurement contract.

The active backend endpoint is accepted directly. A `--base-url` override is
accepted only if its unauthenticated `/health` endpoint proves that it is an
LLM Lab gateway routing the same ready deployment ID and public alias. An
arbitrary, merely reviewed URL is rejected. The exact effective endpoint and
attestation are retained in run metadata.

Every run embeds the complete validated suite definition and its canonical
SHA-256. Case and runner request extensions may add tools, response formats, or
similar fields, but cannot replace `model`, `messages`, `temperature`, `top_p`,
`max_tokens`, `seed`, or `stream`. A response that explicitly reports a
different model name is an invalid response.

`latency_ms` is full end-to-end non-streaming request latency.
`client_completion_tokens_per_second` is completion tokens divided by that
elapsed time. Backend-provided prompt and predicted/decode timings are stored
separately when present. None of these fields is TTFT; this runner does not
measure time to first token.

`nvidia-smi` absence is recorded in `telemetry.jsonl`; it does not fail the run.
An API failure is recorded per sample and the suite continues. Treat
`completed_with_errors` as a failed operational gate even if another sample
passed.

Compare two indexed runs per task:

```bash
uv run llmctl --json benchmark compare RUN_ID_A RUN_ID_B
```

A missing task, changed suite content digest, changed suite ID/version, legacy
run without a suite digest, or unscored task suppresses the composite. Use the
per-case rows to explain regressions; never rank models by averaging different
task sets.

## 9. Stop or roll back

Graceful stop:

```bash
uv run llmctl serve stop
```

Runtime rollback is automatic only when a candidate activation fails. A healthy
but semantically bad candidate should be replaced by explicitly activating the
previous deployment after reviewing its artifact verification and launch plan.

Registry aliases have independent audited rollback. Current aliases are visible
through the CLI; history inspection remains a Python-API operation:

```bash
uv run llmctl registry aliases
```

Inspect history before rolling back:

```bash
uv run python - <<'PY'
from llm_lab.paths import LabPaths
from llm_lab.registry import Registry

alias = "candidate"
paths = LabPaths.discover()
with Registry(paths.registry_path) as registry:
    for event in registry.list_alias_history(alias):
        print(event)
PY
```

Then perform the reviewed mutation:

```bash
uv run llmctl registry rollback-alias candidate \
  --steps 1 --note "operator rollback"
```

This moves a pointer and appends history. It does not alter CAS bytes or the
client-facing deployment alias.

## 10. Garbage collection and capacity recovery

CAS garbage collection is report-only unless `dry_run=False` is explicit:

```bash
uv run llmctl storage report
uv run llmctl storage gc
```

Review every candidate and back up required evidence before destructive GC:

```bash
# Deliberate destructive action after reviewing the dry-run output:
uv run llmctl storage gc --apply
```

Registered manifests and valid on-disk manifests remain roots, so ordinary live
artifacts are not candidates. CAS GC does not clean Docker images, the Hugging
Face cache, build caches, work directories, logs, runs, or DuckDB. Those require
separate retention decisions. Never run broad recursive deletion against the
data root.

## Backup and recovery priorities

Back up, in order:

1. the Git control plane and exact commit;
2. `manifests/` and `registry/catalog.sqlite` (including SQLite WAL/SHM during a
   consistent backup);
3. immutable `runs/` bundles;
4. `results/results.duckdb` (rebuildable from retained bundles);
5. CAS blobs that cannot be reliably rehydrated from pinned upstream commits;
6. acquisition/build caches only if their download/build cost justifies it.

`state/active.json` is operational state, not a substitute for a deployment
manifest. After host recovery, verify artifacts and explicitly activate a
deployment rather than blindly restoring a stale PID/container record.

## Routine checklist

Before each rollout:

1. validate catalog and license/provenance;
2. confirm capacity, reserve, and conversion scratch;
3. pull one pinned artifact and promote it;
4. verify manifest, CAS, and view;
5. confirm the runtime lock or immutable image identity;
6. activate under the GPU lock and check health;
7. run and archive the smoke bundle;
8. run the performance/quality gate with identical suite content digests;
9. inspect per-case errors and telemetry; and
10. only then move clients or audited registry aliases.
