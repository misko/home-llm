# LLM Lab

LLM Lab is a reproducible control plane for storing, serving, and comparing
open-weight language models on a single-GPU workstation. It keeps five
declarative identities separate—model, artifact, deployment, runtime lock, and
benchmark suite—then records their exact applicable combination in an
immutable benchmark run.

The initial catalog is designed for an NVIDIA RTX 4090 with 24 GB VRAM and a
2.7 TB data-plane budget. It exposes deployments through an OpenAI-compatible
chat-completions contract, so clients and benchmark suites do not need to know
which supported backend is active.

> **Interface status:** commands shown as current in this documentation are
> present in `llmctl --help` and backed by the Python modules. Any forthcoming
> command is explicitly marked **Planned**; do not infer a command or flag from
> a design description.

## What is implemented

- Strict, immutable Pydantic schemas and a duplicate-key-safe YAML catalog
- A Typer CLI for catalog, artifact, runtime, gateway, benchmark, registry, and
  storage operations
- Commit-pinned Hugging Face pulls with exact file selection and size checks
- SHA-256 content-addressed storage (CAS), sealed manifests, and materialized views
- A claimed data-root sentinel plus no-follow checks that reject symlinked
  managed roots, state, locks, and logs
- A SQLite registry with atomic aliases, audit history, and alias rollback
- Single-GPU runtime locking, atomic active state, health-gated activation, and
  automatic restoration of the previous deployment after activation failure
- A declarative host-runtime lock that verifies the llama.cpp binary path,
  SHA-256, and version evidence before activation and benchmarking
- A stable OpenAI-compatible gateway with optional bearer authentication,
  streaming pass-through, and per-request active-state routing
- A browser console for chat, reviewed model switching, benchmark launch and
  history, storage inspection, and system/runtime identity
- Backend launch plans for llama.cpp, vLLM, SGLang, TensorRT-LLM, external
  endpoints, and a mock backend
- An async OpenAI-compatible benchmark runner with warmups, repetitions,
  per-sample errors and usage, six scorers, end-to-end performance metrics, and
  best-effort NVIDIA telemetry
- Atomic four-file run bundles plus an append-only DuckDB comparison index

## Five independent identities

| Identity | Answers | Examples | Changes when |
|---|---|---|---|
| Model | What learned behavior and license are intended? | `qwen3.8-27b` | The upstream checkpoint, capabilities, or license changes |
| Artifact | Which exact runnable bytes are used? | `qwen3.8-27b-q4-k-m` | Quantization, format, converter output, files, or source commit changes |
| Deployment | How are those bytes served? | `qwen3.8-27b-4090-8k` | Backend, runtime image/binary, context, KV cache, port, or reasoning policy changes |
| Runtime lock | Which reviewed runtime build executes them? | `llama-cpp-cuda-4090` | Runtime source commit, build recipe, binary path/bytes, or version evidence changes |
| Suite | What workload and scoring contract is run? | `smoke@1.0.0` | Prompts, tools, sampling, repetitions, expectations, or task coverage changes |

A run is evidence that joins these identities with resolved artifact hashes,
verified runtime details, the full suite definition and its SHA-256, hardware,
sampling settings, responses, timing, token usage, errors, and telemetry. Model
names alone are therefore never treated as a reproducible result key.

See [Architecture](docs/architecture.md) for the full data flow and invariants.

## Starter portfolio

The four checked-in artifacts are complementary 4090 operating profiles, not a
claim that four models cover every workload.

The complete checked-in catalog contains **4 models, 4 artifacts, 6 deployment
profiles, 2 benchmark suites, and 1 runtime lock**. The extra profiles are the
Qwen reasoning variant and an offline mock canary; the four primary aliases are
listed below.

| Public alias | Model and role | Runnable artifact | Cataloged bytes | Why it is included |
|---|---|---:|---:|---|
| `local-general` | Qwen3.8 27B; general multimodal reasoning | Community Q4_K_M GGUF | 18.70 GB / 17.42 GiB | Broad chat, reasoning, tools, structured output, vision, and video |
| `local-agent` | Muse Glimmer 30B; multimodal agent/tool use | Official 17 GB-class KQuant GGUF | 18.16 GB / 16.91 GiB | Strong local agent behavior and an official 24 GB GPU quantization |
| `local-fast` | Ling 3.0 Tiny; fast reasoning/tool use | Official Q4_K_M GGUF | 4.82 GB / 4.49 GiB | 7.9B-total/1.3B-active MoE for latency-sensitive work |
| `local-code` | Devstral Small 2 24B; software-engineering agent | Community Q4_K_M GGUF | 15.21 GB / 14.17 GiB | Focused repository exploration, editing, and coding tools |

Together, selected artifact files total **56,893,598,323 bytes** (56.89 GB,
52.99 GiB). A conservative upper bound with separate Hugging Face and CAS copies
is about 113.79 GB (105.97 GiB), before Hub/Xet metadata, conversion scratch, KV
cache, or benchmark output. Materialized views normally reference CAS content.
They can fall back to a copy when links are unavailable, so capacity checks must
still inspect physical use.

Every catalog entry pins a full upstream commit. Community conversions retain a
separate link to the official model identity; a conversion is never silently
treated as the publisher's original bytes.

The primary `local-general` profile defaults reasoning off so short chat,
structured-output, and tool assertions have predictable token budgets. The
same Qwen artifact is also available as `local-general-reasoning`, whose
deployment enables reasoning explicitly. Muse retains its native automatic
reasoning behavior, so the smoke suite allows a larger 256-token response
budget. This is an example of why deployment identity must remain separate from
model and artifact identity.

## Quick start

Requirements are Python 3.11 or newer and
[`uv`](https://docs.astral.sh/uv/). Building or changing the browser console
also requires Node.js 20.19+ or 22.12+ and npm. GPU serving additionally needs
a compatible NVIDIA driver and either Docker with NVIDIA Container Toolkit or
a local backend binary.

```bash
uv sync --dev --locked
uv run pytest -q
```

Keep code and heavyweight data on separate paths. `LLM_LAB_REPO` defaults to the
current directory; `LLM_LAB_DATA` defaults to `.llm-lab-data` under it.

```bash
export LLM_LAB_REPO="$PWD"
export LLM_LAB_DATA="/absolute/path/to/llm-lab-data"

uv run llmctl init
uv run llmctl catalog validate
uv run llmctl storage report
```

Initialization claims the dedicated data root with `.llm-lab-root`. It refuses
an unrelated non-empty directory and rejects symlinked data roots or managed
directory components; do not replace managed paths with links to other trees.

Before downloading, read the capacity and promotion procedure in
[Operations](docs/operations.md). Model pulls are large network operations and
may require accepting upstream terms even when the catalog itself validates.

## Common serving contract

Backends bind to loopback by default. The foreground gateway provides a stable
address while resolving the ready active deployment for every request. In one
terminal, after activating `local-general`, start it with optional bearer auth:

```bash
LLM_LAB_GATEWAY_API_KEY="replace-with-a-secret" \
  uv run llmctl serve gateway --host 127.0.0.1 --port 14000
```

The deployment's `public_alias` remains the client-facing model name:

```bash
curl --fail-with-body http://127.0.0.1:14000/v1/chat/completions \
  -H 'Authorization: Bearer replace-with-a-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-general",
    "messages": [{"role": "user", "content": "Reply with READY."}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

The gateway passes through `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, and `/v1/responses`, including streaming responses. Its
`/health` endpoint remains available without a bearer token. It does not provide
TLS, and direct backend ports do not gain gateway authentication; keep both on
loopback or put a reviewed TLS proxy and firewall in front.

The gateway rejects unreviewed `Host` headers to prevent DNS-rebinding access.
Loopback names and the machine hostname are accepted by default. When a reviewed
reverse proxy uses another hostname, set the complete comma-separated allowlist
with `LLM_LAB_ALLOWED_HOSTS` (for example,
`localhost,127.0.0.1,llm.example.internal`). Host validation is not a user
authentication boundary; remote deployments still need an authenticated TLS
proxy and an explicit origin/session policy.

The raw `/v1/*` routes remain caller-managed: the gateway does not inject the
agent endpoint's generation default, so clients choose their own `max_tokens`
value. For server-managed tools, the
gateway also exposes `POST /api/v1/agent/turns` as a typed SSE stream and
`GET /api/v1/agent/toolsets` as the reviewed tool catalog. The initial
`standard-readonly` toolset contains `web_search`, `web_fetch`, `calculator`,
and `current_time`; it applies strict schemas, deadlines, byte/token/round
budgets, source-URL provenance, and open-world chaining controls outside the
model. `web_search` uses the loopback SearXNG service configured by
`LLM_LAB_SEARXNG_URL`, which defaults to `http://127.0.0.1:18888`.
The shipped limits allow one running and one queued agent turn, six model
rounds, and four serial tool calls per round. `max_tokens` defaults to 32,000
when omitted and accepts values from 1 through 32,768. This value is a per-round
upper bound, not a promised response length: prompt, conversation, and tool
traffic share the active deployment's context window, and the model may stop
earlier. The agent also reserves generation allowance against a 196,608-token
cumulative turn budget. Each model round has a 120-second deadline and the
whole turn has a 180-second deadline; model-round responses are capped at 2 MiB
and caller-visible assistant text at 262,144 characters. Any of those limits,
or the active context window, can end a response before the requested token
ceiling. Each tool call has a 20-second deadline. Web requests additionally
have a 12-second HTTP timeout, a 1 MiB upstream-response cap, and a 60 KiB
normalized result cap. Source URLs are retained in tool events and console
source cards; this first version does not reject otherwise valid assistant text
solely for omitting an inline citation.

```bash
curl --no-buffer --fail-with-body http://127.0.0.1:14000/api/v1/agent/turns \
  -H 'Authorization: Bearer replace-with-a-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Search for the SearXNG documentation."}],
    "toolset": "standard-readonly",
    "temperature": 0,
    "max_tokens": 32000
  }'
```

Caller messages must start and end with `user` and strictly alternate with
`assistant`. Optional session guidance belongs in the bounded `instructions`
field; callers cannot inject system or tool-role messages. The agent endpoint
accepts inline PNG/JPEG/WebP data URLs only, while the raw `/v1` contract keeps
its existing caller-managed multimodal behavior.

## Web console

The gateway serves the compiled console at
[`http://127.0.0.1:14000/ui/`](http://127.0.0.1:14000/ui/). It uses the same
catalog, runtime manager, durable operation records, and OpenAI-compatible chat
routes as the CLI and gateway. New console sessions request up to 32,000 output
tokens by default, with an adjustable ceiling of 32,768. This remains an upper
bound: the active deployment's shared context window and operational limits can
end a response sooner. Model activation is asynchronous: the page
submits one reviewed deployment ID, then polls the durable operation through
artifact verification, switching, readiness, and any rollback.

Keep the production gateway on loopback. From another workstation, use an SSH
tunnel and open the loopback URL in the local browser:

```bash
ssh -N -L 14000:127.0.0.1:14000 user@kalman
```

For frontend development, install the locked npm tree and start Vite. The dev
server listens on port 5173 and proxies API/chat requests to the loopback
gateway:

```bash
npm --prefix web ci
npm --prefix web run dev
# Development only: http://kalman:5173/ui/
```

Port 5173 has no login boundary and exposes lifecycle operations to every host
that can reach it. Use it only on a trusted, firewalled network. The browser
console does not currently implement a bearer-token login flow, so setting
`LLM_LAB_GATEWAY_API_KEY` protects `/api/v1/*` but makes interactive console calls
unauthorized. For shared or untrusted networks, put an authenticated TLS/session
proxy in front rather than exposing Vite or the gateway directly.

Run the frontend gates with:

```bash
npm --prefix web run typecheck
npm --prefix web test
(cd web && npx playwright install chromium && npm run test:e2e)
npm --prefix web run build
```

The production bundle is committed under `src/llm_lab/web_dist` and packaged in
the Python wheel. Run `git diff --exit-code -- src/llm_lab/web_dist` after the
build in release automation to prevent stale browser assets.

## Reproducible results

Each successful benchmark write creates a new directory containing exactly:

```text
run.json          identity, environment, status, and bundle hashes
summary.json      execution and per-case aggregates
samples.jsonl     request, response, scores, latency, usage, and error per sample
telemetry.jsonl   timestamped GPU observations or an explicit unavailable record
```

Bundles are atomically published, never overwritten by the writer, and verified
before DuckDB indexing. Every new bundle carries the complete validated suite
definition and its canonical SHA-256, plus a separate digest of the effective
HTTP request contract after all runner and per-run extensions are applied.
Comparisons remain per task; LLM Lab suppresses a composite when runs have
different suite or request-contract content, task sets, skipped capabilities,
unscored tasks, measured errors, or warmup errors.

Performance output reports end-to-end non-streaming request latency and client
completion throughput (`completion_tokens / elapsed_seconds`). When the backend
returns native timing fields, those are retained separately. These values are
not time-to-first-token (TTFT), which this non-streaming runner does not measure.

## Documentation

- [Architecture and invariants](docs/architecture.md)
- [Central agent and reviewed tool-extension architecture](docs/agent-tool-architecture.md)
- [Pinned local SearXNG research deployment](deploy/research/searxng/README.md)
- [Installation, storage, serving, benchmarking, rollback, and recovery](docs/operations.md)
- [Executable acceptance checklist](docs/acceptance.md)
- [Verified RTX 4090 deployment evidence (2026-09-03)](docs/verification-2026-09-03.md)
