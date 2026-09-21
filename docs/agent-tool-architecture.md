# Central agent and tool architecture

This document defines the extension boundary for LLM Lab's central agent and
read-only tools. The runtime registry currently ships `web_search`, `web_fetch`,
`calculator`, and `current_time` in the `standard-readonly` toolset. The YAML
manifests record reviewed contracts but are not loaded by `llm_lab.catalog`.
MCP is an optional adapter boundary rather than a requirement for built-ins.

## Design goals

1. Keep model serving, agent orchestration, and external tools independently
   replaceable and independently versioned.
2. Give every tool call a strict input/output contract, deadline, size bound,
   bounded invocation event, and explicit effect classification; add immutable
   deployment provenance before general promotion.
3. Treat network results as untrusted evidence, never as instructions.
4. Enforce authority in deterministic host code and operating-system controls;
   prompts and MCP annotations are descriptive layers, not security boundaries.
5. Preserve source URLs and tool identity now, then add retrieval timestamps,
   upstream warnings, contract versions, and persistent audit evidence before
   treating agent runs as reproducible evaluations.

## Components

```text
Browser / API client
        |
        v
Central agent layer  -------->  LLM gateway  -----> active local model
        |
        +---- policy engine
        |       |
        |       +-- reviewed toolset + per-request authority
        |       +-- effect, taint, deadline, and output-size checks
        |
        +---- tool adapter registry
                |
                +-- native adapter ----> loopback SearXNG ----> public engines
                |
                +-- reviewed MCP client ----> pinned MCP servers (optional)
        |
        +---- bounded invocation events (persistent audit is a promotion target)
```

The central agent is the only component allowed to combine model output with
tools. Browsers and models do not receive direct network or Docker access.
Adapters accept catalog IDs and validated arguments; they do not accept shell
fragments, executable paths, arbitrary environment variables, or arbitrary
backend URLs.

The baseline toolset separates service-backed discovery from native helpers:

| Tool | Implementation boundary | Network behavior |
|---|---|---|
| `web_search` | SearXNG JSON adapter at `LLM_LAB_SEARXNG_URL` | SearXNG queries public search engines |
| `web_fetch` | Native bounded HTTP client | Fetches one validated public HTTP(S) URL |
| `calculator` | Native restricted arithmetic parser | No network access |
| `current_time` | Native host clock and time-zone database | No network access |

`web_search` reports a structured, retryable failure when its loopback service
is unavailable; the three native tools do not depend on SearXNG. This first
adapter retains normalized result URLs but does not yet propagate SearXNG's
per-engine warnings or attach a retrieval timestamp. Callers must therefore not
interpret an empty result list as evidence of complete search coverage.

## Explicit remote delegation

An affirmative request such as "Can you run against Opus and Astra using open
router?" creates a delegation plan before the first model round. Names resolve
against the operator-approved model enum: Astra selects `openai/gpt-6-astra`, and
Opus selects the highest approved stable, non-batch Claude Opus version. Unknown
or unavailable explicit model IDs produce an actionable error; the agent must not
silently substitute another model or finish with a generic API-access denial.
When a request does not identify a model and several are approved, the agent asks
the caller to name one. With a single approved model, it can select that model.
Bare mentions, usage questions, and negated requests do not require delegation.
This is a conservative text heuristic, not a general intent classifier.

Only the planned remote models may execute, once each. Calls can be generated
together or in separate rounds. This narrow exception permits multiple reviewers
chosen by the user without allowing a remote result to authorize another tool or
network destination. After the planned attempts, the local model synthesizes
their actual results or failures with further tool execution disabled.

Deployment approval remains explicit: add the intended IDs to the existing
`LLM_LAB_OPENROUTER_ALLOWED_MODELS` list in the service environment and restart
the gateway. For example, this deployment approves `anthropic/claude-opus-5`
and `openai/gpt-6-astra`. API credentials remain outside version control.

## Repeated-answer recovery

When an answer repeats earlier assistant content instead of addressing the latest
request, the orchestrator withholds that answer and allows one recovery. Ordinary
questions recover using the latest request and collected evidence. Explicit
delegation happens before synthesis, so its evidence is reused during recovery,
including failures, rather than silently replaying the remote request.

During required delegation, the tool schema contains only `openrouter_delegate`
and the unresolved approved model IDs. A model that ignores the required tool call
gets one clean retry with the user context and existing evidence preserved. The
final answer runs without tools, with execution disabled on the server as well.
Missing required delegation or another repeated answer ends in a clear, bounded
error.

Remote attribution comes from actual tool results. The completion event retains
`repeated_answer` in `recovery_reasons`, and the UI displays that notice alongside
tool activity and response provenance. Scripted backend tests cover the recovery
transitions; a browser test covers expandable delegation evidence and provenance
after history reload. These tests use simulated remote results and incur no
OpenRouter charges.

## Declarative identities

Tool evidence follows the same separation already used for models:

| Identity | Meaning | Changes when |
|---|---|---|
| Tool contract | Stable semantic name, schemas, effects, and limits | Arguments, result meaning, effects, or normalization changes |
| Tool implementation | Exact adapter package, commit, or image | Executable code or dependencies change |
| Service deployment | Address, isolation, configuration, and executable digest | Runtime settings or network boundary changes |
| Toolset | Reviewed collection and composition policy | Membership or cross-tool authority changes |
| Target invocation evidence | One call's contract version, inputs, timing, output metadata, errors, and deployment identity | Every execution |

`catalog/tools/*.yaml` and `catalog/toolsets/*.yaml` are currently
documentation-only manifests. The existing strict Python catalog deliberately
ignores these directories. Before runtime loading is added, define Pydantic
schemas, duplicate-ID and cross-reference validation, immutable provenance
requirements, fixtures, migration rules, and fail-closed tests.

The v1 SSE API currently emits run and sequence IDs, tool name, call ID, round,
bounded caller-visible arguments, outcome, duration, and a schema-bounded
result or sanitized error. It does not persist tool turns or attach contract
version, retrieval time, deadline, or exact tool-deployment identity to every
event. Those are explicit promotion requirements, not properties of the
current implementation.

## Target invocation envelope

A later auditable invocation format should preserve an internal envelope
equivalent to:

```json
{
  "call_id": "toolcall_...",
  "tool_id": "web_search",
  "contract_version": "1.0.0",
  "arguments": {"query": "..."},
  "deadline": "2026-09-06T12:00:20Z",
  "authority": {"toolset_id": "standard-readonly"}
}
```

The current normalized result distinguishes success, timeout, policy rejection,
upstream failure, and adapter failure; partial-engine details are not yet
propagated. Successful open-world results retain direct source URLs, while
streamed invocation events identify the tool, call, outcome, and duration. A
future persistent audit record should also include retrieval time, contract
version, and exact deployment identity. Limit persisted logs to operational
metadata by default; research queries and returned page content can contain
private or copyrighted material. The server does not persist v1 agent events,
but a browser, reverse proxy, or other caller may do so and must apply its own
retention and redaction policy.

## Read-only research policy

"Read-only" means a call does not intentionally modify files, accounts,
messages, purchases, or remote records. It still consumes compute and sends the
query to external services. The central policy should enforce all of the
following:

- allow only reviewed tool IDs and exact contract versions;
- validate arguments with `additionalProperties: false` before dispatch;
- resolve service locations from operator configuration, then compare the
  normalized URL against the manifest allowlist;
- apply connect, total-time, response-byte, result-count, and concurrency limits;
- reject redirects or resolved addresses into loopback, link-local, private,
  multicast, and metadata networks for the native `web_fetch` adapter;
- permit only `http` and `https` source URLs in normalized search results;
- never place credentials, local file contents, hidden prompts, or unrelated
  conversation data into a search query;
- mark all snippets and fetched content as tainted, untrusted data;
- require a new explicit authority decision before tainted content can flow to
  a mutating or exfiltrating tool; and
- until per-engine warnings are propagated, label empty or sparse search output
  as incomplete rather than fabricating completeness; propagating the exact
  warning metadata remains a promotion requirement.

SearXNG itself is only a metasearch proxy. It does not make upstream content
trustworthy and does not eliminate search-engine logging or rate limits.

## Reviewed MCP extension path

MCP is an adapter option, not the central policy model. New MCP integrations
must pass the same contract, provenance, isolation, and evaluation gates as a
native adapter.

Use the [MCP 2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28)
as the review baseline and record the exact protocol revision in the tool
implementation manifest. The v1 runtime adapter supports reviewed Streamable
HTTP endpoints only, not legacy HTTP+SSE. `stdio` is deliberately disabled
until LLM Lab has an enforceable OS sandbox profile for child MCP servers;
checking an absolute executable path is not sufficient isolation from the
gateway user's files, environment, devices, or network.

The official MCP SDK materializes a tool reply before LLM Lab's central
serialized-result cap can inspect it. Input/output schema validation and the
central cap still protect the model-facing boundary, but they cannot bound the
client's transient memory while a reply is being received. Every MCP server,
reverse proxy, and container/cgroup must therefore impose response-body and
memory limits before the adapter is enabled.

A Streamable HTTP server must:

- bind to `127.0.0.1` unless intentionally placed behind an authenticated TLS
  proxy;
- validate every `Origin` header against an explicit allowlist to prevent DNS
  rebinding;
- authenticate every connection, including local connections when the threat
  model includes other local processes;
- enforce request-body, response-body, session, and memory limits plus short
  idle deadlines;
- expose one reviewed endpoint and no administrative interface on that listener;
- run as a dedicated unprivileged identity with no unnecessary filesystem,
  device, Docker-socket, or network access; and
- pin the server package/source plus all deployable image digests.

For `web_search`, an MCP projection should publish strict
`inputSchema` and `outputSchema`, return normalized data in `structuredContent`,
and include these standard annotations:

```json
{
  "readOnlyHint": true,
  "destructiveHint": false,
  "idempotentHint": true,
  "openWorldHint": true
}
```

Annotations are untrusted hints under the MCP specification. The central agent
must derive authority from locally reviewed manifests and hard policy, not from
what a server claims during discovery. Disable unused MCP capabilities such as
sampling, elicitation, roots, and server-initiated model access. Namespace tool
names with the reviewed server/adapter ID to avoid collisions.

## Review and promotion gate

A new native or MCP tool is not available to an agent until all of these are
complete:

1. Pin source/package/image identities and archive upstream license/security
   information.
2. Review input/output schemas, side effects, open-world behavior, secrets,
   redirects, DNS resolution, and worst-case resource use.
3. Add unit tests for schema validation, normalization, redaction, timeouts,
   response limits, malformed upstream data, and partial failures.
4. Add integration tests against a deterministic fixture server; do not make
   ordinary CI depend on public search engines.
5. Add an opt-in live smoke test that records upstream variability without
   asserting result ranking.
6. Test prompt-injection strings in titles/snippets and verify they remain data.
7. Test composition with every toolset member, especially any tool that can
   read private data, write state, communicate externally, or execute code.
8. Have a human approve the manifest and toolset change before enabling it.

Version contract changes independently from implementation updates. Breaking
schema or semantic changes require a new contract version; an implementation
security update can retain the contract version while changing its immutable
implementation identity.

## Initial operational rollout

1. Run SearXNG locally and validate `/healthz` plus one JSON query manually.
2. Verify the shipped native adapters against deterministic fixture services.
3. Initially expose `standard-readonly` only to test users and test models.
4. Retain source URLs in bounded tool events and show source cards in the UI.
   Host-enforced citation validation for generated prose remains a promotion
   requirement.
5. Benchmark usefulness, latency, engine failures, privacy, and injection
   resistance before enabling the toolset for general chat.
6. Add an MCP facade only when another MCP host is a concrete requirement;
   avoid introducing a second protocol boundary without a consumer.

## Authoritative references

- [MCP architecture](https://modelcontextprotocol.io/docs/learn/architecture)
- [MCP 2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28)
- [MCP Streamable HTTP and local-server security](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP tool schemas and annotations](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [SearXNG container installation](https://docs.searxng.org/admin/installation-docker.html)
- [SearXNG settings](https://docs.searxng.org/admin/settings/index.html)
