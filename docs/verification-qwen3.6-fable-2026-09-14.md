# Qwen3.6 Fable Fusion 711 verification — 2026-09-14

This report records the local qualification of
`DavidAU/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF`
on the RTX 4090 host. Measurements are observations from this machine, not
claims about other hardware.

## Immutable identity

- GGUF repository revision: `9e521b228d6a996d02e43602436367f954bd576d`
- Fine-tune source revision: `e3b8b67b5ee28d887d1f129cec42fd759e4e1261`
- Selected weights: `Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf`
- Weights size: `18,498,575,840` bytes
- Weights SHA-256: `c796c2c011eaa0edf06395ff49cda5bfd4843ad52b86b58a83296dfc33849e4e`
- Vision projector: `mmproj-F16.gguf`, `927,607,360` bytes, SHA-256
  `eacf610d1ee4bd5ed0197a0777dd8f4fceb8eefa27009067c7d496cb68fbde45`
- Selected logical total: `19,426,183,200` bytes
- LLM Lab manifest SHA-256:
  `0f76df725e871465485a01059bbcf4cee0ed5468d71d33b26647320f94ef1b4a`
- Tree SHA-256:
  `3ec96c5c8df875ada13820d8c7a61a207cfd0ad8928c69b9c6d8aee08f1e6f79`

Q4_K_M was selected because it follows the existing llama.cpp catalog
convention and leaves useful VRAM headroom on a 24 GB GPU. The Q5_K_M artifact
alone is 21.18 GB and the Q6_K artifact is 24.03 GB, before the projector,
context, runtime, and other host GPU consumers. The catalog now supports
per-file expected size and SHA-256 fields; pull, promotion, and activation
reconciliation enforce them in addition to the pinned Hub revision.

## Runtime configuration

Both profiles use the locked llama.cpp CUDA runtime at commit
`95ef7fc16054e63b427a3ef00188e055ef7586d8`, whose installed binary SHA-256 is
`16641abf2f14d4323e49bc0c1ba97dd9ef5a63ebd573fe22aa10f08bfa535c52`.
They fully offload at 8K context, one slot, flash attention, Q8_0 K/V cache,
the F16 projector, and two-token MTP speculative decoding.

- `qwen3.6-27b-fable-fusion-711-4090-8k` / `local-fable` disables reasoning
  for concise chat, JSON, and tool behavior.
- `qwen3.6-27b-fable-fusion-711-4090-8k-reasoning` /
  `local-fable-reasoning` enables reasoning explicitly.

The runtime log confirms creation of the MTP draft context and loading of the
multimodal projector. Backend startup from process start to `model loaded` was
3.31–3.37 seconds over three launches. The first complete CLI activation took
13.56 seconds because it also re-hashed the 19.4 GB immutable artifact; a later
profile switch took 22.98 seconds including shutdown, re-verification, and
startup.

## Functional evidence

The successful standard smoke bundle is
`/mnt/md2/llm-lab/runs/20260914T191422Z-smoke-bb5195ca93`. All three samples
passed with no errors:

- chat returned `Earth.`;
- strict JSON returned `{"count": 2, "status": "ok"}`;
- forced tool use emitted `get_weather` with `{"city":"Seattle"}`.

The reasoning bundle is
`/mnt/md2/llm-lab/runs/20260914T191645Z-fable-reasoning-smoke-300eb5c985`.
It passed the discount-then-tax case, returned the correct `$70.40` result,
and exposed 2,115 characters of separate reasoning content. Its 924-token
completion took 10.47 seconds end to end; server decode was 89.45 tok/s, MTP
accepted 602 of 642 draft tokens (93.77%), and prompt processing was 313.58
tok/s for 43 tokens.

An earlier diagnostic run with reasoning enabled and the standard suite's
256-token cap is retained at
`/mnt/md2/llm-lab/runs/20260914T191238Z-smoke-b10d5e04bc`. Chat passed, while
JSON and tool cases reached the cap during verbose reasoning. This is why the
default profile disables reasoning and the explicit reasoning suite budgets
1,024 tokens.

## Performance and memory

The performance bundle is
`/mnt/md2/llm-lab/runs/20260914T191446Z-perf-4090-706955b375`. It contains two
warmups and five measured repetitions per case, with no request errors.

| Case | Mean latency | Client tok/s | Server decode tok/s | Prompt tok/s | MTP acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 output tokens | 1,917.77 ms | 66.74 | 71.16 | 39.91 for 4 uncached tokens | 63.39% |
| 512 output tokens | 8,157.11 ms | 62.77 | 65.54 | 293.99 for 63 tokens | 54.51% |

Before loading, total GPU memory use was 1,713 MiB because two unrelated host
processes were already using the card. Idle loaded use was 21,275 MiB, a
19,562 MiB increase attributable to this deployment snapshot, leaving 2,928
MiB free. During the performance run telemetry stayed at 21,305 MiB, averaged
93.97% GPU utilization, reached 99% utilization, and peaked at 414.31 W.
Observed server RSS ranged from 1,243,496 KiB immediately after load to
2,602,360 KiB after smoke traffic. A separate 0.2-second sampled reasoning
request observed RSS from 1,521,577,984 to 2,030,137,344 bytes and constant
GPU allocation of 21,291 MiB. These RAM numbers describe process RSS and may
include demand-paged mappings and cache state; they are not artifact size.

## Reproduction

From the repository root, using the managed data plane:

```bash
export LLM_LAB_DATA=/mnt/md2/llm-lab
uv run llmctl catalog validate
uv run llmctl artifact pull qwen3.6-27b-fable-fusion-711-mtp-q4-k-m --reserve-gb 100
uv run llmctl artifact verify qwen3.6-27b-fable-fusion-711-mtp-q4-k-m
uv run llmctl serve activate qwen3.6-27b-fable-fusion-711-4090-8k
uv run llmctl benchmark run smoke
uv run llmctl benchmark run perf-4090
uv run llmctl serve activate qwen3.6-27b-fable-fusion-711-4090-8k-reasoning
uv run llmctl benchmark run fable-reasoning-smoke
```

Re-running a benchmark intentionally creates a new write-once run ID. Compare
the embedded artifact, deployment, runtime lock, suite, and request-contract
hashes rather than assuming a run is equivalent by name alone.
