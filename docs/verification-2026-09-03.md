# RTX 4090 deployment verification — 2026-09-03

## Outcome

The four-model LLM Lab portfolio is installed, integrity-verified, loadable on
one NVIDIA GeForce RTX 4090, and reachable through the stable
OpenAI-compatible gateway. All four models passed the three-case smoke suite;
all four completed the repeatable performance suite with zero measured and
zero warmup errors.

The evidence below was generated from the final control-plane source tree, not
copied from publisher benchmarks. It verifies this workstation deployment and
its inference plumbing; it does not claim that the small smoke suite is a
complete model-quality evaluation.

## Fixed test conditions

| Item | Verified value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB reported |
| Driver / compute capability | 550.144.03 / 8.9 |
| Serving backend | pinned llama.cpp host binary |
| Inference policy | batch/concurrency 1, 8,192-token total context, all layers on GPU |
| Attention / KV cache | flash attention on; `q8_0` K and V cache |
| Gateway | `http://127.0.0.1:14000`, per-request active-state routing |
| Smoke suite | `smoke@1.0.0`, suite SHA-256 `7e6a2378effb36064aae1ec383ac3257be746771e231d7f543c5914c9c7f7fe2` |
| Smoke request contract | SHA-256 `2e69d9aa2ff64424f56acb1ce7aab8325ce58ddabc8367146efe7332d3c548ae` |
| Performance suite | `perf-4090@1.1.0`, suite SHA-256 `9350a711acefb48d9baef4febc15a5ec7cbc1f7fbb31211154ae5e37c0f03462` |
| Performance request contract | SHA-256 `fb302e0058394c0fa0caab119822c104116b60095687a2d6c9e89c31857930d8` |

The performance suite makes two warmup requests per case, then five measured
repetitions of each of two fixed chat workloads: a 128-token and a 512-token
generation. Sampling is deterministic (`temperature=0`, `top_p=1`, `seed=42`).

## Installed artifacts and capacity

| Role / artifact | Exact selected bytes | Manifest SHA-256 | Tree SHA-256 |
|---|---:|---|---|
| General — Qwen3.8 27B Q4_K_M + vision projector | 18,700,159,216 | `39bfcde9ff6275c050eb67d8dbedef0a40a9601ebeadfea757ae190de1abf7e7` | `d345b762c97329d6b40d1a1be200ec0d7775ff7d59163664b02bd6ef65a9390a` |
| Agent — Muse Glimmer 30B official KQuant + vision projector | 18,157,050,004 | `910e21a17e5b48d059c542cf322fc7071d16ee86d7075c7717cdcc9cff73ae7c` | `b26983ff268cf5603f6752200c0d3327a86acfd519f91d817632858355b5c967` |
| Fast — Ling 3.0 Tiny Q4_K_M | 4,823,895,906 | `e05da375b9718b068fba24cf49b2a22127c2712bb055c905bb7e829867e5a6c5` | `fafa6a14cbd89dbc6967fb55719403f9b3e9cbe53250ec39349178319923c622` |
| Code — Devstral Small 2 24B Q4_K_M + vision projector | 15,212,493,197 | `0c06e9703a5f86e139c10eab526a1f4111333ed122188eda8df85a18e2a17185` | `dec8d557a2856337fa8d6464412346ffb864c2d130d854d262859370cc58a439` |
| **Total** | **56,893,598,323 bytes (56.89 GB / 52.99 GiB)** | | |

All four artifact verifications checked the sealed manifest, every SHA-256 CAS
blob, and the immutable loader view. A dry-run garbage collection found zero
orphaned blobs.

The models easily fit the requested **2.7 TB decimal** budget. Even the
conservative case in which the Hugging Face acquisition cache and durable CAS
hold separate full copies is 113.79 GB, or about 4.21% of 2.7 TB. After also
reserving the storage policy's 540 GB safety margin, 2,046,212,803,354 bytes
(about 2.05 TB decimal) remain. At verification time the backing filesystem
reported 2,838,337,966,080 free bytes (2.58 TiB).

## Live RTX 4090 results

| Alias / model | Smoke | Server decode mean | Client completion mean | Mean end-to-end latency | Peak device memory | Peak GPU / power | Performance run |
|---|---:|---:|---:|---:|---:|---:|---|
| `local-general` — Qwen3.8 27B | 3/3 | 46.11 tok/s | 44.58 tok/s | 7,150 ms | 19,257 MiB | 99% / 377.50 W | `20260903T201244Z-perf-4090-632b9dfce0` |
| `local-agent` — Muse Glimmer 30B | 3/3 | 49.68 tok/s | 49.14 tok/s | 6,493 ms | 18,621 MiB | 99% / 370.89 W | `20260903T200954Z-perf-4090-c532faccaa` |
| `local-fast` — Ling 3.0 Tiny | 3/3 | 286.04 tok/s | 273.99 tok/s | 1,157 ms | 6,139 MiB | 94% / 161.40 W | `20260903T201519Z-perf-4090-ce22aad9aa` |
| `local-code` — Devstral Small 2 24B | 3/3 | 58.97 tok/s | 58.75 tok/s | 5,442 ms | 16,621 MiB | 97% / 383.20 W | `20260903T200714Z-perf-4090-7f16b9d605` |

Corresponding smoke bundles are:

- Devstral: `20260903T200654Z-smoke-7085fc9bc9`
- Muse: `20260903T200922Z-smoke-789f0ecd34`
- Qwen: `20260903T201215Z-smoke-e6b508064e`
- Ling: `20260903T201505Z-smoke-0bc2f6f1d2`

Every listed run has status `completed`, an immutable four-file bundle, a
verified suite and effective-request-contract digest, and zero API, scoring, or
warmup errors. The smoke comparison produced a valid same-contract composite;
the performance comparison correctly withheld a quality composite because the
performance tasks are intentionally unscored.

The throughput figures are means across all ten measured requests. End-to-end
latency is the non-streaming client wall time, not TTFT. `llama.cpp` supplies the
separate server decode timing. Peak memory/utilization/power are maxima from
`nvidia-smi` samples over the entire GPU, so memory includes the desktop and any
unrelated resident GPU processes; this makes the fit observation conservative,
not a model-only allocation figure. Each advertised native context is much
larger than 8K, but this report proves only the cataloged 8K profile.

## Runtime identity

The deployed runtime is:

- source commit: `95ef7fc16054e63b427a3ef00188e055ef7586d8`
- binary: `/mnt/md2/llm-lab/cache/runtimes/llama-cpp-cuda-4090/llama-server`
- binary SHA-256: `16641abf2f14d4323e49bc0c1ba97dd9ef5a63ebd573fe22aa10f08bfa535c52`
- size/mode/link count: 217,645,944 bytes, `0555`, one link
- version evidence: `0.3.0-dev (build 1, commit 95ef7fc)`, GNU 13.3.0
- build identity: Release, CUDA architecture 89, CUDA enabled, curl enabled,
  `BUILD_SHARED_LIBS=OFF`

The executable has no RPATH/RUNPATH and no mutable project-local llama/ggml
shared-library dependency. It still depends on host system libraries and the
NVIDIA CUDA/driver stack; those are part of the documented platform boundary,
not bytes covered by the executable hash. The launch environment is reduced to
a recorded baseline plus reviewed deployment variables.

## Control plane and release gates

All final runs recorded this control-plane evidence:

- source-tree SHA-256: `7fd422c5dbfe6503f73df8f45fa93b13dabdd7ffb00d2315ffc4184136f5d80c`
- `uv.lock` SHA-256: `73679dc5c3aa2c506a2f19cc3517ce0a0369cc3220d2df922b9872687cdcc079`
- measured Python control-plane files: 18
- package version: 0.1.0

Final validation included:

- **236 passing pytest cases** covering schemas, catalog, hashing, storage,
  registry, symlink/path confinement, runtime lifecycle and rollback, gateway,
  benchmarks, immutable bundles, DuckDB results, CLI, and acceptance flows;
- successful Python bytecode compilation and shell syntax validation;
- successful sdist and wheel builds, followed by a clean wheel installation,
  dependency compatibility check, import, CLI help, isolated data-root
  initialization, and installed-package catalog validation;
- a duplicate-key-safe catalog validation: 4 models, 4 artifacts, 6
  deployments, 2 suites, and 1 runtime lock;
- fresh verification of all four artifacts and all eight listed run bundles;
- enabled and active `llm-lab-gateway.service` with healthy live state;
- a final gateway completion from `local-fast` returning exactly
  `LLM Lab ready`; and
- an independent final audit reporting no remaining P0/P1 release blocker
  under the documented trust boundary.

## Remaining handoff boundary

The workspace did not yet have an initial Git `HEAD` at verification time, so
the run evidence deliberately records `git_commit: null` and `git_dirty: true`.
The exact source-tree and lockfile hashes above preserve byte identity for these
runs, but the operator should review, stage, and make the first repository
commit before treating this as a shared release. LLM Lab data paths, model
weights, caches, and generated run bundles are ignored and must not be added to
Git.

The DuckDB/SQLite metadata stores assume the managed data directories are not
being actively modified by another process running as the same Unix user. The
framework rejects pre-existing unsafe symlinks, hardlinks, writable immutable
evidence, and many check/use races, but it is not a security boundary against a
concurrent same-UID adversary with write access to the data root.
