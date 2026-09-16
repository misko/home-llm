# Heretic after FineWeb

Heretic is installed separately at `/mnt/md2/llm-lab/tools/heretic/.venv`, from
upstream commit `3521f8648a0dccf6e12a92666862632235fac7e6` (version `2.0.0.dev0`).
The checkout is `/mnt/md2/llm-lab/tools/heretic/source`. Its dependency environment
uses Torch 2.7.1+cu126 and Transformers 5.17.0; exact versions are recorded in
`requirements.lock.txt`. The running FineWeb environment has not been modified.

The prepared experiment is
`/mnt/md2/llm-lab/runs/heretic/qwen38-fineweb-20260913`.
`launcher.json` records paths, parent revision and W&B identity;
`config.toml` is the Heretic profile. No Heretic GPU job is scheduled or running.

## Input and output

The input will be the pinned official Qwen3.8 27B BF16 parent with the **final
FineWeb adapter merged into it**. The launcher discovers the completed checkpoint
from `trained.json` and requires `ready.json`, so FineWeb's evaluation and LLM Lab
export finish first. It validates checkpoint hashes and the final step number.
It never uses an intermediate checkpoint or the GGUF as Heretic's input.

The full-precision parent is already downloaded in LLM Lab's HF cache. The CPU
merge writes a separate immutable `fineweb-merged-bf16` directory with provenance
and file hashes. Allow approximately 55 GB for that additional copy, in addition
to the cached parent. The merge can use tens of GB of system RAM. It does not
overwrite the parent, FineWeb checkpoint, or `local-fineweb` deployment.

Heretic will save a separate `heretic-adapter`, trained relative to that merged
FineWeb input. The adapter needs that exact input model: it is not a replacement
for the FineWeb adapter and must not be attached directly to the original GGUF.
`result.json` records its hash and W&B URL. A later GGUF conversion and separate
LLM Lab registration are required to serve the Heretic variant through the lab.

## Run after completion

From `/home/mouse9911/gits/llms`:

```bash
# Read-only readiness check, safe while FineWeb is training.
.venv/bin/python training/heretic/run.py \
  /mnt/md2/llm-lab/runs/heretic/qwen38-fineweb-20260913/launcher.json check

# CPU merge, then disposable full-size GPU compatibility/memory probe.
.venv/bin/python training/heretic/run.py \
  /mnt/md2/llm-lab/runs/heretic/qwen38-fineweb-20260913/launcher.json probe

# Merge/verify input, recheck GPU compatibility, then run the initial study.
.venv/bin/python training/heretic/run.py \
  /mnt/md2/llm-lab/runs/heretic/qwen38-fineweb-20260913/launcher.json run
```

`prepare-input` performs only the CPU merge. All modifying actions refuse to
start until FineWeb training and export are complete. GPU work holds LLM Lab's
training reservation, preventing serving/benchmark conflicts. If a lab model is
active, stop it through LLM Lab before the Heretic GPU probe. Humandescent's
existing processes are left running.

The starting profile uses NF4, BF16 computation, batch size 1, a 19 GiB GPU
placement budget, and up to 112 GiB CPU placement. The local wrapper enables
bitsandbytes CPU offload; actual full-size compatibility is still **unverified**
until the post-training GPU probe. The probe tests generation, residual extraction,
and an intervention on disposable weights, and requires at least 1.5 GB free VRAM.
If necessary, reduce the GPU placement budget in `config.toml` and repeat it.

The wrapper follows FineWeb's live `gpu-duty-cycle.json` (currently 85%), yielding
after forwards and GPU low-rank SVD operations. This is cooperative time sharing,
not a strict peak-utilization or memory cap; loading and some analysis operations
can still burst to 100%. Offloading may make the 27B experiment substantially
slower than Heretic's smaller-model examples. Estimate duration from actual
trials before increasing the study size.

## Study and metrics

The initial study has 32 trials, including 8 startup trials, and generates up to
64 tokens per response. It uses 400 prompts per direction and 100 evaluation
prompts per scorer. Local datasets are pinned to the commits used in upstream's
Qwen3.5 tests:

- `mlabonne/harmless_alpaca`: `02c6a92cfcf11bb0c387334f8146d149d65b587f`
- `mlabonne/harmful_behaviors`: `01cead01398926d81f7c52bdb790ee8cf77ebba7`

Dataset files and provenance are under `/mnt/md2/llm-lab/tools/heretic/datasets`.
These prompts are separate from the FineWeb train/validation/test splits.

Heretic optimizes refusal-keyword rate and KL divergence against the **FineWeb
input**, saving Optuna study progress under `study`. Trial selection uses index 0
of upstream's sorted Pareto front: lowest first objective, then the next objective.
This does not establish the best general-purpose model. Compare the original,
FineWeb-only and Heretic variants on useful tasks before choosing one.

W&B will use project `projectspf/llm-lab-fineweb`, run ID
`qwen38-fw-heretic-20260913`. It records trial metrics, duty-cycle target and system
telemetry. No model weights, prompt texts or responses are uploaded. The W&B run
is created only when the real Heretic study starts.

## Validation and reinstall

Validation completed without allocating GPU memory: CLI/import checks, upstream
configuration tests, checkpoint completion/corruption guards, a tiny-model CPU
merge equivalence test, local prompt loading, and a meta-device check finding all
64 Qwen layers and matching all 512 FineWeb adapter tensors.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /mnt/md2/llm-lab/tools/heretic/.venv/bin/python -m pytest -q \
  training/heretic/test_workflow.py
```

To recreate the environment, check out the pinned source at the path above, create
a Python 3.11 venv, and install `training/heretic/requirements.lock.txt` using
`uv pip install --python PATH_TO_VENV/bin/python -r training/heretic/requirements.lock.txt
--extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match`.
The source dependency in the lock refers to that local pinned checkout.

Upstream documentation: https://github.com/p-e-w/heretic/tree/3521f8648a0dccf6e12a92666862632235fac7e6
