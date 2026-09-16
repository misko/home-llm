# FineWeb local training

This experiment continues pretraining Qwen3.8 27B on raw FineWeb text with QLoRA.
The training environment is isolated from LLM Lab's control environment. The
worker uses a local token snapshot so downloads and tokenization do not stall
the GPU, and checkpoint recovery replays the same shuffled sequence order.

## Current experiment

- Run directory: `/mnt/md2/llm-lab/runs/training/fineweb-24h-20260912`
- W&B: https://wandb.ai/projectspf/llm-lab-fineweb/runs/qwen38-fw-20260912
- GPU: RTX 4090, BF16 computation over a frozen 4-bit base, rank-8 adapters,
  512-token maximum sequences, one sequence per microbatch, eight accumulated
  microbatches. CPU embedding and unused vision weights preserve GPU memory.
- Data: 128M training tokens from a bounded FineWeb sample, with separate
  document groups for validation and test. Exact normalized-text and approximate
  SimHash deduplication precede splitting. This is not a uniform whole-corpus sample.
- Budget: 86,400 seconds in the training phase, including checkpoint and periodic
  validation time. Startup, baseline evaluation and final export take extra time.
- Checkpoints: after the first update and approximately every 15 minutes; retain
  two complete snapshots containing adapter, optimizer, RNG, cursor and elapsed
  budget. A terminated process releases the serving reservation automatically.
- Final output: original Q4_K_M GGUF plus an F16 LoRA adapter, registered in LLM
  Lab as `local-fineweb` after export and inference smoke tests. Select it in the
  model picker to test; the existing default deployment is preserved.

The supervisor runs worker, export and W&B final reporting in order. It resumes
after failure, with systemd allowing three starts per hour. A frozen source copy
and manifest live in the run directory. User lingering is enabled so logging out
does not terminate the service. Host shutdown or sleep still interrupts compute;
the last completed checkpoint is resumable and downtime does not spend the budget.

## Operations

```bash
systemctl --user status llm-lab-fineweb-24h
tail -f /mnt/md2/llm-lab/runs/training/fineweb-24h-20260912/service.log
cat /mnt/md2/llm-lab/runs/training/fineweb-24h-20260912/status.json
cat /mnt/md2/llm-lab/runs/training/fineweb-24h-20260912/session.json
# Gracefully save and pause; start resumes the same budget and W&B run.
systemctl --user stop llm-lab-fineweb-24h
systemctl --user start llm-lab-fineweb-24h
```

W&B receives loss, learning rate, gradient norm, useful target tokens/second,
processed tokens, elapsed/remaining time, validation NLL/perplexity, GPU memory,
utilization, temperature and power. Final summaries include held-out NLL and
original/candidate smoke pass rates. Raw documents and model weights stay local.
Device utilization includes any other processes on the same GPU; token throughput
measures this training worker. A high utilization number alone is not a throughput
or model-quality guarantee.

The running experiment now targets an 85% training duty cycle to share compute
with Humandescent. `gpu-duty-cycle.json` in the run directory contains
`{"duty_cycle": 0.85}`; edits are read at each optimizer update. Training synchronizes
CUDA and briefly sleeps after each microbatch, and evaluation uses the same
throttle. This is cooperative time sharing, not a strict utilization or VRAM cap.
Other GPU jobs can bring total utilization above 90%. Throttling time counts
toward the existing 24-hour budget, so fewer tokens will be processed.
The initial 90% target was reduced to 85% for extra headroom. Short GPU utilization
samples can still read 100% during a microbatch. Evaluate sustained averages and
Humandescent's workload responsiveness; this mechanism does not cap every reading.
Operational setting changes are recorded in `gpu-throttle-changes.jsonl`.

The shared-GPU worker runs from `source-shared-gpu`; the initial source snapshot
is preserved. `source-revisions.json` records this operational code change without
changing the learning recipe or invalidating existing checkpoint fingerprints.
W&B records the running worker hash, duty-cycle target and sleep time. Final
export provenance includes the source revisions and throttle setting.

## Reproduce and validate

Use Python 3.11 and `uv pip install --python training/.venv/bin/python -r
training/requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
--index-strategy unsafe-best-match` in a separate environment. The lock records the
tested Torch/Unsloth/Transformers versions; the existing control environment stays
independent. Configure paths, pinned model revisions, W&B identity and budget in
the run's `config.json`. Preparation consumes pinned parquet paths listed in
`fineweb-files.json`; run `prepare.py` with the training Python to create a snapshot.
The current executable preparation supports FineWeb. Other corpus adapters remain
in the broader plan at `docs/research/web-corpus-training-plan.md`.

Run repository tests with `uv run pytest -q`, and preparation/checkpoint tests with
`training/.venv/bin/python -m pytest -q training/test_pipeline.py`. GPU preflight
uses `worker.py CONFIG --mode probe`, followed by control Python `export.py CONFIG
--probe`; set `PYTHONPATH` to this repository's `src`. The probe checks real weight
updates, adapter and optimizer restoration, GGUF conversion, and inference. Use a
separate run directory and W&B ID for a short real-data calibration.

Raw-web adaptation can weaken instruction following. Compare the resulting model
against the original using held-out loss and LLM Lab tasks before adopting it.
The three smoke cases establish basic serving compatibility, not broad quality.
