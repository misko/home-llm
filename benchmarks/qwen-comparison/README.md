# Qwen comparison v0.1

The four-model comparison is implemented and queued behind the active 24-hour FineWeb training/export. Both Heretic variants start independently from the same completed FineWeb parent. All primary artifacts use the same BF16 merge → pinned converter → Q4_K_M pipeline, 8K context, and nonthinking settings. The matched original Qwen artifact is already registered in Lab as `compare-qwen`.

The [design](../../docs/research/qwen-comparison-benchmark-plan.md) explains the experiment. [manifest.yaml](manifest.yaml) summarizes it; the immutable execution inputs and results live at `/mnt/md2/llm-lab/runs/comparison/qwen-comparison-v0.1`.

## Execution and progress

`llm-lab-qwen-comparison.service` is enabled and running. It verifies frozen source/validation hashes, waits for both FineWeb completion and export receipts, then executes:

1. Merge/export FineWeb, compare reference logits, and verify inference sharing with an actual Humandescent CNN task.
2. Run short development checks and probe **both** Heretic variants before either search.
3. Give each search four hours of accumulated GPU-job wall time, up to 32 trials. Continuously record GPU energy/utilization/free VRAM and real Humandescent task latency; the same sharing policy must pass. Heretic++ must satisfy its development capability constraints to produce a winner.
4. Merge, validate, quantize, and register the remaining artifacts in Lab.
5. Run 937 frozen final cases per model, isolated code grading, and held-out NLL on 65,536 identical target tokens.
6. Run counterbalanced performance blocks, prepare blind review packets, and publish aggregate metrics to W&B.

The supervisor stops at a failed stage and records its log; successful stages have immutable receipts. It ends in `awaiting_blind_review`, with a full blinded rubric packet and a deterministic 20% audit packet stratified by task and request label. The audit import requires a different reviewer, exact coverage of the selected blind IDs, and response-bound primary judgments. It reports category agreement, support agreement, a confusion matrix, and Cohen’s kappa. No four-model quality scores are available yet.

```bash
systemctl --user status llm-lab-qwen-comparison.service
cat /mnt/md2/llm-lab/runs/comparison/qwen-comparison-v0.1/status.json
# Pause the comparison and its current child process group:
systemctl --user stop llm-lab-qwen-comparison.service
# Resume from verified successful stages:
systemctl --user start llm-lab-qwen-comparison.service

# After separate reviewers complete the JSON packets:
PYTHONPATH=/mnt/md2/llm-lab/runs/comparison/qwen-comparison-v0.1/source-v9/src \
  /mnt/md2/llm-lab/tools/heretic/.venv/bin/python -m llm_lab.comparison \
  /mnt/md2/llm-lab/runs/comparison/qwen-comparison-v0.1 import-reviews PRIMARY_REVIEWS.json
PYTHONPATH=/mnt/md2/llm-lab/runs/comparison/qwen-comparison-v0.1/source-v9/src \
  /mnt/md2/llm-lab/tools/heretic/.venv/bin/python -m llm_lab.comparison \
  /mnt/md2/llm-lab/runs/comparison/qwen-comparison-v0.1 import-audit AUDIT_REVIEWS.json
```

The [service unit](../../deploy/systemd/llm-lab-qwen-comparison.service) runs the frozen `source-v9` copy. Workspace edits cannot silently alter the running experiment. A source change requires a new snapshot and validation; existing stage identities cannot be reused across a changed protocol. The training service remains independent.

## Results and controls

The local `report-final.html` is a four-column scorecard with paired differences, confidence intervals, pending counts, operating metrics, and a response-level failure browser. [W&B scorecard](https://wandb.ai/projectspf/llm-lab-fineweb/runs/qcomp-scorecard-final) receives aggregate JSON; raw generated responses stay local. Training continues in [its existing W&B run](https://wandb.ai/projectspf/llm-lab-fineweb/runs/qwen38-fw-20260912).

The scorecard reports the two Heretic searches beside one another: allocated and consumed GPU-job wall time, completed and total trial counts, selected trial IDs, system-level shared-GPU energy, mean utilization, minimum free VRAM, and Humandescent p95 latency. This makes the equal-time primary comparison and equal-completed-trial secondary comparison explicit.

The run directory contains `dataset-lock.json`, `cases.jsonl`, `source-lock.json`, `execution.json`, `artifacts.json`, `validation.json`, `stage-plan.json`, and per-stage logs. Final selections include IFEval, GSM8K, MMLU-Pro, official HumanEval+ v0.1.10, XSTest, and versioned grounding/tool/context fixtures. There are 469 separate development cases. Public benchmark contamination remains a limitation; no zero-contamination claim is made.

The post-freeze FineWeb overlap audit scanned all 189,184 raw documents considered during preparation and reproduced the 185,077 unique documents admitted to the training split. It found no exact or threshold-level 13-word overlap for 929 of the 937 final prompts; eight three- or four-word XSTest prompts were intentionally left unassessed because lexical matches would be ambiguous. A synthetic positive-control prompt was detected at 100% coverage. The immutable result is `contamination-audit.json` in the comparison run. This narrows the FineWeb-adapter contamination concern; it cannot detect paraphrases or material learned by the original Qwen model.

Sharing uses an 85% process duty target and a measured admission check: at least 10 Humandescent samples, p95 ≤250 ms, mean device utilization ≤92%, and ≥1 GiB free VRAM. These are provisional operating gates, not a hard instantaneous GPU cap. Long CUDA kernels can still produce 100% readings. Shared-device energy includes other workloads.

Heretic++ uses a fixed CPU semantic entailment proxy, teacher-forced KL at four positions, and a constrained final selection. Its semantic calibration covers 24 independent labeled pairs and its capability gate uses 32 short development cases; these are limited smoke/selection checks, not broad guarantees. The independent final rubric and blind review remain separate from optimization.

## Validation completed on 2026-09-13

- 413 repository tests and nine Heretic plugin/workflow tests passed in the frozen preflight. These include supervisor restart/failure handling, exhausted-study finalization, and Heretic search telemetry integration.
- All 164 official canonical HumanEval+ solutions passed the pinned isolated evaluator; incorrect programs, timeout handling, and container isolation were checked.
- All 1,406 prepared prompts fit the 8K serving context; the largest prompt plus output allowance uses 7,660 tokens. Training and original tokenizer token IDs match.
- All 24 semantic calibration pairs passed the predeclared threshold checks.
- Native NLL arithmetic self-test, original BF16 CPU reference forward, and matched original conversion completed.
- An actual Humandescent zero-step CNN probe during training took approximately 33–35 ms across three measured requests.
- The frozen search-monitor lifecycle was also exercised during training: 35 GPU samples, two actual Humandescent tasks, 32.8 ms p95 task latency, 75.6% mean utilization, and 1.61 GiB minimum free VRAM. This bounded smoke shortened only the sample-count gate to two; production searches still require ten.

These validate implementation and preparation. Real CUDA sharing, Heretic interventions, full merged-model equivalence, quantization drift, and final benchmark results are still pending behind training.

The separate [12-case native Lab pilot](../../catalog/suites/qwen-comparison-pilot.yaml) remains useful for manual smoke checks; it is development-only and is not the frozen core.
