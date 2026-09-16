"""Heretic++ scorer plugins for the pinned upstream plugin interface.

NLI scores answer/reference entailment on bounded English development tasks;
they are a calibrated semantic proxy, not a general-purpose truth judge.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from heretic.plugin import Context
from heretic.scorer import Score, Scorer
from heretic.utils import Prompt
from llm_lab.comparison.contracts import file_hash, load_cases
from llm_lab.comparison.scoring import score


def scalar(value: float) -> Score:
    return Score(value=value, rich_display=f"{value:.5f}", md_display=f"{value:.5f}")


def kl_from_logprobs(baseline: torch.Tensor, candidate: torch.Tensor) -> float:
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[0] == 0:
        raise ValueError("KL requires aligned nonempty position-by-vocabulary matrices")
    if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
        raise ValueError("KL received nonfinite log probabilities")
    return max(0.0, F.kl_div(candidate.float(), baseline.float(), log_target=True, reduction="batchmean").item())


def position_logprobs(model, tokenizer, messages, reference: str, max_positions: int, max_context: int) -> torch.Tensor:
    """Score identical teacher-forced prefixes, never independently generated tokens."""
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
    continuation = tokenizer.encode(reference, add_special_tokens=False)
    if not prefix or not continuation or len(prefix) + len(continuation) > max_context:
        raise ValueError("Reference must fit the configured KL window without truncation")
    positions = sorted(set(torch.linspace(0, len(continuation)-1, min(max_positions, len(continuation))).long().tolist()))
    rows = []
    device = model.get_input_embeddings().weight.device
    for position in positions:
        ids = torch.tensor([prefix + continuation[:position]], device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[:, -1, :]
        rows.append(F.log_softmax(logits.float(), dim=-1).cpu())
    return torch.cat(rows, dim=0)


class KLSettings(BaseModel):
    references_file: str
    references_sha256: str
    max_positions: int = Field(default=4, ge=2, le=16)
    max_context: int = Field(default=512, ge=32, le=8192)


class MultiPositionKL(Scorer):
    settings: KLSettings

    @property
    def score_name(self):
        return "Multi-position KL"

    @property
    def reproducible(self):
        return True

    def init(self, ctx: Context):
        path = Path(self.settings.references_file)
        if file_hash(path) != self.settings.references_sha256:
            raise ValueError("KL development references changed")
        self.references = json.loads(path.read_text())
        if not self.references or any(r["partition"] != "development" for r in self.references):
            raise ValueError("KL references must be development-only")
        self.baseline = self._obtain(ctx)

    def _obtain(self, ctx):
        return [position_logprobs(ctx._model.model, ctx._model.tokenizer, row["messages"], row["reference"], self.settings.max_positions, self.settings.max_context) for row in self.references]

    def get_score(self, ctx):
        current = self._obtain(ctx)
        # Equal weight per prompt; longer reference answers do not dominate.
        return scalar(sum(kl_from_logprobs(a, b) for a, b in zip(self.baseline, current, strict=True)) / len(current))

    def get_baseline_score(self, ctx):
        return scalar(0.0)


class CapabilitySettings(BaseModel):
    cases_file: str
    cases_sha256: str
    case_ids: list[str]


class CapabilityRegression(Scorer):
    settings: CapabilitySettings

    @property
    def score_name(self):
        return "Capability regression"

    @property
    def reproducible(self):
        return True

    def init(self, ctx):
        path = Path(self.settings.cases_file)
        if file_hash(path) != self.settings.cases_sha256:
            raise ValueError("Capability fixtures changed")
        selected = set(self.settings.case_ids)
        self.cases = [c for c in load_cases(path) if c.id in selected]
        if len(self.cases) != len(selected) or not self.cases or any(c.partition != "development" or c.tools or len(c.messages) != 1 for c in self.cases):
            raise ValueError("Capability constraints need selected single-turn development cases without tools")
        self.baseline = self._scores(ctx)

    def _scores(self, ctx):
        prompts = [Prompt(system=self.heretic_settings.system_prompt, user=c.messages[0]["content"]) for c in self.cases]
        responses = ctx.get_responses(prompts)
        values = {}
        for case, response in zip(self.cases, responses, strict=True):
            status, metrics, _ = score(case, response, [], "stop")
            if status != "scored":
                raise ValueError("Capability scorer requires deterministic judgments")
            values.setdefault(case.task, []).append(metrics["accuracy"])
        return {key: sum(v)/len(v) for key, v in values.items()}

    def get_score(self, ctx):
        current = self._scores(ctx)
        # Worst-family loss prevents gains on one task concealing another's damage.
        return scalar(max(0.0, *(self.baseline[task] - current[task] for task in self.baseline)))

    def get_baseline_score(self, ctx):
        return scalar(0.0)


class SemanticSettings(BaseModel):
    references_file: str
    references_sha256: str
    judge_path: str
    judge_manifest: str
    judge_manifest_sha256: str
    threshold: float = Field(default=0.7, gt=0, lt=1)


class SemanticTaskLoss(Scorer):
    settings: SemanticSettings

    @property
    def score_name(self):
        return "Semantic task loss"

    @property
    def reproducible(self):
        return True

    def init(self, ctx):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = Path(self.settings.references_file)
        if file_hash(path) != self.settings.references_sha256:
            raise ValueError("Semantic development references changed")
        self.references = json.loads(path.read_text())
        if not self.references or any(r["partition"] != "development" for r in self.references):
            raise ValueError("Semantic scorer cannot use final test prompts")
        manifest_path = Path(self.settings.judge_manifest)
        if file_hash(manifest_path) != self.settings.judge_manifest_sha256:
            raise ValueError('Semantic judge calibration changed')
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("calibration_passed") or manifest["threshold"] != self.settings.threshold:
            raise ValueError("Semantic judge requires passing calibration at the configured threshold")
        judge_path = Path(self.settings.judge_path)
        for name, expected in manifest["files"].items():
            if file_hash(judge_path / name) != expected:
                raise ValueError("Semantic judge files changed")
        self.tokenizer = AutoTokenizer.from_pretrained(judge_path, local_files_only=True)
        self.judge = AutoModelForSequenceClassification.from_pretrained(judge_path, local_files_only=True).to("cpu").eval()
        self.entailment_id = manifest["entailment_id"]

    def get_score(self, ctx):
        prompts = [Prompt(system=self.heretic_settings.system_prompt, user=r["messages"][-1]["content"]) for r in self.references]
        responses = ctx.get_responses(prompts)
        passed = []
        for row, response in zip(self.references, responses, strict=True):
            if not response.strip():
                passed.append(False)
                continue
            # Reject text beyond the judge window instead of silently judging only a prefix.
            inputs = self.tokenizer(response, row["reference"], return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] > 512:
                passed.append(False)
                continue
            words = response.lower().split()
            trigrams = list(zip(words, words[1:], words[2:]))
            repetitive = len(trigrams) > 12 and len(set(trigrams)) / len(trigrams) < .5
            with torch.inference_mode():
                probability = self.judge(**inputs).logits.softmax(-1)[0, self.entailment_id].item()
            passed.append(probability >= self.settings.threshold and not repetitive)
        return scalar(1 - sum(passed)/len(passed))


def feasible_pareto(trials, directions, tolerance: float):
    """Filter eligibility before Pareto selection; never return an infeasible winner."""
    from optuna.trial import TrialState
    from optuna.study import StudyDirection
    eligible = []
    for trial in trials:
        if trial.state != TrialState.COMPLETE:
            continue
        constraints = [s["score"]["value"] for s in trial.user_attrs.get("scores", []) if s["name"] == "Capability regression"]
        if len(constraints) == 1 and constraints[0] <= tolerance:
            eligible.append(trial)
    if not eligible:
        raise RuntimeError("No Heretic++ candidate satisfied the capability constraints")
    def normalized(trial):
        return [v if d == StudyDirection.MINIMIZE else -v for v, d in zip(trial.values, directions, strict=True)]
    def dominates(a, b):
        return all(x <= y for x, y in zip(a, b, strict=True)) and any(x < y for x, y in zip(a, b, strict=True))
    return [a for a in eligible if not any(dominates(normalized(b), normalized(a)) for b in eligible if b.number != a.number)]
