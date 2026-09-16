import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from optuna.study import StudyDirection
from optuna.trial import TrialState

spec = importlib.util.spec_from_file_location('heretic_plus_test', Path(__file__).with_name('plus.py'))
plus = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plus
spec.loader.exec_module(plus)


def test_kl_identity_perturbation_and_alignment():
    original = torch.tensor([[1.,2.,3.],[2.,4.,1.]]).log_softmax(-1)
    assert plus.kl_from_logprobs(original, original) == pytest.approx(0, abs=1e-7)
    modified = torch.tensor([[3.,2.,1.],[1.,2.,4.]]).log_softmax(-1)
    assert plus.kl_from_logprobs(original, modified) > .1
    with pytest.raises(ValueError, match='aligned'):
        plus.kl_from_logprobs(original, modified[:1])


def test_teacher_forcing_uses_identical_reference_prefixes_and_no_padding():
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs['return_dict'] is False
            return [1,2,3]
        def encode(self, *args, **kwargs): return [4,5,6,7]
    class Model:
        def __init__(self): self.inputs = []
        def get_input_embeddings(self): return SimpleNamespace(weight=torch.empty(1))
        def __call__(self, **kwargs):
            self.inputs.append(kwargs['input_ids'].tolist()[0])
            assert torch.all(kwargs['attention_mask'] == 1)
            assert kwargs['logits_to_keep'] == 1
            return SimpleNamespace(logits=torch.zeros(1,1,10))
    model=Model()
    values=plus.position_logprobs(model,Tokenizer(),[], 'answer', 3, 32)
    assert model.inputs == [[1,2,3], [1,2,3,4], [1,2,3,4,5,6]]
    assert values.shape == (3,10)
    with pytest.raises(ValueError, match='without truncation'):
        plus.position_logprobs(model,Tokenizer(),[], 'answer', 3, 4)


def trial(number, values, regression):
    return SimpleNamespace(number=number, values=values, state=TrialState.COMPLETE,
        user_attrs={'scores':[{'name':'Capability regression','score':{'value':regression}}]})


def test_feasible_selection_ignores_infeasible_dominating_candidate():
    bad=trial(0,[0,0], .5)
    good=trial(1,[.2,.1], .02)
    worse=trial(2,[.3,.2], .01)
    assert plus.feasible_pareto([bad,good,worse], [StudyDirection.MINIMIZE]*2, .05) == [good]
    with pytest.raises(RuntimeError, match='No Heretic'):
        plus.feasible_pareto([bad], [StudyDirection.MINIMIZE]*2, .05)


def test_plugin_contracts_validate():
    for cls in [plus.MultiPositionKL, plus.SemanticTaskLoss, plus.CapabilityRegression]:
        cls.validate_contract()
        assert cls.get_settings_model() is not None
