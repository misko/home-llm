import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from run import can_finalize_exhausted_study, completed_checkpoint
from merge_input import merge_adapter, preserve_unloaded_tensors
from monitoring import summarize_search
from common import atomic_json, digest


def test_gate_requires_training_and_export_then_validates_checkpoint(tmp_path):
    config = {'training_run':str(tmp_path)}
    with pytest.raises(RuntimeError, match='both finish'):
        completed_checkpoint(config)
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    weights = checkpoint / 'adapter_model.safetensors'; weights.write_bytes(b'fixture')
    atomic_json(checkpoint / 'manifest.json', {'state':{'step':7}, 'files':{weights.name:digest(weights)}})
    atomic_json(tmp_path / 'trained.json', {'step':7, 'checkpoint':str(checkpoint)})
    with pytest.raises(RuntimeError, match='both finish'):
        completed_checkpoint(config)
    atomic_json(tmp_path / 'ready.json', {'phase':'ready'})
    assert completed_checkpoint(config) == checkpoint
    weights.write_bytes(b'corruption')
    with pytest.raises(ValueError, match='Invalid checkpoint'):
        completed_checkpoint(config)


def test_exhausted_study_can_only_finalize_a_completed_trial():
    state = lambda name: SimpleNamespace(state=SimpleNamespace(name=name))
    assert can_finalize_exhausted_study([state('FAIL'), state('COMPLETE')])
    assert not can_finalize_exhausted_study([state('FAIL'), state('PRUNED')])
    assert not can_finalize_exhausted_study([])


def test_cpu_merge_keeps_trained_adapter_behavior(tmp_path):
    torch.manual_seed(42)
    config = Qwen2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=64, max_position_embeddings=64)
    parent = Qwen2ForCausalLM(config).eval()
    adapted = get_peft_model(copy.deepcopy(parent), LoraConfig(
        r=2, lora_alpha=2, target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM')).eval()
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if 'lora_B' in name: parameter.fill_(0.15)
    tokens = torch.tensor([[1,2,3,4]])
    with torch.no_grad():
        expected = adapted(tokens).logits
        original = parent(tokens).logits
    assert not torch.allclose(expected, original)
    adapted.save_pretrained(tmp_path / 'adapter')
    merged = merge_adapter(parent, tmp_path / 'adapter').eval()
    with torch.no_grad(): actual = merged(tokens).logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert not any('lora_' in name for name, _ in merged.named_parameters())


def test_merge_preserves_parent_tensors_not_instantiated_by_model(tmp_path):
    from safetensors.torch import load_file, save_file

    parent = tmp_path / 'parent'; parent.mkdir()
    destination = tmp_path / 'merged'; destination.mkdir()
    kept = torch.arange(4, dtype=torch.bfloat16)
    omitted = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    save_file({'model.kept': kept, 'mtp.omitted': omitted}, parent / 'parent.safetensors')
    atomic_json(parent / 'model.safetensors.index.json', {
        'metadata': {'total_size': kept.numel() * kept.element_size() + omitted.numel() * omitted.element_size()},
        'weight_map': {'model.kept': 'parent.safetensors', 'mtp.omitted': 'parent.safetensors'},
    })
    save_file({'model.kept': kept}, destination / 'model.safetensors')
    atomic_json(destination / 'model.safetensors.index.json', {
        'metadata': {'total_size': kept.numel() * kept.element_size()},
        'weight_map': {'model.kept': 'model.safetensors'},
    })

    assert preserve_unloaded_tensors(parent, destination, max_shard_bytes=8) == ['mtp.omitted']
    index = json.loads((destination / 'model.safetensors.index.json').read_text())
    shard = destination / index['weight_map']['mtp.omitted']
    torch.testing.assert_close(load_file(shard)['mtp.omitted'], omitted)
    assert index['metadata']['total_size'] == (kept.numel() + omitted.numel()) * kept.element_size()
    assert preserve_unloaded_tensors(parent, destination, max_shard_bytes=8) == []


def test_search_monitor_summary_integrates_sessions_and_applies_policy(tmp_path):
    samples = [
        {'available':True, 'index':0, 'timestamp':'2026-09-13T00:00:00Z',
         'power_draw_w':100, 'gpu_utilization_percent':80, 'memory_total_mib':24000, 'memory_used_mib':20000},
        {'available':True, 'index':0, 'timestamp':'2026-09-13T00:00:02Z',
         'power_draw_w':200, 'gpu_utilization_percent':90, 'memory_total_mib':24000, 'memory_used_mib':21000},
    ]
    (tmp_path/'telemetry-search-0001.json').write_text(json.dumps(samples))
    with (tmp_path/'humandescent-search-0001.jsonl').open('w') as stream:
        for index in range(10):
            stream.write(json.dumps({'latency_seconds':.03+index/10000})+'\n')
    policy={'minimum_humandescent_samples':10, 'max_humandescent_p95_seconds':.25,
            'max_mean_gpu_utilization_percent':92, 'minimum_free_gpu_mib':1024}
    result=summarize_search(tmp_path,policy)
    assert result['passed']
    assert result['gpu_energy_joules_shared']==300
    assert result['gpu_energy_observed_seconds']==2
    assert result['minimum_free_gpu_mib']==3000
