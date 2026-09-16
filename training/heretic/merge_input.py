"""CPU-only merge of the final FineWeb adapter into the pinned official parent."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import atomic_json, check_space, digest, identity
from run import completed_checkpoint, validate_install
from llm_lab.runtime import ExclusiveGpuLock


def merge_adapter(base, checkpoint):
    from peft import PeftModel
    return PeftModel.from_pretrained(base, checkpoint, is_trainable=False).merge_and_unload(safe_merge=True)


def preserve_unloaded_tensors(parent: Path, destination: Path, max_shard_bytes: int = 4_000_000_000):
    """Copy parent tensors omitted by the Transformers inference class.

    Qwen3.5 checkpoints contain an auxiliary ``mtp`` prediction layer.  The
    inference model class does not instantiate that layer, so a normal
    ``save_pretrained`` silently drops its tensors even though llama.cpp uses
    them when converting and loading the model.  LoRA only changes tensors
    instantiated by the model, making a byte-for-byte copy of absent parent
    tensors the correct merged result.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    parent_index_path = parent / 'model.safetensors.index.json'
    destination_index_path = destination / 'model.safetensors.index.json'
    if not parent_index_path.is_file() or not destination_index_path.is_file():
        return []
    parent_index = json.loads(parent_index_path.read_text())
    destination_index = json.loads(destination_index_path.read_text())
    missing = sorted(set(parent_index['weight_map']) - set(destination_index['weight_map']))
    if not missing:
        return []

    groups = []
    current = []
    current_bytes = 0
    tensor_bytes = {}
    for name in missing:
        source = parent / parent_index['weight_map'][name]
        with safe_open(source, framework='pt', device='cpu') as handle:
            tensor = handle.get_tensor(name)
        size = tensor.numel() * tensor.element_size()
        tensor_bytes[name] = size
        if current and current_bytes + size > max_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append((name, tensor))
        current_bytes += size
    if current:
        groups.append(current)

    for index, group in enumerate(groups, 1):
        shard = f'model-preserved-{index:05d}-of-{len(groups):05d}.safetensors'
        save_file({name: tensor.contiguous() for name, tensor in group}, destination / shard,
                  metadata={'format': 'pt'})
        for name, _ in group:
            destination_index['weight_map'][name] = shard
    metadata = destination_index.setdefault('metadata', {})
    metadata['total_size'] = int(metadata.get('total_size', 0)) + sum(tensor_bytes.values())
    atomic_json(destination_index_path, destination_index)
    return missing


def merge_with_validation(base, checkpoint, tokenizer):
    """Check full-model next-token distributions before and after the CPU merge."""
    import torch
    from peft import PeftModel
    tuned = PeftModel.from_pretrained(base, checkpoint, is_trainable=False).eval()
    messages = [{'role':'user','content':'What is two plus two? Answer briefly.'}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                         tokenize=True, return_dict=False, return_tensors='pt')
    with torch.inference_mode():
        before = tuned(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float().cpu()
        merged = tuned.merge_and_unload(safe_merge=True)
        after = merged(input_ids=ids, use_cache=False, logits_to_keep=1).logits.float().cpu()
    log_before = before.log_softmax(-1)
    log_after = after.log_softmax(-1)
    kl = torch.nn.functional.kl_div(log_after, log_before, reduction='batchmean', log_target=True).item()
    mean_error = (before-after).abs().mean().item()
    if not torch.isfinite(after).all() or kl > .01 or mean_error > .1:
        raise RuntimeError(f'Full-model merge equivalence probe failed: KL={kl}, mean error={mean_error}')
    return merged, {'teacher_forced_next_token_kl':max(0.,kl),'mean_absolute_logit_error':mean_error,
                    'kl_tolerance':.01,'mean_error_tolerance':.1,'prompt_tokens':ids.shape[-1],
                    'note':'Single development prompt; verifies merge behavior, not broad capability.'}


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    validate_install(config)
    checkpoint = completed_checkpoint(config)
    root = Path(config['run_dir']); root.mkdir(parents=True, exist_ok=True)
    with ExclusiveGpuLock(root / 'input-merge.lock', timeout_seconds=0):
        training = json.loads((Path(config['training_run']) / 'config.json').read_text())
        if training['base_revision'] != config['base_revision']:
            raise RuntimeError('FineWeb and Heretic parent revisions differ.')
        parent = Path(config['base_path'])
        parent_index = json.loads((parent / 'model.safetensors.index.json').read_text())
        parent_hashes = {name: digest(parent / name) for name in sorted(set(parent_index['weight_map'].values()))}
        provenance = {'base_revision': config['base_revision'], 'base_shards': parent_hashes,
            'fineweb_checkpoint': str(checkpoint), 'adapter_sha256': digest(checkpoint / 'adapter_model.safetensors'),
            'adapter_config_sha256': digest(checkpoint / 'adapter_config.json')}
        fingerprint = identity(provenance)
        destination = Path(config['input_model'])
        if destination.exists():
            manifest = json.loads((destination / 'fineweb-input-manifest.json').read_text())
            if manifest['fingerprint'] != fingerprint:
                raise RuntimeError('Existing Heretic input came from a different checkpoint.')
            for name, expected in manifest['files'].items():
                if digest(destination / name) != expected:
                    raise RuntimeError(f'Merged input hash differs: {name}')
            print('Verified previously merged FineWeb input', flush=True)
            return
        check_space(Path(config['data_root']), 120_000_000_000)
        temporary = destination.with_name(destination.name + '.partial')
        if temporary.exists():
            raise RuntimeError(f'Incomplete prior merge at {temporary}; inspect before retrying.')
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        torch.set_num_threads(4)
        print('Loading pinned BF16 parent on CPU and merging final FineWeb adapter', flush=True)
        base = AutoModelForImageTextToText.from_pretrained(parent, dtype=torch.bfloat16,
            device_map='cpu', local_files_only=True, trust_remote_code=False)
        if any(p.device.type != 'cpu' for p in base.parameters()):
            raise RuntimeError('Input merge must stay on CPU.')
        processor = AutoProcessor.from_pretrained(parent, local_files_only=True, trust_remote_code=False)
        merged, merge_validation = merge_with_validation(base, checkpoint, processor.tokenizer)
        merged.save_pretrained(temporary, safe_serialization=True, max_shard_size='4GB')
        preserved = preserve_unloaded_tensors(parent, temporary)
        processor.save_pretrained(temporary)
        atomic_json(temporary / 'fineweb-input-manifest.json', {
            **provenance, 'fingerprint': fingerprint, 'merge_validation':merge_validation,
            'preserved_parent_tensors': preserved,
            'files': {p.name: digest(p) for p in temporary.iterdir() if p.is_file()},
        })
        for p in temporary.iterdir():
            if p.is_file(): p.chmod(0o444)
        os.replace(temporary, destination)
        print(f'Heretic input ready: {destination}', flush=True)


if __name__ == '__main__':
    main()
