"""Merge a completed Heretic adapter into its exact FineWeb parent on CPU."""
import json
import os
import sys
from pathlib import Path

from merge_input import merge_with_validation, preserve_unloaded_tensors
from llm_lab.comparison.contracts import atomic_json, file_hash, fingerprint
from llm_lab.runtime import ExclusiveGpuLock


def main():
    config=json.loads(Path(sys.argv[1]).read_text())
    root=Path(config['run_dir']);parent=Path(config['input_model'])
    result=json.loads((root/'result.json').read_text())
    adapter=Path(result['adapter'])
    if Path(result['input_model'])!=parent or file_hash(adapter/'adapter_model.safetensors')!=result['adapter_sha256']:
        raise ValueError('Heretic adapter or parent identity differs')
    parent_manifest=json.loads((parent/'fineweb-input-manifest.json').read_text())
    for name,expected in parent_manifest['files'].items():
        if file_hash(parent/name)!=expected:raise ValueError('FineWeb parent changed: '+name)
    provenance={'input_fingerprint':parent_manifest['fingerprint'],'adapter_sha256':result['adapter_sha256'],
                'adapter_config_sha256':file_hash(adapter/'adapter_config.json'),'search_fingerprint':result['fingerprint']}
    destination=root/'heretic-merged-bf16'
    with ExclusiveGpuLock(root/'variant-merge.lock',timeout_seconds=0):
        if destination.exists():
            previous=json.loads((destination/'merge-manifest.json').read_text())
            if previous['provenance']!=provenance:raise ValueError('Existing merged variant differs')
            for name,expected in previous['files'].items():
                if file_hash(destination/name)!=expected:raise ValueError('Merged variant changed')
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        torch.set_num_threads(4)
        base=AutoModelForImageTextToText.from_pretrained(parent,dtype=torch.bfloat16,device_map='cpu',local_files_only=True,trust_remote_code=False)
        if any(p.device.type!='cpu' for p in base.parameters()):raise ValueError('Merge must remain on CPU')
        processor=AutoProcessor.from_pretrained(parent,local_files_only=True,trust_remote_code=False)
        merged,validation=merge_with_validation(base,adapter,processor.tokenizer)
        temporary=destination.with_name(destination.name+'.partial')
        if temporary.exists():raise ValueError('Incomplete merge exists; inspect before retrying')
        merged.save_pretrained(temporary,safe_serialization=True,max_shard_size='4GB')
        preserved=preserve_unloaded_tensors(parent,temporary)
        processor.save_pretrained(temporary)
        atomic_json(temporary/'merge-manifest.json',{'provenance':provenance,'validation':validation,
                    'preserved_parent_tensors':preserved,
                    'files':{p.name:file_hash(p) for p in temporary.iterdir() if p.is_file()}})
        for p in temporary.iterdir():
            if p.is_file():p.chmod(0o444)
        os.replace(temporary,destination)


if __name__=='__main__':main()
