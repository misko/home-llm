"""Capture a bounded full-precision reference distribution using CPU only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from llm_lab.comparison.contracts import atomic_json, file_hash, fingerprint

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path);parser.add_argument('arm');parser.add_argument('model',type=Path)
args=parser.parse_args()
torch.set_num_threads(4)
destination=args.root/'references'/args.arm;destination.mkdir(parents=True,exist_ok=True)
identity={p.name:file_hash(p) for p in args.model.glob('*.safetensors')}
if not identity:raise ValueError('Full-precision model shards are missing')
manifest=destination/'manifest.json'
if manifest.exists():
    value=json.loads(manifest.read_text())
    if value['source_shards']!=identity:raise ValueError('Reference parent changed')
    for name,expected in value['files'].items():
        if file_hash(destination/name)!=expected:raise ValueError('Reference data changed')
    print('Verified existing BF16 reference',flush=True)
else:
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    messages=[{'role':'user','content':'What is two plus two? Answer briefly.'}]
    ids=tok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,tokenize=True,return_dict=False)
    print('Loading BF16 reference on CPU',flush=True)
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cpu',local_files_only=True,trust_remote_code=False).eval()
    if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('Reference must stay on CPU')
    with torch.inference_mode():
        logits=model(input_ids=torch.tensor([ids]),use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu().numpy()
    if not np.isfinite(logits).all():raise ValueError('Nonfinite reference logits')
    np.save(destination/'logits.npy',logits)
    np.asarray(ids,dtype='<u4').tofile(destination/'tokens.uint32')
    atomic_json(manifest,{'source_shards':identity,'messages':messages,'prompt_tokens':len(ids),
                         'tokenizer_vocab_sha256':fingerprint(tok.get_vocab()),'dtype':'bfloat16',
                         'files':{name:file_hash(destination/name) for name in ['logits.npy','tokens.uint32']}})
    print('BF16 reference ready',flush=True)
