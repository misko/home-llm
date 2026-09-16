"""Read tokenizer/chat-template identities without loading model weights."""
import argparse
import json
import sys
from pathlib import Path

from llm_lab.comparison.contracts import atomic_json, fingerprint

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('source',type=Path);parser.add_argument('model',type=Path);parser.add_argument('output',type=Path)
args=parser.parse_args()
sys.path.insert(0,str(args.source/'gguf-py'))
from gguf import GGUFReader

reader=GGUFReader(args.model)
for required in ['tokenizer.chat_template','tokenizer.ggml.tokens','tokenizer.ggml.model']:
    if required not in reader.fields:raise ValueError('Required serving metadata is missing: '+required)
fields={}
for name in ['general.architecture','tokenizer.chat_template','tokenizer.ggml.model','tokenizer.ggml.pre',
             'tokenizer.ggml.tokens','tokenizer.ggml.token_type','tokenizer.ggml.bos_token_id','tokenizer.ggml.eos_token_id',
             'tokenizer.ggml.add_bos_token','tokenizer.ggml.add_eos_token']:
    value=reader.fields[name].contents() if name in reader.fields else None
    fields[name]=fingerprint(value)
atomic_json(args.output,{'fields':fields,'tokenizer_and_template_sha256':fingerprint(fields)})
print(args.output)
