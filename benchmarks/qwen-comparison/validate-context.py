"""Check actual chat-template token lengths and FineWeb/original vocabulary parity."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from llm_lab.comparison.contracts import atomic_json, file_hash, fingerprint, load_cases

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
root=args.root;config=json.loads((root/'execution.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(config['base_path'],local_files_only=True)
training=json.loads((Path(config['training_run'])/'config.json').read_text())
training_tokenizer=AutoTokenizer.from_pretrained(training['model_path'],local_files_only=True)
if tokenizer.get_vocab()!=training_tokenizer.get_vocab() or tokenizer.eos_token_id!=training_tokenizer.eos_token_id:
    raise ValueError('FineWeb token IDs do not match the original model tokenizer')
rows=[]
for case in load_cases(root/'cases.jsonl'):
    ids=tokenizer.apply_chat_template(case.messages,tools=case.tools or None,enable_thinking=False,
        tokenize=True,return_dict=False,add_generation_prompt=True)
    if not isinstance(ids,list) or not all(isinstance(x,int) for x in ids):raise ValueError('Tokenizer returned something other than token IDs')
    if len(ids)+case.max_tokens>8192:raise ValueError(f'{case.id} exceeds the serving context')
    rows.append({'case_id':case.id,'prompt_tokens':len(ids),'max_output_tokens':case.max_tokens})
result={'cases':len(rows),'passed':True,'cases_sha256':file_hash(root/'cases.jsonl'),
        'tokenizer_vocab_sha256':fingerprint(tokenizer.get_vocab()),'training_tokenizer_parity':True,'rows':rows}
atomic_json(root/'context-validation.json',result)
print(f'Validated {len(rows)} cases; longest prompt: {max(r["prompt_tokens"] for r in rows)} tokens.')
