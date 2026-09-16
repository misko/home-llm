"""CPU validation of the frozen expanded tests against their canonical solutions."""
import argparse
import json
from pathlib import Path

from llm_lab.comparison.code_eval import evaluate
from llm_lab.comparison.contracts import atomic_json, file_hash

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path)
parser.add_argument('image')
args = parser.parse_args()
path = args.root / 'sources/HumanEvalPlus-v0.1.10.jsonl'
output = args.root / 'code-evaluator-validation.json'
results = []
for row in map(json.loads, path.read_text().splitlines()):
    result = evaluate(row['prompt'] + row['canonical_solution'], row['test'], row['entry_point'], args.image, problem=row)
    results.append({'task_id':row['task_id'], **result})
    atomic_json(output, {'image':args.image, 'source_sha256':file_hash(path), 'results':results,
                        'complete':len(results) == 164, 'passed':all(r['passed'] for r in results)})
    print(row['task_id'], result['passed'], result['reason'], flush=True)
    if not result['passed']:
        raise SystemExit(f"Canonical solution failed: {result}")
