"""Publish final evaluation/availability metrics to the same W&B run."""
import json
import sys
from pathlib import Path
import wandb

config=json.loads(Path(sys.argv[1]).read_text());run=Path(config['run_dir'])
ready=json.loads((run/'ready.json').read_text())
w=wandb.init(entity=config['wandb']['entity'],project=config['wandb']['project'],
    id=config['wandb']['id'],resume='allow',dir=str(run),settings=wandb.Settings(disable_git=True))
w.summary.update({'phase':'ready','llm_lab_alias':ready['public_alias'],
    'llm_lab_deployment':ready['deployment_id'],
    'original_smoke_pass_rate':ready['baseline'].get('pass_rate'),
    'candidate_smoke_pass_rate':ready['candidate'].get('pass_rate')})
w.finish()
