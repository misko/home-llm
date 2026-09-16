"""Create a source snapshot for the background pipeline; run validation afterward."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from llm_lab.comparison.contracts import atomic_json, file_hash

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path);parser.add_argument('--version',default='v1');args=parser.parse_args()
root=args.root;config=json.loads((root/'execution.json').read_text());repo=Path(config['repo_root'])
destination=root/('source-'+args.version)
if destination.exists():raise ValueError('Source snapshot already exists; never overwrite a running version')
files=list((repo/'src/llm_lab').rglob('*.py'))
files += [p for p in (repo/'src/llm_lab/web_dist').rglob('*') if p.is_file()]
files+=list((repo/'training/heretic').glob('*.py'))
files += [repo/'training/common.py',repo/'training/throttle.py',repo/'training/heretic/requirements.lock.txt']
files += [p for p in (repo/'benchmarks/qwen-comparison').iterdir() if p.suffix in {'.py','.cpp','.json'}]
temporary=destination.with_name(destination.name+'.partial');temporary.mkdir(parents=True)
hashes={}
for path in files:
    relative=path.relative_to(repo);target=temporary/relative
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);target.chmod(0o444)
    hashes[str(relative)]=file_hash(target)
os.replace(temporary,destination)
atomic_json(root/'source-lock.json',{'version':args.version,'origin_repo':str(repo),
    'origin_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    'files':hashes})
config.update(code_root=str(destination),environment_lock=str(destination/'training/heretic/requirements.lock.txt'))
atomic_json(root/'execution.json',config)
profiles=json.loads((root/'search-profiles.json').read_text())
upstream=Path(profiles['upstream_launcher']);launcher=json.loads(upstream.read_text())
launcher.update(code_root=str(destination),environment_lock=config['environment_lock'])
atomic_json(upstream,launcher)
subprocess.run([config['control_python'],str(destination/'training/heretic/prepare_comparison.py'),str(upstream),str(root)],
               env={**os.environ,'PYTHONPATH':str(destination/'src')},check=True)
print(destination)
