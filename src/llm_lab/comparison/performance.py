"""Counterbalanced performance blocks with fixed prompts and no prefix caching."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from llm_lab.catalog import Catalog
from llm_lab.paths import LabPaths
from llm_lab.runtime import RuntimeManager

from .contracts import Case, atomic_json, file_hash, fingerprint
from .humandescent import probe
from .runner import generate
from .sharing import process_duty_cycle


ARMS=['qwen','qwen-ft-fineweb','qwen-heretic','qwen-heretic-plus']


def orders():
    return [ARMS[i:]+ARMS[:i] for i in range(len(ARMS))]


def scenarios():
    records='\n'.join(f'entry{i}: value{i}' for i in range(512))
    return [
        Case(id='performance/short',group='performance/short',task='performance',partition='development',messages=[{'role':'user','content':'Return only READY.'}],max_tokens=16,scorer='exact',expected='READY'),
        Case(id='performance/prefill',group='performance/prefill',task='performance',partition='development',messages=[{'role':'user','content':records+'\nReturn the value of entry256 only.'}],max_tokens=16,scorer='exact',expected='value256'),
        Case(id='performance/decode',group='performance/decode',task='performance',partition='development',messages=[{'role':'user','content':'Explain binary search in detail, with examples and its computational complexity. Write at least 400 words.'}],max_tokens=256,scorer='exact',expected='performance-only'),
    ]


def run(root: Path):
    config=json.loads((root/'execution.json').read_text())
    records=json.loads((root/'artifacts.json').read_text())
    if set(records)!=set(ARMS):raise RuntimeError('All selected artifacts are required for counterbalanced performance testing')
    sharing=json.loads((root/'sharing-probe.json').read_text())
    if not sharing['passed'] or sharing['duty_cycle']!=config['inference_duty_cycle']:raise RuntimeError('GPU sharing needs validation')
    paths=LabPaths.discover(repo_root=config['repo_root'],data_root=config['data_root'])
    catalog=Catalog.load(paths.catalog_root)
    directory=root/'performance';directory.mkdir(exist_ok=True)
    sampling={'temperature':0,'top_p':1,'seed':20260913,'cache_prompt':False}
    for block,order in enumerate(orders()):
        for arm in order:
            target=directory/f'block-{block}-{arm}.json'
            identity=fingerprint({'block':block,'order':order,'arm':arm,'artifact':records[arm],
                                  'sampling':sampling,'scenarios':[c.model_dump() for c in scenarios()],
                                  'source':file_hash(Path(__file__)),'duty_cycle':config['inference_duty_cycle']})
            if target.exists():
                if json.loads(target.read_text())['identity']!=identity:raise ValueError('Performance protocol changed')
                continue
            record=records[arm];deployment=catalog.get_deployment(record['deployment_id'])
            manager=RuntimeManager(paths)
            manager.activate(deployment,record['view'],runtime_lock=catalog.get_runtime_lock(deployment.runtime_lock_id))
            try:
                with manager.benchmark_lease(check_health=True) as status:
                    if not status.ready:raise RuntimeError('Performance deployment is not ready')
                    with process_duty_cycle(status.state.launch.pid,config['inference_duty_cycle']), probe(config,directory/f'humandescent-{block}-{arm}.jsonl',interval_seconds=1):
                        async def measure():
                            rows=[]
                            async with httpx.AsyncClient(base_url=status.state.base_url,timeout=300) as client:
                                await generate(client,status.state.public_alias,scenarios()[0],sampling)  # excluded warmup
                                tasks=scenarios();tasks=tasks[block%3:]+tasks[:block%3]
                                for case in tasks:
                                    for repetition in range(3):
                                        result=await generate(client,status.state.public_alias,case,sampling)
                                        rows.append({'scenario':case.id,'repetition':repetition,**result})
                            return rows
                        rows=asyncio.run(measure())
            finally:
                manager.stop()
            atomic_json(target,{'identity':identity,'arm':arm,'block':block,'order':order,'rows':rows})
    return {'blocks':4,'requests_per_arm':36,'warmups_excluded':True,'directory':str(directory)}
