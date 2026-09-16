"""Run one registered comparison arm under the existing Lab GPU lease."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import subprocess
from pathlib import Path

import httpx

from llm_lab.catalog import Catalog
from llm_lab.paths import LabPaths
from llm_lab.runtime import RuntimeManager
from llm_lab.storage import ArtifactStore
from llm_lab.telemetry import NvidiaTelemetrySampler

from .contracts import atomic_json, file_hash, fingerprint, load_cases
from .runner import run_cases
from .sharing import process_duty_cycle
from .humandescent import probe, read_samples


def evaluate_arm(root: Path, arm: str, *, partition='final', limit=None, retry_infrastructure=False):
    config=json.loads((root/'execution.json').read_text())
    training=Path(config['training_run'])
    if not (training/'ready.json').exists() or not (training/'trained.json').exists():
        raise RuntimeError('FineWeb training and export must finish before GPU evaluation')
    sharing=json.loads((root/'sharing-probe.json').read_text())
    if not sharing['passed'] or sharing['duty_cycle']!=config['inference_duty_cycle']:
        raise RuntimeError('Validate inference sharing at the current duty cycle before evaluation')
    if partition == 'final':
        profiles=json.loads((root/'search-profiles.json').read_text())
        for key in ['upstream_launcher','plus_launcher']:
            search=json.loads(Path(profiles[key]).read_text())
            if not (Path(search['run_dir'])/'result.json').exists():
                raise RuntimeError('Freeze both Heretic selections before opening the final test')
        if limit is not None:
            raise ValueError('Final evaluation cannot silently shrink the frozen workload')
    records=json.loads((root/'artifacts.json').read_text())
    record=records[arm]
    if record['serving_metadata']!=records['qwen']['serving_metadata']:
        raise ValueError('Tokenizer or chat template differs from the matched baseline')
    paths=LabPaths.discover(repo_root=config['repo_root'],data_root=config['data_root'])
    catalog=Catalog.load(paths.catalog_root)
    deployment=catalog.get_deployment(record['deployment_id'])
    runtime=catalog.get_runtime_lock(deployment.runtime_lock_id)
    with ArtifactStore(paths) as store:
        store.verify(deployment.artifact_id,verify_view=True)
    cases=[c for c in load_cases(root/'cases.jsonl') if c.partition == partition]
    if limit is not None:
        if limit < 1:
            raise ValueError('Development limit must be positive')
        cases=sorted([c for c in cases if c.task in {'grounded-local','tool-json-local'} and not c.tools], key=lambda c:c.id)[:limit]
    definition=deployment.model_dump(mode='json')
    serving={key:definition[key] for key in ['context_size','parallel','gpu_layers','flash_attention','kv_cache_type_k','kv_cache_type_v','reasoning_mode','extra_args']}
    protocol={'sampling':{'temperature':0,'top_p':1,'seed':20260913,'cache_prompt':False},'serving':serving,
              'runtime_commit':runtime.commit,'runtime_sha256':runtime.binary_sha256,
              'dataset_sha256':file_hash(root/'cases.jsonl'),'partition':partition,'development_limit':limit,
              'inference_duty_cycle':config['inference_duty_cycle'], 'scheduler_period_seconds':.2,
              'environment_lock_sha256':file_hash(Path(config['environment_lock'])),
              'implementation':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
              'humandescent':config['humandescent'],
              'humandescent_probe_sha256':file_hash(Path(config.get('code_root',config['repo_root']))/'benchmarks/qwen-comparison/humandescent-probe.py'),
              'artifact':record}
    run_root=root/'arms'/arm/partition
    run_root.mkdir(parents=True,exist_ok=True)
    session=len(list(run_root.glob('humandescent-*.jsonl')))+1
    manager=RuntimeManager(paths)
    # The comparison owns its activation; training and other Lab runs are protected
    # by the manager's existing transition/benchmark locks.
    manager.activate(deployment,record['view'],runtime_lock=runtime)
    tracking=None
    try:
        with manager.benchmark_lease(check_health=True) as status:
            if not status.ready or status.state.deployment_id != deployment.id:
                raise RuntimeError('The intended comparison deployment is not ready')
            state=status.state
            if state.launch.pid is None:
                raise RuntimeError('Comparison sharing currently requires a local process backend')
            with process_duty_cycle(state.launch.pid,config['inference_duty_cycle']), probe(config,run_root/f'humandescent-{session:04d}.jsonl'):
                import wandb
                tracking=wandb.init(**config['wandb'],id='qcomp-'+fingerprint([arm,protocol])[:20],
                    name=arm+'-'+partition,resume='allow',dir=str(run_root),
                    config={'arm':arm,'protocol_sha256':fingerprint(protocol),'protocol':protocol},
                    settings=wandb.Settings(disable_git=True))
                case_tasks={c.id:c.task for c in cases}
                def log(row):
                    data={'evaluation/case_id':row.case_id,'evaluation/task':case_tasks[row.case_id],
                          'evaluation/status':row.status,'performance/latency_seconds':row.latency_seconds}
                    data.update({'score/'+key:value for key,value in row.metrics.items()})
                    if row.ttft_seconds is not None:data['performance/ttft_seconds']=row.ttft_seconds
                    data.update({'backend/'+key:value for key,value in row.server_timings.items() if isinstance(value,(int,float))})
                    hum=[r for r in read_samples(run_root) if 'latency_seconds' in r]
                    if hum:data['humandescent/task_latency_seconds']=hum[-1]['latency_seconds']
                    tracking.log(data)
                async def execute():
                    sampler=NvidiaTelemetrySampler(interval_seconds=1)
                    await sampler.start()
                    try:
                        async with httpx.AsyncClient(base_url=state.base_url, timeout=httpx.Timeout(600,connect=10)) as client:
                            return await run_cases(run_root,arm,cases,protocol,client,state.public_alias,
                                retry_infrastructure=retry_infrastructure,on_result=log)
                    finally:
                        samples=await sampler.stop()
                        telemetry=[sample.to_dict() for sample in samples]
                        # Preserve telemetry across resumed sessions.
                        session=len(list(run_root.glob('telemetry-*.json')))+1
                        atomic_json(run_root/f'telemetry-{session:04d}.json',telemetry)
                rows=asyncio.run(execute())
                summary={'completed':len(rows),'scored':sum(r.status=='scored' for r in rows),
                         'pending_judgment':sum(r.status=='pending_judgment' for r in rows),
                         'infrastructure_errors':sum(r.status=='infrastructure_error' for r in rows)}
                tracking.summary.update(summary)
                atomic_json(run_root/'tracking.json',{'url':tracking.url,**summary})
                if summary['infrastructure_errors']:
                    raise RuntimeError('Saved infrastructure errors require an explicit retry')
                return summary
    finally:
        import sys
        if tracking is not None:tracking.finish(exit_code=1 if sys.exc_info()[0] else 0)
        manager.stop()
