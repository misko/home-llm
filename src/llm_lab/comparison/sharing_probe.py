"""Validate actual GPU sharing and Humandescent latency before long jobs."""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from llm_lab.catalog import Catalog
from llm_lab.paths import LabPaths
from llm_lab.runtime import RuntimeManager
from llm_lab.telemetry import NvidiaTelemetrySampler

from .contracts import Case, atomic_json
from .humandescent import probe, read_samples
from .runner import generate
from .sharing import process_duty_cycle


def check(root: Path):
    config=json.loads((root/'execution.json').read_text())
    if not (Path(config['training_run'])/'ready.json').exists():
        raise RuntimeError('Wait for FineWeb export before probing inference sharing')
    record=json.loads((root/'artifacts.json').read_text())['qwen']
    paths=LabPaths.discover(repo_root=config['repo_root'],data_root=config['data_root'])
    catalog=Catalog.load(paths.catalog_root);deployment=catalog.get_deployment(record['deployment_id'])
    manager=RuntimeManager(paths)
    directory=root/'sharing-probe';directory.mkdir(exist_ok=True)
    index=len(list(directory.glob('humandescent-*.jsonl')))+1
    manager.activate(deployment,record['view'],runtime_lock=catalog.get_runtime_lock(deployment.runtime_lock_id))
    started=time.time()
    try:
        with manager.benchmark_lease(check_health=True) as status:
            if not status.ready:raise RuntimeError('Sharing probe model is not ready')
            with process_duty_cycle(status.state.launch.pid,config['inference_duty_cycle']), probe(config,directory/f'humandescent-{index:04d}.jsonl',interval_seconds=.5):
                async def workload():
                    sampler=NvidiaTelemetrySampler(interval_seconds=.5)
                    await sampler.start()
                    responses=[]
                    try:
                        async with httpx.AsyncClient(base_url=status.state.base_url,timeout=300) as client:
                            for i in range(3):
                                case=Case(id=f'sharing/{i}',group=f'sharing/{i}',task='sharing',partition='development',
                                    messages=[{'role':'user','content':'Explain how a hash table works, including examples and collision handling. Write at least 600 words.'}],
                                    max_tokens=512,scorer='exact',expected='load-probe-only')
                                responses.append(await generate(client,status.state.public_alias,case,{'temperature':0,'seed':20260913,'cache_prompt':False}))
                    finally:
                        samples=await sampler.stop()
                        atomic_json(directory/f'telemetry-{index:04d}.json',[s.to_dict() for s in samples])
                    return samples,responses
                samples,responses=asyncio.run(workload())
    finally:
        manager.stop()
    hum=sorted(r['latency_seconds'] for r in read_samples(directory) if 'latency_seconds' in r and r['timestamp']>=started)
    usable=[s for s in samples if s.available and s.index==0]
    utilizations=[s.gpu_utilization_percent for s in usable if s.gpu_utilization_percent is not None]
    memory=[s.memory_total_mib-s.memory_used_mib for s in usable if s.memory_total_mib is not None and s.memory_used_mib is not None]
    policy=config['sharing_policy']
    p95=hum[min(len(hum)-1,int(len(hum)*.95))] if hum else None
    mean=statistics.mean(utilizations) if utilizations else None
    passed=(len(hum)>=policy['minimum_humandescent_samples'] and p95<=policy['max_humandescent_p95_seconds']
            and mean is not None and mean<=policy['max_mean_gpu_utilization_percent']
            and memory and min(memory)>=policy['minimum_free_gpu_mib'])
    result={'passed':bool(passed),'duty_cycle':config['inference_duty_cycle'],'policy':policy,
            'humandescent_samples':len(hum),'humandescent_p95_seconds':p95,
            'mean_gpu_utilization_percent':mean,'minimum_free_gpu_mib':min(memory) if memory else None,
            'response_tokens':[r['usage'].get('completion_tokens') for r in responses],
            'note':'Cooperative process scheduling; queued kernels may extend into a pause. No hard utilization cap.'}
    atomic_json(root/'sharing-probe.json',result)
    if not passed:raise RuntimeError('Inference sharing did not meet the configured responsiveness/headroom criteria')
    return result
