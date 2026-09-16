"""Convert the adapter, verify llama.cpp inference, and register a test candidate.

Run with the LLM Lab control environment. The converter uses the isolated
training Python, while serving uses the existing verified llama.cpp binary.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml
from common import atomic_json, check_space, digest, verify_checkpoint
from llm_lab.benchmark import run_benchmark
from llm_lab.catalog import Catalog
from llm_lab.paths import LabPaths
from llm_lab.runtime import ProcessLauncher, build_backend_command, verify_runtime_lock, wait_for_readiness
from llm_lab.schema import ArtifactSpec, DeploymentSpec, ModelSpec
from llm_lab.storage import ArtifactStore
from llm_lab.training_runtime import training_lease


def write_catalog(path, value):
    text=yaml.safe_dump(value.model_dump(mode='json',exclude_none=True),sort_keys=False)
    if path.exists():
        if yaml.safe_load(path.read_text())!=yaml.safe_load(text):
            raise ValueError(f'Refusing to replace different catalog entry {path}')
        return
    temporary=path.with_suffix('.yaml.partial')
    temporary.write_text(text);os.replace(temporary,path)


def inference_check(paths,catalog,tree,adapter,run,label):
    original=catalog.get_deployment('qwen3.8-27b-4090-8k')
    dep=original.model_copy(update={'id':f'fineweb-{label}', 'public_alias':f'fineweb-{label}',
        'lora_adapter':('/models/'+adapter if adapter else None),'context_size':4096,'port':18089})
    lock=catalog.runtime_locks[dep.runtime_lock_id]
    evidence=verify_runtime_lock(paths,dep,lock,tree)
    plan=build_backend_command(dep,tree,paths=paths)
    plan=replace(plan,expected_executable_sha256=evidence.binary_sha256,
        expected_executable_root=str(paths.data_root),expected_executable_version_contains=lock.version_contains)
    launcher=ProcessLauncher();record=launcher.start(plan)
    try:
        wait_for_readiness(plan.health_url,300)
        result=asyncio.run(run_benchmark(catalog.get_suite('smoke'),base_url=plan.base_url,
            served_model=dep.public_alias,deployment=dep.model_dump(mode='json'),
            model_id='qwen3.8-27b',runtime={'source_commit':lock.commit,'binary_sha256':evidence.binary_sha256},
            available_capabilities=['chat','structured_output','tools'],collect_telemetry=True))
        output=run/f'{label}-smoke'
        if not output.exists():
            result.write_bundle(output)
        if any(sample.error for sample in result.samples):
            raise RuntimeError(f'Candidate inference request failed: {result.summary}')
        if not any(sample.output_text for sample in result.samples):
            raise RuntimeError('Candidate produced no text')
        return result.summary
    finally:
        launcher.stop(record,timeout_seconds=20)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path)
    parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    c=json.loads(args.config.read_text());run=Path(c['run_dir'])
    if not args.probe and (run/'ready.json').exists():
        print((run/'ready.json').read_text());return
    paths=LabPaths.discover(repo_root=c['repo_root'],data_root=c['data_root'])
    catalog=Catalog.load(paths.catalog_root)
    if args.probe:
        checkpoint=run/'probe-adapter'
    else:
        checkpoint=Path(json.loads((run/'trained.json').read_text())['checkpoint'])
        verify_checkpoint(checkpoint)
    tree=run/('probe-export' if args.probe else 'export');tree.mkdir(exist_ok=True)
    check_space(paths.data_root,2_000_000_000)
    source=paths.data_root/'cache/llama.cpp'
    runtime=catalog.runtime_locks['llama-cpp-cuda-4090']
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=runtime.commit:
        raise ValueError('Converter commit differs from pinned llama.cpp runtime')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Converter checkout contains modifications')
    adapter='fineweb-adapter.gguf'
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':''}
    subprocess.run([c['training_python'],str(source/'convert_lora_to_gguf.py'),str(checkpoint),
        '--base',c['export_base_path'],'--outfile',str(tree/adapter),'--outtype','f16'],
        env=environment,check=True)
    base=catalog.get_artifact('qwen3.8-27b-q4-k-m')
    with ArtifactStore(paths) as store:
        store.verify(base.id,verify_view=True)
        base_view=store.view_path(base.id)
        for name in ['Qwen3.8-27B-Q4_K_M.gguf','mmproj-Qwen3.8-27B-f16.gguf']:
            target=tree/name
            if not target.exists():
                os.link((base_view/name).resolve(),target)
    with training_lease(paths):
        baseline=inference_check(paths,catalog,base_view,None,run,'original') if not (run/'original-smoke').exists() else json.loads((run/'original-smoke/summary.json').read_text())
        candidate=inference_check(paths,catalog,tree,adapter,run,'probe' if args.probe else 'candidate')
    if args.probe:
        atomic_json(run/'export-probe.json',{'baseline':baseline,'candidate':candidate,'adapter_sha256':digest(tree/adapter)})
        return
    model_id='qwen3.8-27b-fineweb-24h-20260912'
    artifact_id=model_id+'-q4-k-m-lora'
    adapter_hash=digest(tree/adapter)
    provenance={'run':str(run),'adapter_sha256':adapter_hash,'checkpoint':str(checkpoint),
        'training':json.loads((run/'trained.json').read_text()),'config':c,
        'dataset':json.loads((run/'dataset/manifest.json').read_text()),
        'evaluation':json.loads((run/'final-evaluation.json').read_text()),
        'base_artifact':base.model_dump(mode='json'),'converter_commit':revision}
    for name in ('source-revisions.json', 'gpu-duty-cycle.json'):
        if (run/name).exists():
            provenance[name]=json.loads((run/name).read_text())
    atomic_json(tree/'training-provenance.json',provenance)
    (tree/'README.md').write_text('# Qwen FineWeb 24-hour experiment\n\n'
        'QLoRA continued pretraining on a bounded FineWeb sample. Experimental test candidate.\n'
        'Served as the original Q4_K_M weights plus a float16 LoRA adapter.\n'
        'Training and evaluation details are in training-provenance.json.\n')
    model=catalog.get_model('qwen3.8-27b').model_copy(update={
        'id':model_id,'display_name':'Qwen3.8 27B · FineWeb 24h',
        'description':'Experimental FineWeb continued-pretraining adapter; compare against original Qwen.',
    })
    # Revalidate the local learned identity while preserving official parent provenance.
    md=model.model_dump(mode='json');md['upstream']={'provider':'local','local_path':str(checkpoint),'revision':adapter_hash}
    model=ModelSpec.model_validate(md)
    artifact=ArtifactSpec.model_validate({
        'id':artifact_id,'model_id':model_id,'source':{'provider':'local','local_path':str(tree),'revision':adapter_hash},
        'format':'gguf','quantization':'Q4_K_M + F16 LoRA',
        'expected_size_bytes':sum(p.stat().st_size for p in tree.iterdir() if p.is_file()),
        'files':[{'pattern':'Qwen3.8-27B-Q4_K_M.gguf','role':'weights'},
            {'pattern':'mmproj-Qwen3.8-27B-f16.gguf','role':'vision_projector'},
            {'pattern':adapter,'role':'adapter'}, {'pattern':'README.md','role':'model_card'},
            {'pattern':'training-provenance.json','role':'config'}],
        'notes':'Original frozen base GGUF plus FineWeb LoRA adapter; provenance includes training checkpoint and data hashes.',
    })
    dd=catalog.get_deployment('qwen3.8-27b-4090-8k').model_dump(mode='json')
    dd.update(id=model_id+'-4090',artifact_id=artifact_id,public_alias='local-fineweb',lora_adapter='/models/'+adapter)
    deployment=DeploymentSpec.model_validate(dd)
    with ArtifactStore(paths,free_reserve_bytes=540_000_000_000) as store:
        promoted=store.promote(artifact,tree,resolved_revision=adapter_hash)
        store.verify(artifact.id,verify_view=True)
    write_catalog(paths.catalog_root/'models'/f'{model.id}.yaml',model)
    write_catalog(paths.catalog_root/'artifacts'/f'{artifact.id}.yaml',artifact)
    write_catalog(paths.catalog_root/'deployments'/'local-fineweb.yaml',deployment)
    Catalog.load(paths.catalog_root)
    ready={'phase':'ready','deployment_id':deployment.id,'public_alias':deployment.public_alias,
        'artifact_id':artifact.id,'view':str(promoted.view_path),'baseline':baseline,'candidate':candidate}
    atomic_json(run/'ready.json',ready);atomic_json(run/'status.json',ready)
    print(json.dumps(ready),flush=True)


if __name__=='__main__':
    main()
