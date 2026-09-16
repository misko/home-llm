"""Resumable stage coordinator; GPU stages wait for completed FineWeb export."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from llm_lab.runtime import ExclusiveGpuLock

from .contracts import atomic_json, file_hash, fingerprint


def stage_plan(root: Path, config: dict) -> list[dict]:
    code=Path(config.get('code_root',config['repo_root']))
    scripts=code/'training/heretic'
    profiles=json.loads((root/'search-profiles.json').read_text())
    upstream=Path(profiles['upstream_launcher']);plus=Path(profiles['plus_launcher'])
    upstream_config=json.loads(upstream.read_text())
    upstream_root=Path(upstream_config['run_dir'])
    fineweb_input=Path(upstream_config['input_model'])
    plus_root=Path(json.loads(plus.read_text())['run_dir'])
    ctl=config['control_python'];evaluation=config['evaluation_python']
    base=[evaluation,'-m','llm_lab.comparison',str(root)]
    reference=[evaluation,str(code/'benchmarks/qwen-comparison/bf16-reference.py'),str(root)]
    stages=[
        {'id':'qwen-reference','command':reference+['qwen',config['base_path']],'cpu':True},
        {'id':'fineweb-input','command':[ctl,str(scripts/'run.py'),str(upstream),'prepare-input'],'cpu':True},
        {'id':'fineweb-reference','command':reference+['qwen-ft-fineweb',str(fineweb_input)],'cpu':True},
        {'id':'fineweb-artifact','command':[ctl,'-m','llm_lab.comparison',str(root),'build-artifact','qwen-ft-fineweb',str(fineweb_input),'--parent','qwen'],'cpu':True},
        {'id':'sharing-probe','command':base+['probe-sharing']},
        {'id':'qwen-quantization','command':base+['check-quantization','qwen']},
        {'id':'fineweb-quantization','command':base+['check-quantization','qwen-ft-fineweb']},
        {'id':'pilot-qwen','command':base+['run','qwen','--partition','development','--limit','12']},
        {'id':'pilot-fineweb','command':base+['run','qwen-ft-fineweb','--partition','development','--limit','12']},
        {'id':'heretic-upstream-probe','command':[ctl,str(scripts/'run.py'),str(upstream),'probe']},
        {'id':'heretic-plus-probe','command':[ctl,str(scripts/'run.py'),str(plus),'probe']},
        {'id':'heretic-upstream','command':[ctl,str(scripts/'run.py'),str(upstream),'run']},
        {'id':'heretic-plus','command':[ctl,str(scripts/'run.py'),str(plus),'run']},
        {'id':'merge-heretic','command':[evaluation,str(scripts/'merge_variant.py'),str(upstream)],'cpu':True},
        {'id':'heretic-reference','command':reference+['qwen-heretic',str(upstream_root/'heretic-merged-bf16')],'cpu':True},
        {'id':'heretic-artifact','command':[ctl,'-m','llm_lab.comparison',str(root),'build-artifact','qwen-heretic',str(upstream_root/'heretic-merged-bf16'),'--parent','qwen-ft-fineweb'],'cpu':True},
        {'id':'merge-heretic-plus','command':[evaluation,str(scripts/'merge_variant.py'),str(plus)],'cpu':True},
        {'id':'heretic-plus-reference','command':reference+['qwen-heretic-plus',str(plus_root/'heretic-merged-bf16')],'cpu':True},
        {'id':'heretic-plus-artifact','command':[ctl,'-m','llm_lab.comparison',str(root),'build-artifact','qwen-heretic-plus',str(plus_root/'heretic-merged-bf16'),'--parent','qwen-ft-fineweb'],'cpu':True},
        {'id':'heretic-quantization','command':base+['check-quantization','qwen-heretic']},
        {'id':'heretic-plus-quantization','command':base+['check-quantization','qwen-heretic-plus']},
    ]
    for arm in ['qwen','qwen-ft-fineweb','qwen-heretic','qwen-heretic-plus']:
        stages += [{'id':'core-'+arm,'command':base+['run',arm]},
                   {'id':'code-'+arm,'command':base+['score-code',arm],'cpu':True},
                   {'id':'nll-'+arm,'command':base+['nll',arm]}]
    stages += [{'id':'performance','command':base+['performance']},
               {'id':'review-packet','command':base+['review-packet'],'cpu':True},
               {'id':'audit-packet','command':base+['audit-packet'],'cpu':True},
               {'id':'publish','command':base+['publish'],'cpu':True}]
    return stages


def verify_preflight(root: Path, config: dict):
    snapshot=json.loads((root/'source-lock.json').read_text())
    code=Path(config['code_root'])
    for relative,expected in snapshot['files'].items():
        if file_hash(code/relative)!=expected:raise ValueError('Frozen source changed: '+relative)
    validation=json.loads((root/'validation.json').read_text())
    if not validation['passed'] or validation['source_fingerprint']!=fingerprint(snapshot):
        raise ValueError('Required validation does not match this source snapshot')
    context=json.loads((root/'context-validation.json').read_text())
    if not context['passed'] or not context['training_tokenizer_parity'] or context['cases_sha256']!=file_hash(root/'cases.jsonl'):
        raise ValueError('Context/tokenizer validation does not match frozen cases')
    coding=json.loads((root/'code-evaluator-validation.json').read_text())
    if not coding['complete'] or not coding['passed'] or coding['image']!=config['code_image']:
        raise ValueError('Code evaluator has not passed full canonical validation')


def supervise(root: Path):
    config=json.loads((root/'execution.json').read_text())
    stopping=False
    child=None
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with ExclusiveGpuLock(root/'supervisor.lock',timeout_seconds=0):
        verify_preflight(root,config)
        training=Path(config['training_run'])
        while not ((training/'trained.json').exists() and (training/'ready.json').exists()):
            atomic_json(root/'status.json',{'phase':'waiting_for_fineweb','training_run':str(training),'updated_unix':time.time()})
            if stopping:return
            time.sleep(10)
        verify_preflight(root,config)
        stages=stage_plan(root,config)
        receipts=root/'stages';receipts.mkdir(exist_ok=True)
        environment={**os.environ,'PYTHONPATH':str(Path(config['code_root'])/'src'),'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4'}
        for stage in stages:
            if stopping:return
            receipt=receipts/(stage['id']+'.json')
            identity=fingerprint({'stage':stage,'source':json.loads((root/'source-lock.json').read_text()),'config':config})
            if receipt.exists():
                if json.loads(receipt.read_text())['identity']!=identity:raise ValueError('Cannot reuse a stage from another protocol')
                continue
            verify_preflight(root,config)
            atomic_json(root/'status.json',{'phase':'running','stage':stage['id'],'updated_unix':time.time()})
            started=time.time()
            with (receipts/(stage['id']+'.log')).open('a') as log:
                child=subprocess.Popen(stage['command'],env={**environment,**({'CUDA_VISIBLE_DEVICES':''} if stage.get('cpu') else {})},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                code=child.wait();child=None
            if stopping:
                atomic_json(root/'status.json',{'phase':'paused','stage':stage['id'],'updated_unix':time.time()})
                return
            if code:
                atomic_json(root/'status.json',{'phase':'failed','stage':stage['id'],'exit_code':code,'log':str(receipts/(stage['id']+'.log'))})
                raise RuntimeError('Comparison stage failed: '+stage['id'])
            atomic_json(receipt,{'identity':identity,'started_unix':started,'finished_unix':time.time(),'command':stage['command']})
        atomic_json(root/'status.json',{'phase':'awaiting_blind_review','packet':str(root/'review-packet-final.json'),
                                      'audit_packet':str(root/'review-audit-packet-final.json'),
                                      'report':str(root/'report-final.html'),'updated_unix':time.time()})
