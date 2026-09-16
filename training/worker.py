"""Single-GPU, time-budgeted QLoRA continued pretraining with exact data replay."""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Unsloth must patch Transformers before importing it.
from unsloth import FastModel
import bitsandbytes as bnb
import numpy as np
import torch
import wandb
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from common import atomic_json, check_space, digest, elapsed_total, identity, verify_checkpoint
from throttle import read_duty_cycle, yield_gpu
from llm_lab.paths import LabPaths
from llm_lab.training_runtime import training_lease


def load_model(config):
    # Transformers 5.5 recursively applies the text-only prefix conversion to
    # this multimodal checkpoint, incorrectly moving model.language_model.*
    # to model.*. Its stored keys already match the conditional model exactly.
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    register_checkpoint_conversion_mapping('qwen3_5_text', [], overwrite=True)
    model, processor = FastModel.from_pretrained(
        model_name=config['model_path'], max_seq_length=config['sequence_length'],
        dtype=torch.bfloat16, load_in_4bit=True, full_finetuning=False,
        offload_embedding=True, local_files_only=True, device_map={'':0},
    )
    # Text-only continued pretraining never calls the vision encoder.
    # Keep its unchanged weights in system RAM to leave GPU headroom.
    if hasattr(model.model, 'visual'):
        model.model.visual.to('cpu')
        torch.cuda.empty_cache()
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        r=config['lora_rank'], lora_alpha=config['lora_rank'], lora_dropout=0,
        bias='none', use_gradient_checkpointing='unsloth', random_state=config['seed'],
    )
    model.config.use_cache = False
    if hasattr(model.config, 'text_config'):
        model.config.text_config.use_cache = False
    FastModel.for_training(model)
    for name, layer in model.named_modules():
        if isinstance(layer, bnb.nn.Linear4bit) and getattr(layer.weight, 'quant_state', None) is None:
            raise RuntimeError(f'Missing quantization state after checkpoint loading: {name}')
    return model, processor


class TokenDataset:
    def __init__(self, path, split):
        self.tokens = np.memmap(path/f'{split}.bin', dtype='<u4', mode='r')
        self.index = np.load(path/f'{split}.npy', mmap_mode='r')

    def __len__(self):
        return len(self.index)

    def get(self, index):
        offset, length = self.index[index]
        return torch.tensor(np.array(self.tokens[offset:offset+length], dtype=np.int64), device='cuda').unsqueeze(0)


def evaluate(model, data, seed, count=32, duty_cycle=1.0):
    model.eval()
    chosen = np.random.default_rng(seed).permutation(len(data))[:count]
    total_loss = 0.; total_tokens = 0
    with torch.no_grad():
        for i in chosen:
            active_started = time.monotonic()
            ids = data.get(i)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(input_ids=ids, labels=ids, use_cache=False)
            n = ids.numel()-1
            loss = float(output.loss)
            if duty_cycle < 1:
                torch.cuda.synchronize()
                yield_gpu(time.monotonic() - active_started, duty_cycle)
            if not math.isfinite(loss):
                raise RuntimeError('Nonfinite evaluation loss')
            total_loss += loss*n; total_tokens += n
    model.train()
    mean = total_loss/total_tokens
    return {'nll':mean, 'perplexity':math.exp(mean), 'target_tokens':total_tokens, 'sequences':len(chosen)}


def save_checkpoint(run, model, optimizer, state, fingerprint):
    root = run/'checkpoints'; root.mkdir(exist_ok=True)
    destination = root/f"step-{state['step']:08d}"
    if destination.exists():
        return destination
    check_space(run, 3_000_000_000)
    work = root/(destination.name+'.partial')
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    model.save_pretrained(work, safe_serialization=True)
    payload = {
        'state':state.copy(), 'optimizer':optimizer.state_dict(),
        'python_rng':random.getstate(), 'numpy_rng':np.random.get_state(),
        'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all(),
    }
    torch.save(payload, work/'training.pt')
    for path in work.iterdir():
        if path.is_file():
            with path.open('rb') as f:
                os.fsync(f.fileno())
    atomic_json(work/'manifest.json', {
        'fingerprint':fingerprint, 'state':state.copy(),
        'files':{p.name:digest(p) for p in work.iterdir() if p.is_file()},
    })
    os.replace(work, destination)
    atomic_json(run/'latest.json', {'checkpoint':str(destination)})
    # Preserve latest two. Validation is reported without automatic checkpoint selection.
    checkpoints = sorted(p for p in root.glob('step-*') if p.is_dir() and not p.name.endswith('.partial'))
    for old in checkpoints[:-2]:
        shutil.rmtree(old)
    return destination


def execute(config, mode):
    run = Path(config['run_dir']); run.mkdir(parents=True, exist_ok=True)
    paths = LabPaths.discover(repo_root=config['repo_root'], data_root=config['data_root'])
    with training_lease(paths):
        atomic_json(run/'status.json', {'phase':'loading', 'pid':os.getpid(), 'mode':mode})
        check_space(paths.data_root, 5_000_000_000)
        torch.set_num_threads(4)
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        model, processor = load_model(config)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError('No trainable adapter parameters')
        print('TRAINABLE', sum(p.numel() for p in trainable), flush=True)
        optimizer = bnb.optim.AdamW8bit(trainable, lr=config['learning_rate'], weight_decay=0.01)
        if mode == 'probe':
            tok = getattr(processor, 'tokenizer', processor)
            text = ('A reproducible experiment records its data, parameters, and measured results. ' * 180)
            ids = tok(text, return_tensors='pt', add_special_tokens=False)['input_ids'][:, :config['sequence_length']].cuda()
            frozen = next(p for p in model.parameters() if not p.requires_grad and p.numel()>16)
            frozen_before = frozen.detach().flatten()[:16].clone()
            before = [p.detach().cpu().clone() for p in trainable]
            durations=[]
            for step in range(3):
                t=time.monotonic()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss=model(input_ids=ids, labels=ids, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite probe loss')
                loss.backward(); norm=torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize()
                durations.append(time.monotonic()-t)
                print('PROBE',step,float(loss),durations[-1],flush=True)
            if not any(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,trainable)):
                raise RuntimeError('Adapter weights did not update')
            if not torch.equal(frozen_before, frozen.detach().flatten()[:16]):
                raise RuntimeError('Frozen parameter changed')
            model.save_pretrained(run/'probe-adapter', safe_serialization=True)
            loaded=load_file(str(run/'probe-adapter/adapter_model.safetensors'))
            set_peft_model_state_dict(model, loaded)
            probe_run=run/'probe-resume';probe_run.mkdir(exist_ok=True)
            probe_state={'step':3,'cursor':3,'epoch':0,'target_tokens':3*(ids.numel()-1),'training_seconds':sum(durations)}
            saved=save_checkpoint(probe_run,model,optimizer,probe_state,'probe')
            verify_checkpoint(saved)
            restored=torch.load(saved/'training.pt',map_location='cpu',weights_only=False)
            optimizer.load_state_dict(restored['optimizer'])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                resumed_loss=model(input_ids=ids,labels=ids,use_cache=False).loss
            resumed_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            atomic_json(run/'probe.json', {
                'loss':float(loss), 'grad_norm':float(norm), 'step_seconds':durations,
                'target_tokens_per_second':(ids.numel()-1)/durations[-1],
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'free_vram_bytes':torch.cuda.mem_get_info()[0],
                'adapter_updated':True, 'frozen_sample_unchanged':True, 'adapter_reloaded':True,
                'optimizer_resume_backward_passed':True,
            })
            atomic_json(run/'status.json', {'phase':'probe_complete'})
            return
        dataset_path=run/'dataset'
        manifest=json.loads((dataset_path/'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            if digest(dataset_path/name) != expected:
                raise ValueError(f'Dataset hash mismatch: {name}')
        fingerprint=identity({'config':config,'dataset':manifest['snapshot_id']})
        tracking=wandb.init(
            project=config['wandb']['project'], entity=config['wandb'].get('entity'),
            name=config['wandb']['name'], id=config['wandb']['id'], resume='allow',
            dir=str(run), config={**config, 'dataset_snapshot':manifest['snapshot_id']},
            allow_val_change=True,
            settings=wandb.Settings(init_timeout=60, disable_git=True),
        )
        tracking.define_metric('training/update')
        tracking.summary['worker_sha256']=digest(Path(__file__))
        tracking.summary['source_directory']=str(Path(__file__).parent.parent)
        tracking.define_metric('training/*', step_metric='training/update')
        atomic_json(run/'wandb.json',{'url':tracking.url,'id':tracking.id,'project':tracking.project})
        train=TokenDataset(dataset_path,'train'); validation=TokenDataset(dataset_path,'validation')
        state={'step':0,'cursor':0,'epoch':0,'target_tokens':0,'training_seconds':0.0}
        latest=run/'latest.json'
        if latest.exists():
            checkpoint=Path(json.loads(latest.read_text())['checkpoint'])
            sealed=verify_checkpoint(checkpoint)
            if sealed['fingerprint'] != fingerprint:
                raise ValueError('Resume config/data fingerprint differs')
            set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter_model.safetensors')))
            saved=torch.load(checkpoint/'training.pt',map_location='cpu',weights_only=False)
            optimizer.load_state_dict(saved['optimizer']); state=saved['state']
            random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
            print('RESUMED',state,flush=True)
        elif not (run/'baseline.json').exists():
            atomic_json(run/'baseline.json', evaluate(model,validation,config['seed'],duty_cycle=read_duty_cycle(run)))
        baseline=json.loads((run/'baseline.json').read_text())
        tracking.summary['baseline_validation_nll']=baseline['nll']
        tracking.summary['baseline_validation_perplexity']=baseline['perplexity']
        tracking.summary['phase']='training'
        stop=[False]
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,lambda *_:stop.__setitem__(0,True))
        previous=state['training_seconds']; started=time.monotonic(); saved_at=started
        budget=config['training_seconds']
        atomic_json(run/'session.json', {
            'started_at':datetime.now(timezone.utc).isoformat(),
            'estimated_training_finish':(datetime.now(timezone.utc)+timedelta(seconds=max(0,budget-previous))).isoformat(),
            'resumed_from_step':state['step'], 'previous_training_seconds':previous,
        })
        torch.cuda.reset_peak_memory_stats()
        order=np.random.default_rng(config['seed']+state['epoch']).permutation(len(train))
        metrics=(run/'metrics.jsonl').open('a',buffering=1)
        gpu_sample={};gpu_sample_time=0.
        try:
            while elapsed_total(previous,started)<budget and not stop[0]:
                batch_start=time.monotonic(); batch_tokens=0; weighted_loss=0.
                duty_cycle=read_duty_cycle(run); throttle_seconds=0.
                fraction=min(1.,elapsed_total(previous,started)/budget)
                lr=config['learning_rate']*min(1.,max(fraction,1e-6)/0.03)*(0.1+0.9*(1-fraction))
                for group in optimizer.param_groups:
                    group['lr']=lr
                samples=[]
                for _ in range(config['gradient_accumulation']):
                    if state['cursor']>=len(order):
                        state['epoch']+=1;state['cursor']=0
                        order=np.random.default_rng(config['seed']+state['epoch']).permutation(len(train))
                    index=int(order[state['cursor']]);state['cursor']+=1
                    ids=train.get(index);samples.append(ids);batch_tokens+=ids.numel()-1
                for ids in samples:
                    active_started=time.monotonic()
                    n=ids.numel()-1
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=model(input_ids=ids,labels=ids,use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite training loss')
                    (loss*(n/batch_tokens)).backward();weighted_loss+=float(loss.detach())*n
                    if duty_cycle < 1:
                        torch.cuda.synchronize()
                        throttle_seconds+=yield_gpu(time.monotonic()-active_started,duty_cycle)
                active_started=time.monotonic()
                norm=torch.nn.utils.clip_grad_norm_(trainable,1.0,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
                if duty_cycle < 1:
                    throttle_seconds+=yield_gpu(time.monotonic()-active_started,duty_cycle)
                state['step']+=1;state['target_tokens']+=batch_tokens
                state['training_seconds']=elapsed_total(previous,started)
                record={**state,'phase':'training','loss':weighted_loss/batch_tokens,'lr':lr,
                    'gpu_duty_cycle_target':duty_cycle,'throttle_seconds':throttle_seconds,
                    'grad_norm':float(norm),'tokens_per_second':batch_tokens/(time.monotonic()-batch_start),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'remaining_seconds':max(0,budget-state['training_seconds']), 'pid':os.getpid()}
                if time.monotonic()-gpu_sample_time>=10:
                    try:
                        values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used,memory.free',
                            '--format=csv,noheader,nounits','-i','0'],text=True,timeout=3).strip().split(',')
                        gpu_sample=dict(zip(['gpu_utilization_percent','gpu_power_watts','gpu_temperature_c','gpu_memory_used_mib','gpu_memory_free_mib'],map(float,values)))
                    except (OSError,ValueError,subprocess.SubprocessError):
                        gpu_sample={}
                    gpu_sample_time=time.monotonic()
                record.update(gpu_sample)
                metrics.write(json.dumps(record)+'\n');atomic_json(run/'status.json',record)
                print(json.dumps(record),flush=True)
                tracking.log({
                    'training/update':state['step'], 'training/loss':record['loss'],
                    'training/learning_rate':lr, 'training/gradient_norm':float(norm),
                    'training/tokens_per_second':record['tokens_per_second'],
                    'training/target_tokens':state['target_tokens'],
                    'training/elapsed_hours':state['training_seconds']/3600,
                    'training/remaining_hours':record['remaining_seconds']/3600,
                    'training/epoch':state['epoch'],
                    'training/gpu_duty_cycle_target':duty_cycle,
                    'training/throttle_seconds':throttle_seconds,
                    'training/peak_allocated_gib':record['peak_allocated_bytes']/2**30,
                    'training/reserved_gib':torch.cuda.memory_reserved()/2**30,
                    **{'training/'+key:value for key,value in gpu_sample.items()},
                })
                if state['step']==1 or time.monotonic()-saved_at>=config['checkpoint_seconds']:
                    save_checkpoint(run,model,optimizer,state,fingerprint);saved_at=time.monotonic()
                    tracking.summary['checkpoint_step']=state['step']
                if state['step']%1000==0:
                    result=evaluate(model,validation,config['seed'],count=16,duty_cycle=read_duty_cycle(run))
                    tracking.log({'training/update':state['step'],
                        'validation/nll':result['nll'],'validation/perplexity':result['perplexity']})
            state['training_seconds']=elapsed_total(previous,started)
            checkpoint=save_checkpoint(run,model,optimizer,state,fingerprint)
            if stop[0]:
                atomic_json(run/'status.json',{'phase':'paused',**state});return
            atomic_json(run/'final-evaluation.json', {
                'validation':evaluate(model,validation,config['seed'],duty_cycle=read_duty_cycle(run)),
                'test':evaluate(model,TokenDataset(dataset_path,'test'),config['seed'],duty_cycle=read_duty_cycle(run)),
            })
            atomic_json(run/'trained.json',{'checkpoint':str(checkpoint),**state})
            atomic_json(run/'status.json',{'phase':'trained',**state})
            result=json.loads((run/'final-evaluation.json').read_text())
            tracking.summary.update({'final_validation_nll':result['validation']['nll'],
                'final_test_nll':result['test']['nll'], 'training_seconds':state['training_seconds'],
                'training_target_tokens':state['target_tokens'], 'phase':'trained'})
        finally:
            metrics.close()
            failed=sys.exc_info()[0] is not None
            if failed or stop[0]:
                tracking.summary['phase']='failed' if failed else 'paused'
            tracking.finish(exit_code=1 if failed else 0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path)
    parser.add_argument('--mode',choices=['probe','train'],default='train');args=parser.parse_args()
    config=json.loads(args.config.read_text())
    try:
        if args.mode=='train' and (Path(config['run_dir'])/'trained.json').exists():
            return
        execute(config,args.mode)
    except BaseException as exc:
        atomic_json(Path(config['run_dir'])/'status.json',{
            'phase':'failed','error':str(exc),'traceback':traceback.format_exc(),'pid':os.getpid()})
        raise


if __name__=='__main__':
    main()
