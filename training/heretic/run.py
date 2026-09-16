"""Prepare and run Heretic on the completed FineWeb checkpoint.

Use LLM Lab's control Python. GPU operations execute in the isolated Heretic env.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import digest, verify_checkpoint, identity


def can_finalize_exhausted_study(trials) -> bool:
    """Return whether a resumed study has a candidate it can export.

    Optuna state values are intentionally inspected by name so this small
    recovery predicate remains testable without importing Optuna in the
    control environment.
    """
    return any(getattr(getattr(trial, 'state', None), 'name', None) == 'COMPLETE'
               for trial in trials)


def experiment_fingerprint(config):
    root=Path(config['run_dir']);scripts=Path(__file__).resolve().parent
    manifest=json.loads((Path(config['input_model'])/'fineweb-input-manifest.json').read_text())
    files=[scripts/name for name in ['worker.py','run.py','merge_input.py','monitoring.py']]
    files += [scripts.parent/'common.py',scripts.parent/'throttle.py']
    if config.get('variant')=='plus':files.append(scripts/'plus.py')
    return identity({'input':manifest['fingerprint'],'profile':digest(root/'config.toml'),
                     'heretic_commit':config['heretic_commit'],'launcher':config,
                     'source':{p.name:digest(p) for p in files}})


def completed_checkpoint(config):
    run = Path(config['training_run'])
    if not (run / 'trained.json').exists() or not (run / 'ready.json').exists():
        raise RuntimeError('FineWeb training and LLM Lab export must both finish first.')
    trained = json.loads((run / 'trained.json').read_text())
    ready = json.loads((run / 'ready.json').read_text())
    if ready.get('phase') != 'ready':
        raise RuntimeError('FineWeb export is not ready.')
    checkpoint = Path(trained['checkpoint'])
    manifest = verify_checkpoint(checkpoint)
    if manifest['state']['step'] != trained['step']:
        raise RuntimeError('Final checkpoint does not match completed training.')
    return checkpoint


def validate_install(config):
    source = Path(config['heretic_source'])
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != config['heretic_commit']:
        raise RuntimeError('Heretic source revision changed.')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Heretic source has tracked modifications.')
    if digest(Path(config['environment_lock'])) != config['environment_lock_sha256']:
        raise RuntimeError('Heretic environment lock changed.')
    base = Path(config['base_path'])
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    missing = [name for name in set(index['weight_map'].values()) if not (base / name).is_file()]
    if missing:
        raise RuntimeError(f'Pinned base download is incomplete: {len(missing)} missing shards.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('action', choices=['check', 'prepare-input', 'probe', 'run'])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_install(config)
    if args.action == 'check':
        try:
            checkpoint = completed_checkpoint(config)
            print(f'Environment and base ready; final checkpoint: {checkpoint}')
        except RuntimeError as exc:
            print(f'Environment and base ready. Pending: {exc}')
        return
    # Gate before importing torch or launching any GPU process.
    completed_checkpoint(config)
    result_path=Path(config['run_dir'])/'result.json'
    if args.action=='run' and result_path.exists():
        result=json.loads(result_path.read_text())
        if result['fingerprint']!=experiment_fingerprint(config) or digest(Path(result['adapter'])/'adapter_model.safetensors')!=result['adapter_sha256']:
            raise RuntimeError('Completed Heretic result has incompatible provenance.')
        print('Verified completed Heretic study; no rerun needed.')
        return
    environment = {**os.environ, 'PYTHONPATH': str(Path(config.get('code_root', config['repo_root'])) / 'src'),
        'HF_HOME': config['hf_home'], 'OMP_NUM_THREADS': '4', 'RAYON_NUM_THREADS': '4',
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'PYTHONUNBUFFERED': '1'}
    scripts = Path(__file__).resolve().parent
    def execute(name, mode=None, cpu=False):
        command = [config['heretic_python'], str(scripts / name), str(args.config.resolve())]
        if mode:
            command.append(mode)
        child = subprocess.Popen(command, env={**environment, **({'CUDA_VISIBLE_DEVICES': ''} if cpu else {})})
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait(timeout=60)
            raise
        if code:
            raise SystemExit(code)
    execute('merge_input.py', cpu=True)
    if args.action == 'prepare-input':
        return
    probe_path=Path(config['run_dir'])/'gpu-probe.json'
    probe_valid=probe_path.exists() and json.loads(probe_path.read_text()).get('fingerprint')==experiment_fingerprint(config)
    if args.action=='probe' or not probe_valid:
        execute('worker.py', 'probe')
    if args.action == 'run':
        execute('worker.py', 'run')


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc))
