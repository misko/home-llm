"""Frozen token-level evaluation through the same quantized artifacts."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from llm_lab.paths import LabPaths
from llm_lab.training_runtime import training_lease

from .contracts import atomic_json, file_hash
from .sharing import process_duty_cycle


def prepare(root: Path):
    config=json.loads((root/'execution.json').read_text())
    repo=Path(config.get('code_root',config['repo_root']));data=Path(config['data_root'])
    source=repo/'benchmarks/qwen-comparison/token-nll.cpp'
    cpp=data/'cache/llama.cpp'
    binary=data/'tools/comparison/token-nll'
    command=['g++','-std=c++17','-O2',str(source),'-I'+str(cpp/'include'),'-I'+str(cpp/'ggml/include'),
             '-L'+str(cpp/'build/bin'),'-Wl,-rpath,'+str(cpp/'build/bin'),'-lllama','-lggml','-lggml-base','-o',str(binary)]
    subprocess.run(command,check=True)
    subprocess.run([str(binary),'--self-test'],check=True)
    dataset=Path(config['training_run'])/'dataset'
    original=json.loads((dataset/'manifest.json').read_text())
    if file_hash(dataset/'test.bin')!=original['files']['test.bin']:
        raise ValueError('FineWeb held-out tokens changed')
    tokens=root/'nll-tokens.uint32'
    with (dataset/'test.bin').open('rb') as stream:
        content=stream.read(512*256*4)
    if len(content)!=512*256*4:
        raise ValueError('Insufficient held-out tokens')
    if tokens.exists() and tokens.read_bytes()!=content:
        raise ValueError('Frozen NLL input differs')
    tokens.write_bytes(content)
    manifest={'binary':str(binary),'binary_sha256':file_hash(binary),'source_sha256':file_hash(source),
              'build_command':command,'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
              'libraries':{str(p.resolve()):file_hash(p) for p in (cpp/'build/bin').glob('lib*.so')},
              'tokens_sha256':file_hash(tokens),'source_test_sha256':original['files']['test.bin'],
              'target_tokens':65536,'window_tokens':512,'windows':256,'score_positions':'next tokens at positions 256 through 511 inclusive',
              'bos_added':False,'chat_template_applied':False,'window_memory_reset':True,
              'note':'Same packed held-out stream for all arms; this windowing differs from the training report.'}
    path=root/'nll-protocol.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('NLL protocol changed after freezing')
    atomic_json(path,manifest)
    return manifest


def run(root: Path, arm: str):
    config=json.loads((root/'execution.json').read_text())
    protocol=json.loads((root/'nll-protocol.json').read_text())
    records=json.loads((root/'artifacts.json').read_text())
    if set(records)!={'qwen','qwen-ft-fineweb','qwen-heretic','qwen-heretic-plus'}:
        raise RuntimeError('All four selected artifacts must be registered before final NLL evaluation')
    binary=Path(protocol['binary'])
    if file_hash(binary)!=protocol['binary_sha256'] or file_hash(root/'nll-tokens.uint32')!=protocol['tokens_sha256']:
        raise ValueError('NLL executable or frozen tokens changed')
    for name,expected in protocol['libraries'].items():
        if file_hash(Path(name))!=expected:raise ValueError('NLL linked library changed: '+name)
    record=records[arm]
    model=Path(record['view'])/'model-Q4_K_M.gguf'
    if file_hash(model)!=record['provenance']['weight_sha256']:
        raise ValueError('NLL model differs from registered comparison artifact')
    output=root/'arms'/arm/'nll.json'
    if output.exists():
        previous=json.loads(output.read_text())
        if previous['artifact_identity']!=record['identity'] or previous['protocol']!=protocol:
            raise ValueError('NLL result has different provenance')
        return previous
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.partial.json')
    paths=LabPaths.discover(repo_root=config['repo_root'],data_root=config['data_root'])
    with training_lease(paths), output.with_suffix('.log').open('w') as log:
        child=subprocess.Popen([str(binary),str(model),str(root/'nll-tokens.uint32'),str(temporary)],stdout=log,stderr=log)
        try:
            with process_duty_cycle(child.pid,config['inference_duty_cycle'],allow_target_exit=True):
                code=child.wait()
            if code:raise RuntimeError(f'NLL process failed with exit {code}; see {output.with_suffix(".log")}')
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    result=json.loads(temporary.read_text())
    if result['target_tokens']!=65536:
        raise ValueError('NLL evaluated an unexpected token count')
    result.update(protocol=protocol,artifact_identity=record['identity'])
    atomic_json(output,result);temporary.unlink()
    import wandb
    with wandb.init(**config['wandb'],id='qcomp-nll-'+arm,resume='allow',dir=str(output.parent),
                    config={'arm':arm,'artifact_identity':record['identity'],'protocol':protocol},
                    settings=wandb.Settings(disable_git=True)) as tracking:
        tracking.log({'fineweb/nll':result['nll'],'fineweb/target_tokens':65536})
    return result
