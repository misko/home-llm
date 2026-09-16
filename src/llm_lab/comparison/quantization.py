"""Compare BF16 and GGUF next-token distributions on a development prompt."""
import json
import subprocess
from pathlib import Path

from llm_lab.paths import LabPaths
from llm_lab.training_runtime import training_lease

from .contracts import atomic_json, file_hash
from .sharing import process_duty_cycle


def check(root: Path, arm: str):
    import numpy as np
    config=json.loads((root/'execution.json').read_text())
    if not (Path(config['training_run'])/'ready.json').exists():raise RuntimeError('Wait for training/export before GPU checks')
    record=json.loads((root/'artifacts.json').read_text())[arm]
    reference=root/'references'/arm
    manifest=json.loads((reference/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if file_hash(reference/name)!=expected:raise ValueError('BF16 reference changed')
    expected_shards={k:v for k,v in record['provenance']['build']['input_files'].items() if k.endswith('.safetensors')}
    if manifest['source_shards']!=expected_shards:raise ValueError('Reference and converted artifact have different parents')
    binary=Path(config['data_root'])/'tools/comparison/token-nll'
    protocol=json.loads((root/'nll-protocol.json').read_text())
    if file_hash(binary)!=protocol['binary_sha256']:raise ValueError('Native evaluation binary changed')
    output=reference/'gguf-logits.float32'
    paths=LabPaths.discover(repo_root=config['repo_root'],data_root=config['data_root'])
    with training_lease(paths),(reference/'quantization.log').open('w') as log:
        child=subprocess.Popen([str(binary),'--reference',str(Path(record['view'])/'model-Q4_K_M.gguf'),str(reference/'tokens.uint32'),str(output)],stdout=log,stderr=log)
        try:
            with process_duty_cycle(child.pid,config['inference_duty_cycle'],allow_target_exit=True):
                code=child.wait()
            if code:raise RuntimeError('Quantization comparison failed; inspect '+str(reference/'quantization.log'))
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    before=np.load(reference/'logits.npy').astype(np.float64)
    after=np.fromfile(output,dtype='<f4').astype(np.float64)
    if before.shape!=after.shape or not np.isfinite(after).all():raise ValueError('Quantization logits are invalid')
    def log_softmax(values):
        shifted=values-values.max()
        return shifted-np.log(np.exp(shifted).sum())
    a,b=log_softmax(before),log_softmax(after)
    kl=max(0.,float(np.sum(np.exp(a)*(a-b))))
    result={'bf16_to_gguf_kl':kl,'top_token_agrees':bool(before.argmax()==after.argmax()),
            'artifact_identity':record['identity'],'reference_manifest_sha256':file_hash(reference/'manifest.json'),
            'native_binary_sha256':file_hash(binary),'passed':kl<.2,
            'kl_tolerance':.2,'note':'Single development prompt; smoke check for major conversion damage, not broad capability equivalence.'}
    atomic_json(reference/'quantization.json',result)
    if not result['passed']:raise RuntimeError('Quantization drift exceeded the development smoke-check tolerance')
    return result
