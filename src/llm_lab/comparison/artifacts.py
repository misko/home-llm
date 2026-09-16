"""Build matched GGUF artifacts on CPU and register distinct Lab deployments."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from llm_lab.catalog import Catalog
from llm_lab.paths import LabPaths
from llm_lab.runtime import ExclusiveGpuLock
from llm_lab.schema import ArtifactSpec, DeploymentSpec, ModelSpec
from llm_lab.storage import ArtifactStore

from .contracts import atomic_json, file_hash, fingerprint


def register(paths: LabPaths, arm: str, tree: Path, provenance: dict) -> dict:
    catalog = Catalog.load(paths.catalog_root)
    identity = fingerprint(provenance)
    model_id = 'qwen-comparison-' + arm
    original = catalog.get_model('qwen3.8-27b').model_dump(mode='json')
    original.update(id=model_id, display_name={'qwen':'Qwen · matched baseline', 'qwen-ft-fineweb':'Qwen · FineWeb', 'qwen-heretic':'Qwen · Heretic', 'qwen-heretic-plus':'Qwen · Heretic++'}[arm],
                    description='Matched text-only Qwen comparison artifact. See comparison-provenance.json.',
                    modalities=['text'], upstream={'provider':'local','local_path':str(tree),'revision':identity})
    model = ModelSpec.model_validate(original)
    artifact = ArtifactSpec.model_validate({'id':model_id+'-q4-k-m','model_id':model_id,
        'source':{'provider':'local','local_path':str(tree),'revision':identity}, 'format':'gguf','quantization':'Q4_K_M',
        'expected_size_bytes':sum(p.stat().st_size for p in tree.iterdir() if p.is_file()),
        'files':[{'pattern':'model-Q4_K_M.gguf','role':'weights'}, {'pattern':'comparison-provenance.json','role':'config'}]})
    definition = catalog.get_deployment('qwen3.8-27b-4090-8k').model_dump(mode='json')
    definition.update(id=model_id+'-4090',artifact_id=artifact.id, public_alias='compare-'+arm,
                      port=18091,mmproj=None,lora_adapter=None)
    deployment = DeploymentSpec.model_validate(definition)
    with ArtifactStore(paths, free_reserve_bytes=540_000_000_000) as store:
        promoted = store.promote(artifact, tree, resolved_revision=identity)
        store.verify(artifact.id, verify_view=True)
    for folder, value in [('models',model),('artifacts',artifact),('deployments',deployment)]:
        path=paths.catalog_root/folder/(value.id+'.yaml')
        data=value.model_dump(mode='json',exclude_none=True)
        if path.exists() and yaml.safe_load(path.read_text()) != data:
            raise ValueError('Catalog identity already exists with different content: '+str(path))
        temporary=path.with_suffix('.partial')
        temporary.write_text(yaml.safe_dump(data,sort_keys=False));os.replace(temporary,path)
    Catalog.load(paths.catalog_root)
    return {'deployment_id':deployment.id,'artifact_id':artifact.id,'model_id':model.id,
            'view':str(promoted.view_path),'provenance':provenance,'identity':identity}


def build(root: Path, arm: str, input_model: Path, *, parent: str | None) -> dict:
    config=json.loads((root/'execution.json').read_text())
    paths=LabPaths.discover(repo_root=config['repo_root'],data_root=config['data_root'])
    runtime=Catalog.load(paths.catalog_root).get_runtime_lock('llama-cpp-cuda-4090')
    converter=paths.data_root/'cache/llama.cpp'
    if subprocess.check_output(['git','-C',str(converter),'rev-parse','HEAD'],text=True).strip()!=runtime.commit:
        raise ValueError('Converter does not match the locked runtime')
    if subprocess.check_output(['git','-C',str(converter),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('Converter checkout is modified')
    source_files={p.name:file_hash(p) for p in sorted(input_model.iterdir()) if p.is_file()}
    if not any(name.endswith('.safetensors') for name in source_files):
        raise ValueError('Input must contain merged full-precision safetensors')
    quantizer=converter/'build/bin/llama-quantize'
    provenance={'arm':arm,'parent':parent,'input_model':str(input_model),'input_files':source_files,
                'converter_commit':runtime.commit,'converter_sha256':file_hash(converter/'convert_hf_to_gguf.py'),
                'quantizer_sha256':file_hash(quantizer),'quantization':'Q4_K_M','conversion_type':'bf16',
                'environment_lock_sha256':file_hash(Path(config['environment_lock']))}
    tree=root/'artifacts'/arm;tree.mkdir(parents=True,exist_ok=True)
    with ExclusiveGpuLock(root/'artifacts.lock',timeout_seconds=0):
        manifest=tree/'comparison-provenance.json'
        weight=tree/'model-Q4_K_M.gguf'
        if manifest.exists():
            previous=json.loads(manifest.read_text())
            if previous['build']!=provenance or file_hash(weight)!=previous['weight_sha256']:
                raise ValueError('Existing comparison artifact differs')
        else:
            import shutil
            if shutil.disk_usage(root).free < 650_000_000_000:
                raise ValueError('Insufficient space for BF16 conversion and storage reserve')
            intermediate=root/'artifacts'/(arm+'-bf16.gguf')
            environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
            # Incomplete conversion outputs may be replaced; published artifacts may not.
            subprocess.run([config['evaluation_python'],str(converter/'convert_hf_to_gguf.py'),str(input_model),
                            '--outfile',str(intermediate),'--outtype','bf16'],env=environment,check=True)
            subprocess.run([str(quantizer),str(intermediate),str(weight),'Q4_K_M','4'],env=environment,check=True)
            atomic_json(manifest,{'build':provenance,'weight_sha256':file_hash(weight)})
            intermediate.unlink()
        record=register(paths,arm,tree,json.loads(manifest.read_text()))
        inspector=Path(config.get('code_root',config['repo_root']))/'benchmarks/qwen-comparison/inspect-gguf.py'
        metadata=root/'artifacts'/(arm+'-serving-metadata.json')
        subprocess.run([config['evaluation_python'],str(inspector),str(converter),str(weight),str(metadata)],check=True,
                       env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        record['serving_metadata']=json.loads(metadata.read_text())
        records_path=root/'artifacts.json'
        records=json.loads(records_path.read_text()) if records_path.exists() else {}
        if arm in records:
            previous=dict(records[arm])
            previous.setdefault('serving_metadata',record['serving_metadata'])
            if previous!=record:raise ValueError('Registered comparison arm differs')
        records[arm]=record;atomic_json(records_path,records)
        return record
