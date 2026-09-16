"""Exercise final registration with tiny artifacts; GPU conversion is probed separately."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from llm_lab.catalog import Catalog
from llm_lab.paths import LabPaths
from llm_lab.storage import ArtifactStore


def test_export_registers_verified_candidate_and_is_repeatable(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'training'))
    spec = importlib.util.spec_from_file_location('training_export', root / 'training/export.py')
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    repo = tmp_path / 'repo'
    shutil.copytree(root / 'catalog', repo / 'catalog')
    for relative in (
        'models/qwen3.8-27b-fineweb-24h-20260912.yaml',
        'artifacts/qwen3.8-27b-fineweb-24h-20260912-q4-k-m-lora.yaml',
        'deployments/local-fineweb.yaml',
    ):
        (repo / 'catalog' / relative).unlink(missing_ok=True)
    paths = LabPaths.discover(repo_root=repo, data_root=tmp_path / 'data')
    catalog = Catalog.load(paths.catalog_root)
    base = tmp_path / 'base'
    base.mkdir()
    for name in ('Qwen3.8-27B-Q4_K_M.gguf', 'mmproj-Qwen3.8-27B-f16.gguf', 'README.md'):
        (base / name).write_bytes(name.encode())
    artifact = catalog.get_artifact('qwen3.8-27b-q4-k-m').model_copy(update={
        'expected_size_bytes': None,
    })
    with ArtifactStore(paths) as store:
        store.promote(artifact, base, resolved_revision='test-base')
    run = tmp_path / 'run'
    checkpoint = run / 'checkpoint'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'adapter')
    export.atomic_json(checkpoint / 'manifest.json', {'files': {
        'adapter_model.safetensors': export.digest(checkpoint / 'adapter_model.safetensors'),
    }})
    export.atomic_json(run / 'trained.json', {'checkpoint': str(checkpoint), 'step': 3})
    export.atomic_json(run / 'final-evaluation.json', {'validation': {'nll': 2.0}})
    export.atomic_json(run / 'dataset/manifest.json', {'snapshot_id': 'test'})
    config = run / 'config.json'
    export.atomic_json(config, {
        'run_dir': str(run), 'repo_root': str(repo), 'data_root': str(paths.data_root),
        'training_python': sys.executable, 'export_base_path': str(base),
    })
    revision = catalog.runtime_locks['llama-cpp-cuda-4090'].commit
    monkeypatch.setattr(export.subprocess, 'check_output',
                        lambda args, **kw: revision if 'rev-parse' in args else '')
    def convert(args, **kwargs):
        Path(args[args.index('--outfile') + 1]).write_bytes(b'GGUF adapter fixture')
    monkeypatch.setattr(export.subprocess, 'run', convert)
    monkeypatch.setattr(export, 'inference_check', lambda *a: {'pass_rate': 1.0})
    monkeypatch.setattr(export, 'check_space', lambda *a: None)
    monkeypatch.setattr(sys, 'argv', ['export.py', str(config)])
    export.main()
    ready = json.loads((run / 'ready.json').read_text())
    loaded = Catalog.load(paths.catalog_root)
    deployment = loaded.get_deployment(ready['deployment_id'])
    assert deployment.public_alias == 'local-fineweb'
    assert deployment.lora_adapter == '/models/fineweb-adapter.gguf'
    with ArtifactStore(paths) as store:
        assert store.verify(ready['artifact_id'], verify_view=True).file_count == 5
    export.main()
    assert json.loads((run / 'ready.json').read_text()) == ready
