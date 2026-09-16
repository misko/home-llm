from pathlib import Path
import subprocess
import sys

import pytest
from llm_lab.paths import LabPaths
from llm_lab.errors import DeploymentError
from llm_lab.runtime import RuntimeManager, build_backend_command
from llm_lab.schema import DeploymentSpec
from llm_lab.training_runtime import training_lease, assert_training_idle


def test_training_reservation_blocks_activation_but_not_status(tmp_path):
    paths=LabPaths.discover(repo_root=tmp_path,data_root=tmp_path/'data')
    deployment=DeploymentSpec(id='mock',artifact_id='mock',public_alias='mock',backend='mock')
    with training_lease(paths):
        with pytest.raises(DeploymentError,match='training job'):
            RuntimeManager(paths).activate(deployment)
        with pytest.raises(DeploymentError,match='training job'):
            with RuntimeManager(paths).benchmark_lease():
                pass
        assert not RuntimeManager(paths).status().active
        with pytest.raises(DeploymentError):
            with training_lease(paths):
                pass
    assert_training_idle(paths)


def test_adapter_is_not_selected_as_base_model(tmp_path):
    base=tmp_path/'base.gguf';base.write_bytes(b'base')
    adapter=tmp_path/'fineweb.gguf';adapter.write_bytes(b'adapter')
    deployment=DeploymentSpec(id='adapted',artifact_id='adapted',public_alias='adapted',
        backend='llama_cpp',executable='/bin/echo',lora_adapter='/models/fineweb.gguf')
    plan=build_backend_command(deployment,tmp_path)
    assert plan.command[plan.command.index('--model')+1]==str(base)
    assert plan.command[plan.command.index('--lora')+1]==str(adapter)
    adapter.unlink()
    with pytest.raises(DeploymentError,match='adapter does not exist'):
        build_backend_command(deployment,tmp_path)


@pytest.mark.parametrize('path',['../outside.gguf','/tmp/outside.gguf','/models/../outside.gguf'])
def test_adapter_path_cannot_escape_artifact(path):
    with pytest.raises(ValueError):
        DeploymentSpec(id='adapted',artifact_id='a',public_alias='a',backend='llama_cpp',
            executable='/bin/echo',lora_adapter=path)


def test_kernel_releases_reservation_after_process_death(tmp_path):
    paths=LabPaths.discover(repo_root=tmp_path,data_root=tmp_path/'data');paths.initialize()
    code='''
import sys,time
from llm_lab.paths import LabPaths
from llm_lab.training_runtime import training_lease
with training_lease(LabPaths.discover(data_root=sys.argv[1])):
 print('locked',flush=True)
 time.sleep(60)
'''
    child=subprocess.Popen([sys.executable,'-c',code,str(paths.data_root)],stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='locked'
        with pytest.raises(DeploymentError):
            assert_training_idle(paths)
    finally:
        child.kill();child.wait(timeout=10)
    assert_training_idle(paths)
