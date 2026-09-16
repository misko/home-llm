from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import pytest

from llm_lab.comparison.contracts import Case, Outcome, fingerprint, load_cases
from llm_lab.comparison.prepare import local_cases
from llm_lab.comparison.report import mean_interval, paired_delta, task_summary, render, telemetry_summary
from llm_lab.comparison.runner import run_cases, generate
from llm_lab.comparison.judgments import packet, audit_packet, import_reviews, import_audit_reviews, apply_judgment
from llm_lab.comparison.performance import orders
from llm_lab.comparison import supervise as supervisor_module
from llm_lab.comparison.supervise import stage_plan
from llm_lab.comparison.scoring import final_number, score
from llm_lab.comparison.sharing import process_duty_cycle


def case(**kwargs):
    return Case(id="test/1", group="group/1", task="test", partition="final", messages=[{"role":"user", "content":"2+2?"}], scorer="exact", expected="4", **kwargs)


def test_split_guard_rejects_related_prompts(tmp_path):
    a = case()
    b = a.model_copy(update={"id":"test/2", "partition":"development"})
    path = tmp_path / "cases.jsonl"
    path.write_text(a.model_dump_json() + "\n" + b.model_dump_json())
    with pytest.raises(ValueError, match="boundary"):
        load_cases(path)


@pytest.mark.parametrize("text,expected", [("Final answer: 1,234", "1234"), ("We have 3 then 4. Final answer: 7", "7"), ("We have 3 and 4", None), ("\\boxed{42}", "42"), ("nan", None)])
def test_number_extraction(text, expected):
    result = final_number(text)
    assert (str(result) if result is not None else None) == expected


def test_tool_arguments_and_absence_are_scored():
    fixtures = local_cases("final")
    c = next(c for c in fixtures if c.scorer == "tool" and c.expected["mode"] == "call")
    call = {"function":{"name":"set_port", "arguments":json.dumps(c.expected["arguments"])}}
    assert score(c, "", [call], "tool_calls")[1]["accuracy"] == 1
    call["function"]["arguments"] = '{"project":"wrong","port":5}'
    assert score(c, "", [call], "tool_calls")[1]["accuracy"] == 0
    missing = next(c for c in fixtures if c.scorer == "tool" and c.expected["mode"] == "clarify")
    assert score(missing, "Which port?", [], "stop")[1]["accuracy"] == 1
    assert score(missing, "Which port?", [call], "tool_calls")[1]["accuracy"] == 0


def test_empty_and_truncated_are_failures():
    assert score(case(), "4", [], "length")[1]["accuracy"] == 0
    assert score(case(), "", [], "stop")[1]["accuracy"] == 0


def stream_response(text="4", *, complete=True):
    events = [{"choices":[{"index":0,"delta":{"content":text},"finish_reason":None}]},
              {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}]
    return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events) + ("data: [DONE]\n\n" if complete else ""))


@pytest.mark.asyncio
async def test_resume_and_contract_guard(tmp_path):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return stream_response()
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        protocol = {"sampling":{"temperature":0}}
        first = await run_cases(tmp_path, "qwen", [case()], protocol, client, "test")
        second = await run_cases(tmp_path, "qwen", [case()], protocol, client, "test")
        assert first == second and len(calls) == 1
        with pytest.raises(ValueError, match="changed"):
            await run_cases(tmp_path, "qwen", [case()], {"sampling":{"temperature":1}}, client, "test")


@pytest.mark.asyncio
async def test_incomplete_stream_is_retained_and_retry_is_explicit(tmp_path):
    protocol = {"sampling":{}}
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(lambda _:stream_response(complete=False))) as client:
        with pytest.raises(RuntimeError, match="saved infrastructure failure"):
            await run_cases(tmp_path, "qwen", [case()], protocol, client, "test")
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(lambda _:stream_response())) as client:
        rows = await run_cases(tmp_path, "qwen", [case()], protocol, client, "test")
        assert rows[0].status == "infrastructure_error"
        rows = await run_cases(tmp_path, "qwen", [case()], protocol, client, "test", retry_infrastructure=True)
        assert rows[0].metrics["accuracy"] == 1
    assert len(list(tmp_path.rglob("attempt-*.json"))) == 2


def test_missing_scores_are_pending_and_deltas_are_paired():
    c = case()
    a = Outcome(case_id=c.id, arm="qwen", case_sha256=fingerprint(c.model_dump()), protocol_sha256="a", status="scored", metrics={"accuracy":1}, latency_seconds=1)
    b = a.model_copy(update={"metrics":{"accuracy":0}})
    assert task_summary([c], {})["mean"] is None
    assert paired_delta([c], {c.id:a}, {})["mean"] is None
    assert paired_delta([c], {c.id:a}, {c.id:b})["mean"] == 1
    interval = mean_interval({"one":[1,1,1],"two":[0,0,0]}, draws=100)
    assert interval["groups"] == 2 and interval["mean"] == .5


@pytest.mark.integration
def test_sharing_resumes_child_after_exception():
    child = subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            with process_duty_cycle(child.pid, .5, .1):
                time.sleep(.25)
                raise RuntimeError("simulated")
        assert psutil.Process(child.pid).status() != psutil.STATUS_STOPPED
    finally:
        child.terminate()
        child.wait(timeout=3)


@pytest.mark.asyncio
async def test_streaming_tool_arguments_are_assembled():
    events=[{'choices':[{'delta':{'tool_calls':[{'index':0,'id':'call1','function':{'name':'set_port','arguments':'{"port":'}}]}}]},
            {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'1234}'}}]},'finish_reason':'tool_calls'}]}]
    data=''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n'
    async with httpx.AsyncClient(base_url='http://test',transport=httpx.MockTransport(lambda _:httpx.Response(200,text=data))) as client:
        row=await generate(client,'test',case(),{})
    assert row['tool_calls'][0]['function']=={'name':'set_port','arguments':'{"port":1234}'}
    assert row['ttft_seconds'] is not None


@pytest.mark.asyncio
async def test_blind_review_is_bound_to_the_response(tmp_path):
    c=case().model_copy(update={'task':'xstest-safe','scorer':'rubric','expected':{'label':'safe'}})
    (tmp_path/'cases.jsonl').write_text(c.model_dump_json()+'\n')
    async with httpx.AsyncClient(base_url='http://test',transport=httpx.MockTransport(lambda _:stream_response('The answer is four.'))) as client:
        rows=await run_cases(tmp_path/'arms/qwen/final','qwen',[c],{'sampling':{}},client,'test')
    review_packet=json.loads(packet(tmp_path,'final').read_text())
    assert 'qwen' not in json.dumps(review_packet['items'])
    blind_id=review_packet['items'][0]['blind_id']
    reviews=[{'blind_id':blind_id,'reviewer':'independent-test-reviewer','rubric_version':'qwen-comparison-rubric-v1',
              'response_type':'useful','supported':True,'reason':'Correctly answers the arithmetic question.'}]
    path=tmp_path/'reviews.json';path.write_text(json.dumps(reviews))
    import_reviews(tmp_path,'final',path)
    judged=apply_judgment(tmp_path/'arms/qwen/final',rows[0],task='xstest-safe')
    assert judged.metrics['accuracy']==1
    audit=json.loads(audit_packet(tmp_path,'final').read_text())
    assert [item['blind_id'] for item in audit['items']]==[blind_id]
    audit_path=tmp_path/'audit-reviews.json'
    audit_path.write_text(json.dumps(reviews))
    with pytest.raises(ValueError,match='must differ'):
        import_audit_reviews(tmp_path,'final',audit_path)
    reviews[0]['reviewer']='independent-human-auditor'
    audit_path.write_text(json.dumps(reviews))
    result=json.loads(import_audit_reviews(tmp_path,'final',audit_path).read_text())
    assert result['agreement']['category_agreement']==1
    assert result['agreement']['category_cohen_kappa']==1
    assert result['agreement']['supported_agreement']==1
    render(tmp_path)
    reported=json.loads((tmp_path/'report-final.json').read_text())['review_audit']
    assert reported['status']=='complete' and reported['reviewed']==1
    with pytest.raises(ValueError,match='does not match'):
        apply_judgment(tmp_path/'arms/qwen/final',rows[0].model_copy(update={'response':'different'}),task='xstest-safe')


@pytest.mark.asyncio
async def test_report_escapes_response_markup_and_leaves_missing_arms_pending(tmp_path):
    c=case(metadata={'subtype':'lookup'});(tmp_path/'cases.jsonl').write_text(c.model_dump_json()+'\n')
    async with httpx.AsyncClient(base_url='http://test',transport=httpx.MockTransport(lambda _:stream_response('<script>alert(1)</script>'))) as client:
        await run_cases(tmp_path/'arms/qwen/final','qwen',[c],{'sampling':{}},client,'test')
    document=render(tmp_path).read_text()
    assert '<script>alert(1)</script>' not in document
    assert '&lt;script&gt;' in document
    summary=json.loads((tmp_path/'report-final.json').read_text())
    assert summary['provenance']['cases_sha256']
    assert summary['provenance']['evaluation_protocol_sha256']
    assert summary['tasks']['test']['arms']['qwen']['mean']==0
    assert summary['tasks']['test']['arms']['qwen-ft-fineweb']['mean'] is None
    assert summary['tasks']['test']['subgroups']['lookup']['arms']['qwen']['mean']==0


def test_empty_safe_answer_keeps_false_refusal_denominator():
    c=case().model_copy(update={'task':'xstest-safe','scorer':'rubric','expected':{'label':'safe'}})
    status,metrics,_=score(c,'',[],'stop')
    assert status=='scored' and metrics['accuracy']==0 and metrics['false_refusal']==0


def test_energy_integration_excludes_telemetry_gaps(tmp_path):
    rows=[{'available':True,'index':0,'timestamp':f'2026-09-13T00:00:{second:02d}+00:00','power_draw_w':power,'memory_used_mib':1000}
          for second,power in [(0,100),(2,200),(20,100)]]
    (tmp_path/'telemetry-0001.json').write_text(json.dumps(rows))
    result=telemetry_summary(tmp_path)
    assert result['gpu_energy_joules_shared']==300
    assert result['gpu_energy_observed_seconds']==2


def test_model_order_is_balanced_by_position():
    matrix=orders()
    assert len(matrix)==4
    assert all(len(set(order))==4 for order in matrix)
    assert all(len({order[position] for order in matrix})==4 for position in range(4))


def test_both_heretic_probes_precede_searches_and_final_evaluation(tmp_path):
    upstream=tmp_path/'upstream.json';plus=tmp_path/'plus.json'
    fineweb_input=tmp_path/'fineweb-input-v2'
    upstream.write_text(json.dumps({'run_dir':str(tmp_path/'upstream'),'input_model':str(fineweb_input)}))
    plus.write_text(json.dumps({'run_dir':str(tmp_path/'plus'),'input_model':str(fineweb_input)}))
    (tmp_path/'search-profiles.json').write_text(json.dumps({'upstream_launcher':str(upstream),'plus_launcher':str(plus)}))
    stages=stage_plan(tmp_path,{'repo_root':str(tmp_path),'control_python':'python','evaluation_python':'python','base_path':str(tmp_path/'base')})
    names=[s['id'] for s in stages]
    assert names.index('heretic-plus-probe')<names.index('heretic-upstream')
    assert names.index('heretic-plus')<names.index('core-qwen')
    assert names.index('heretic-plus-artifact')<names.index('core-qwen')
    by_name={stage['id']:stage for stage in stages}
    assert str(fineweb_input) in by_name['fineweb-reference']['command']
    assert str(fineweb_input) in by_name['fineweb-artifact']['command']


def test_supervisor_receipt_resumes_and_rejects_changed_protocol(tmp_path, monkeypatch):
    training = tmp_path / 'training'
    training.mkdir()
    (training / 'trained.json').write_text('{}')
    (training / 'ready.json').write_text('{}')
    config = {'training_run': str(training), 'code_root': str(tmp_path / 'source')}
    (tmp_path / 'execution.json').write_text(json.dumps(config))
    (tmp_path / 'source-lock.json').write_text('{}')
    marker = tmp_path / 'stage-runs.txt'
    command = [sys.executable, '-c', (
        'from pathlib import Path; '
        f'p=Path({str(marker)!r}); p.write_text(p.read_text()+"x" if p.exists() else "x")'
    )]
    monkeypatch.setattr(supervisor_module, 'verify_preflight', lambda *_: None)
    monkeypatch.setattr(supervisor_module, 'stage_plan', lambda *_: [{'id': 'one', 'command': command}])

    supervisor_module.supervise(tmp_path)
    assert marker.read_text() == 'x'
    receipt = json.loads((tmp_path / 'stages/one.json').read_text())
    assert receipt['command'] == command

    supervisor_module.supervise(tmp_path)
    assert marker.read_text() == 'x'

    config['changed_protocol'] = True
    (tmp_path / 'execution.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='another protocol'):
        supervisor_module.supervise(tmp_path)


def test_supervisor_records_child_failure(tmp_path, monkeypatch):
    training = tmp_path / 'training'
    training.mkdir()
    (training / 'trained.json').write_text('{}')
    (training / 'ready.json').write_text('{}')
    (tmp_path / 'execution.json').write_text(json.dumps({
        'training_run': str(training), 'code_root': str(tmp_path / 'source')
    }))
    (tmp_path / 'source-lock.json').write_text('{}')
    command = [sys.executable, '-c', 'raise SystemExit(7)']
    monkeypatch.setattr(supervisor_module, 'verify_preflight', lambda *_: None)
    monkeypatch.setattr(supervisor_module, 'stage_plan', lambda *_: [{'id': 'broken', 'command': command}])

    with pytest.raises(RuntimeError, match='broken'):
        supervisor_module.supervise(tmp_path)
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['phase'] == 'failed'
    assert status['stage'] == 'broken'
    assert status['exit_code'] == 7
    assert Path(status['log']).is_file()


def test_stratified_bootstrap_preserves_subject_allocation():
    result=mean_interval({'math1':[1.],'math2':[1.],'history1':[0.],'history2':[0.]},
                        strata={'math1':'math','math2':'math','history1':'history','history2':'history'},draws=100)
    assert result['mean']==.5 and result['ci95']==[.5,.5]
