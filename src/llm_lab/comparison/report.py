from __future__ import annotations

import html
import json
import random
import statistics
from datetime import datetime
from collections import defaultdict
from pathlib import Path

from .contracts import Case, Outcome, atomic_json, file_hash, fingerprint, load_cases
from .runner import latest_outcomes
from .judgments import apply_judgment
from .humandescent import read_samples


ARMS = {"qwen": "Qwen", "qwen-ft-fineweb": "FT FineWeb", "qwen-heretic": "Heretic", "qwen-heretic-plus": "Heretic++"}


def mean_interval(values: dict[str, list[float]], *, strata: dict[str,str] | None=None, seed=20260913, draws=2000) -> dict:
    """Bootstrap prompt groups, preserving within-group dependence."""
    if not values:
        return {"mean": None, "ci95": None, "groups": 0}
    groups = list(values.values())
    mean = sum(map(sum, groups)) / sum(map(len, groups))
    if len(groups) < 2:
        return {"mean": mean, "ci95": None, "groups": len(groups)}
    rng = random.Random(seed)
    by_stratum=defaultdict(list)
    for group,items in values.items():by_stratum[(strata or {}).get(group,'all')].append(items)
    samples = []
    for _ in range(draws):
        sampled = [item for population in by_stratum.values() for item in rng.choices(population,k=len(population))]
        samples.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
    samples.sort()
    return {"mean": mean, "ci95": [samples[int(draws * 0.025)], samples[min(draws - 1, int(draws * 0.975))]], "groups": len(groups)}


def task_summary(cases: list[Case], outcomes: dict[str, Outcome], metric="accuracy") -> dict:
    values = defaultdict(list)
    counts = defaultdict(int)
    for case in cases:
        row = outcomes.get(case.id)
        status = row.status if row else "not_run"
        counts[status] += 1
        if row and row.status == "scored" and metric in row.metrics:
            values[case.group].append(row.metrics[metric])
    # No partial denominator masquerades as a complete task score.
    complete = sum(len(v) for v in values.values()) == len(cases)
    strata={c.group:c.metadata.get('subject','all') for c in cases}
    result = mean_interval(values,strata=strata) if complete else {"mean": None, "ci95": None, "groups": len(values)}
    return {**result, "complete": complete, "expected": len(cases), "scored_metric": sum(map(len, values.values())), "statuses": dict(counts)}


def paired_delta(cases: list[Case], left: dict[str, Outcome], right: dict[str, Outcome], metric="accuracy") -> dict:
    groups = defaultdict(list)
    for case in cases:
        a, b = left.get(case.id), right.get(case.id)
        if not a or not b or a.status != "scored" or b.status != "scored" or metric not in a.metrics or metric not in b.metrics:
            return {"mean": None, "ci95": None, "groups": 0}
        groups[case.group].append(a.metrics[metric] - b.metrics[metric])
    return mean_interval(groups,strata={c.group:c.metadata.get('subject','all') for c in cases})


def telemetry_summary(run_root: Path) -> dict:
    energy=0.;observed=0.;peak=None
    for path in run_root.glob('telemetry-*.json'):
        samples=[s for s in json.loads(path.read_text()) if s['available'] and s.get('index')==0]
        for row in samples:
            memory=row.get('memory_used_mib')
            if memory is not None:peak=max(peak or 0,memory)
        for before,after in zip(samples,samples[1:]):
            seconds=(datetime.fromisoformat(after['timestamp'])-datetime.fromisoformat(before['timestamp'])).total_seconds()
            powers=[before.get('power_draw_w'),after.get('power_draw_w')]
            if 0 < seconds <= 5 and all(v is not None for v in powers):
                energy+=seconds*sum(powers)/2;observed+=seconds
    return {'gpu_energy_joules_shared':energy if observed else None,'gpu_energy_observed_seconds':observed,
            'gpu_peak_memory_mib_shared':peak}


def validate_run(root: Path, expected_cases: list[Case]) -> dict[str, Outcome]:
    run_path = root / "run.json"
    if not run_path.exists():
        return {}
    run = json.loads(run_path.read_text())
    contract = {key: run[key] for key in ("arm", "protocol", "cases")}
    if fingerprint(contract) != run["sha256"]:
        raise ValueError("Run provenance was modified")
    allowed = {c.id: fingerprint(c.model_dump()) for c in expected_cases}
    outcomes = latest_outcomes(root)
    for case_id, row in outcomes.items():
        if row.arm != run["arm"] or row.protocol_sha256 != run["sha256"] or allowed.get(case_id) != row.case_sha256:
            raise ValueError("Result does not match frozen protocol or case")
        outcomes[case_id] = apply_judgment(root, row, task=next(c.task for c in expected_cases if c.id == case_id))
    return outcomes


def scorecard_record(selected: list[Case], results: dict[str, dict[str, Outcome]], metric="accuracy") -> dict:
    arms = {arm: task_summary(selected, results[arm], metric=metric) for arm in ARMS}
    contrasts = {label: paired_delta(selected, results[left], results[right], metric=metric)
                 for label, left, right in [
                     ("fineweb_minus_original", "qwen-ft-fineweb", "qwen"),
                     ("heretic_minus_fineweb", "qwen-heretic", "qwen-ft-fineweb"),
                     ("plus_minus_heretic", "qwen-heretic-plus", "qwen-heretic"),
                     ("plus_minus_fineweb", "qwen-heretic-plus", "qwen-ft-fineweb"),
                 ]}
    return {"arms": arms, "plus_minus_heretic": contrasts["plus_minus_heretic"], "contrasts": contrasts}


def scorecard_row(label: str, selected: list[Case], record: dict, *, lower_is_better=False) -> str:
    cells = [f"<th>{html.escape(label)}<small>{len(selected)} cases</small></th>"]
    for summary in record["arms"].values():
        if summary["mean"] is None:
            cell = f'<span class="pending">Pending</span><small>{summary["scored_metric"]}/{summary["expected"]} scored</small>'
        else:
            cell = f'<strong>{summary["mean"]:.1%}</strong>'
            if summary["ci95"]:
                cell += f'<small>{summary["ci95"][0]:.1%}–{summary["ci95"][1]:.1%}</small>'
        cells.append(f"<td>{cell}</td>")
    delta = record["plus_minus_heretic"]
    difference = "Pending" if delta["mean"] is None else f'{delta["mean"] * 100:+.1f} pp'
    if delta["ci95"]:
        difference += f'<small>{delta["ci95"][0]*100:+.1f} to {delta["ci95"][1]*100:+.1f} pp</small>'
    if lower_is_better:
        difference += '<small>lower is better</small>'
    cells.append(f"<td>{difference}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def render(root: Path, *, partition="final") -> Path:
    cases = [c for c in load_cases(root / "cases.jsonl") if c.partition == partition]
    results = {arm: validate_run(root / "arms" / arm / partition, cases) for arm in ARMS}
    contracts = []
    for arm in ARMS:
        path = root / "arms" / arm / partition / "run.json"
        if path.exists():
            protocol = json.loads(path.read_text())["protocol"]
            # Artifact identity is arm-specific. Everything else must match.
            contracts.append(fingerprint({k: v for k, v in protocol.items() if k != "artifact"}))
    if len(set(contracts)) > 1:
        raise ValueError("Cannot compare runs with different evaluation protocols")
    provenance = {
        "cases_sha256": file_hash(root / "cases.jsonl"),
        "evaluation_protocol_sha256": contracts[0] if contracts else None,
    }
    for name in ("dataset-lock.json", "source-lock.json", "validation.json"):
        path = root / name
        if path.exists():
            provenance[name.removesuffix(".json").replace("-", "_") + "_sha256"] = file_hash(path)
    source_lock = root / "source-lock.json"
    if source_lock.exists():
        provenance["source_fingerprint"] = fingerprint(json.loads(source_lock.read_text()))
    execution = root / "execution.json"
    if execution.exists():
        provenance["base_revision"] = json.loads(execution.read_text()).get("base_revision")
    table = []
    report = {"partition": partition, "headline_composite": None, "provenance": provenance,
              "tasks": {}, "performance": {}}
    for task in sorted({c.task for c in cases}):
        selected = [c for c in cases if c.task == task]
        record = scorecard_record(selected, results)
        report["tasks"][task] = record
        table.append(scorecard_row(task, selected, record))
        subtypes = sorted({c.metadata.get("subtype") for c in selected if c.metadata.get("subtype")})
        if subtypes:
            record["subgroups"] = {}
            for subtype in subtypes:
                subgroup = [c for c in selected if c.metadata.get("subtype") == subtype]
                subgroup_record = scorecard_record(subgroup, results)
                record["subgroups"][subtype] = subgroup_record
                table.append(scorecard_row(f"{task} · {subtype}", subgroup, subgroup_record))
        if task == 'xstest-safe':
            refusal = scorecard_record(selected, results, metric="false_refusal")
            record["false_refusal"] = refusal
            table.append(scorecard_row("xstest-safe · false refusal ↓", selected, refusal, lower_is_better=True))
    failures = []
    performance_rows = []
    for arm, outcomes in results.items():
        timings = [r.latency_seconds for r in outcomes.values() if r.status != "infrastructure_error"]
        ttfts = [r.ttft_seconds for r in outcomes.values() if r.ttft_seconds is not None]
        report["performance"][arm] = {"observed_responses": len(timings), "latency_p50_seconds": statistics.median(timings) if timings else None,
                                    "latency_p95_seconds": sorted(timings)[min(len(timings)-1, int(len(timings)*.95))] if timings else None,
                                    "ttft_p50_seconds": statistics.median(ttfts) if ttfts else None}
        report["performance"][arm]["expected_responses"] = len(cases)
        report["performance"][arm]["infrastructure_error_count"] = sum(r.status == "infrastructure_error" for r in outcomes.values())
        report["performance"][arm]["pending_judgment_count"] = sum(r.status == "pending_judgment" for r in outcomes.values())
        report["performance"][arm]["truncated_rate"] = (sum(r.metrics.get("truncated", 0) for r in outcomes.values()) / len(cases)) if len(outcomes) == len(cases) else None
        report["performance"][arm]["empty_rate"] = (sum(r.metrics.get("empty", 0) for r in outcomes.values()) / len(cases)) if len(outcomes) == len(cases) else None
        tokens=sum(max(0,r.usage.get('completion_tokens',0)-1) for r in outcomes.values() if r.status!='infrastructure_error' and r.ttft_seconds is not None)
        decode_seconds=sum(max(0,r.latency_seconds-r.ttft_seconds) for r in outcomes.values() if r.ttft_seconds is not None)
        report['performance'][arm]['decode_tokens_per_second']=tokens/decode_seconds if decode_seconds and tokens else None
        native=[r.server_timings for r in outcomes.values() if r.server_timings.get('prompt_ms',0)>0]
        report['performance'][arm]['prefill_tokens_per_second']=sum(t.get('prompt_n',0) for t in native)/(sum(t['prompt_ms'] for t in native)/1000) if native else None
        native_decode=[r.server_timings for r in outcomes.values() if r.server_timings.get('predicted_ms',0)>0]
        if native_decode:
            report['performance'][arm]['decode_tokens_per_second']=sum(t.get('predicted_n',0) for t in native_decode)/(sum(t['predicted_ms'] for t in native_decode)/1000)
        report['performance'][arm].update(telemetry_summary(root/'arms'/arm/partition))
        hum=sorted(r['latency_seconds'] for r in read_samples(root/'arms'/arm/partition) if 'latency_seconds' in r)
        report['performance'][arm]['humandescent_samples']=len(hum)
        report['performance'][arm]['humandescent_p50_seconds']=statistics.median(hum) if hum else None
        report['performance'][arm]['humandescent_p95_seconds']=hum[min(len(hum)-1,int(len(hum)*.95))] if hum else None
        correct=sum(r.metrics.get('accuracy',0) for r in outcomes.values() if r.status=='scored')
        complete=len(outcomes)==len(cases) and all(r.status=='scored' for r in outcomes.values())
        energy=report['performance'][arm]['gpu_energy_joules_shared']
        report['performance'][arm]['gpu_joules_per_correct_shared']=energy/correct if complete and correct and energy is not None else None
        nll_path=root/'arms'/arm/'nll.json'
        report['performance'][arm]['fineweb_nll'] = json.loads(nll_path.read_text())['nll'] if nll_path.exists() else None
        quant_path=root/'references'/arm/'quantization.json'
        report['performance'][arm]['bf16_to_gguf_kl']=json.loads(quant_path.read_text())['bf16_to_gguf_kl'] if quant_path.exists() else None
        for case in cases:
            row = outcomes.get(case.id)
            if row and (row.status != "scored" or row.metrics.get("accuracy") != 1):
                title = html.escape(f"{ARMS[arm]} · {case.id} · {row.status}")
                body = html.escape(json.dumps({"messages": case.messages, "expected": case.expected, "result": row.model_dump()}, indent=2, ensure_ascii=False))
                failures.append(f"<details><summary>{title}</summary><pre>{body}</pre></details>")
    report['search'] = {}
    search_rows = []
    execution_path = root/'execution.json'
    search_root = (Path(json.loads(execution_path.read_text())['data_root'])/'runs/heretic'
                   if execution_path.exists() else root/'heretic')
    profile_path = root/'search-profiles.json'
    if profile_path.exists():
        profiles = json.loads(profile_path.read_text())
        search_results = {
            'qwen-heretic': Path(json.loads(Path(profiles['upstream_launcher']).read_text())['run_dir'])/'result.json',
            'qwen-heretic-plus': Path(json.loads(Path(profiles['plus_launcher']).read_text())['run_dir'])/'result.json',
        }
    else:
        search_results = {
            'qwen-heretic': search_root/'qwen38-fineweb-20260913/result.json',
            'qwen-heretic-plus': search_root/'qwen38-fineweb-20260913-plus/result.json',
        }
    search_metrics = [
        ('completed_trials', 'Completed trials ↑'),
        ('consumed_hours', 'GPU-job wall time (h)'),
        ('gpu_energy_kwh_shared', 'Shared GPU energy (kWh)'),
        ('mean_gpu_utilization_percent', 'Mean GPU utilization (%)'),
        ('minimum_free_gpu_mib', 'Minimum free VRAM (MiB)'),
        ('humandescent_p95_seconds', 'Humandescent p95 (s) ↓'),
    ]
    for arm, result_path in search_results.items():
        if result_path.exists():
            value = json.loads(result_path.read_text())
            shared = value['shared_gpu']
            report['search'][arm] = {
                'completed_trials': value['trials']['completed_trials'],
                'total_trials': value['trials']['total_trials'],
                'trial_states': value['trials']['trial_states'],
                'selected_trial_numbers': value['trials']['selected_trial_numbers'],
                'consumed_hours': value['search_budget']['consumed_seconds']/3600,
                'allocated_hours': value['search_budget']['budget_seconds']/3600,
                'gpu_energy_kwh_shared': (shared['gpu_energy_joules_shared']/3_600_000
                                          if shared['gpu_energy_joules_shared'] is not None else None),
                'mean_gpu_utilization_percent': shared['mean_gpu_utilization_percent'],
                'minimum_free_gpu_mib': shared['minimum_free_gpu_mib'],
                'humandescent_samples': shared['humandescent_samples'],
                'humandescent_p95_seconds': shared['humandescent_p95_seconds'],
                'sharing_policy_passed': shared['passed'],
                'result_sha256': file_hash(result_path),
            }
        else:
            report['search'][arm] = {key:None for key,_ in search_metrics}
    for metric,label in search_metrics:
        cells=[]
        for arm in ('qwen-heretic','qwen-heretic-plus'):
            value=report['search'][arm].get(metric)
            cells.append('<td>'+('Pending' if value is None else f'{value:.3f}')+'</td>')
        search_rows.append('<tr><th>'+label+'</th>'+''.join(cells)+'</tr>')
    audit_packet_path=root/f'review-audit-packet-{partition}.json'
    audit_result_path=root/f'review-audit-results-{partition}.json'
    if audit_result_path.exists():
        audit_result=json.loads(audit_result_path.read_text())
        report['review_audit']={**audit_result['agreement'], 'status':'complete',
                                'audit_packet_sha256':audit_result['audit_packet_sha256'],
                                'review_source_sha256':audit_result['review_source_sha256']}
    elif audit_packet_path.exists():
        audit=json.loads(audit_packet_path.read_text())
        report['review_audit']={'status':'pending', 'selected':len(audit['items']), 'reviewed':0,
                                'category_agreement':None, 'category_cohen_kappa':None,
                                'supported_agreement':None, 'audit_packet_sha256':file_hash(audit_packet_path)}
    else:
        report['review_audit']={'status':'pending_packet', 'selected':None, 'reviewed':0,
                                'category_agreement':None, 'category_cohen_kappa':None,
                                'supported_agreement':None}
    for metric,label in [('fineweb_nll','FineWeb NLL ↓'),('latency_p50_seconds','Latency p50 (s) ↓'),('latency_p95_seconds','Latency p95 (s) ↓'),('ttft_p50_seconds','First token p50 (s) ↓'),('prefill_tokens_per_second','Prefill tokens/s ↑'),('decode_tokens_per_second','Decode tokens/s ↑'),('gpu_peak_memory_mib_shared','Peak shared GPU VRAM (MiB)'),('gpu_joules_per_correct_shared','Shared GPU joules/correct ↓'),('humandescent_p50_seconds','Humandescent task p50 (s) ↓'),('humandescent_p95_seconds','Humandescent task p95 (s) ↓')]:
        cells=[]
        for arm in ARMS:
            value=report['performance'][arm][metric]
            cells.append('<td>'+('Pending' if value is None else f'{value:.3f}')+'</td>')
        performance_rows.append('<tr><th>'+label+'</th>'+''.join(cells)+'</tr>')
    report['counterbalanced_performance']={}
    for arm in ARMS:
        blocks=[json.loads(p.read_text()) for p in (root/'performance').glob(f'block-*-{arm}.json')]
        summary={}
        for scenario in ['performance/short','performance/prefill','performance/decode']:
            values={str(b['block']):[r['latency_seconds'] for r in b['rows'] if r['scenario']==scenario] for b in blocks}
            summary[scenario]=mean_interval(values) if len(blocks)==4 else {'mean':None,'ci95':None,'groups':len(blocks)}
        report['counterbalanced_performance'][arm]=summary
    output = root / f"report-{partition}.html"
    headers = "".join(f"<th>{name}</th>" for name in ARMS.values())
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen · Model comparison</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10151b;color:#e5edf5}body{max-width:1200px;margin:64px auto;padding:0 24px}h1{font-size:44px;letter-spacing:-2px;margin-bottom:10px}p{color:#aab7c5;line-height:1.65;max-width:850px}.eyebrow{color:#83e2c3;text-transform:uppercase;font-size:12px;letter-spacing:2px}table{border-collapse:collapse;width:100%;margin:32px 0;background:#171f28;border-radius:12px;overflow:hidden}th,td{padding:18px;text-align:left;border-bottom:1px solid #293440}thead{color:#aab7c5;font-size:13px}tbody th{font-weight:500}small{display:block;color:#8496a8;font-size:12px;margin-top:6px}.pending{color:#8496a8}strong{color:#83e2c3}details{border-bottom:1px solid #293440;padding:16px 0}summary{cursor:pointer}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.5}.scroll{overflow:auto}footer{margin-top:40px;color:#8496a8;font-size:12px}</style>
<div class="eyebrow">LLM Lab · Reproducible evaluation</div><h1>Four models. One protocol.</h1>'''
    document += f"<p>{html.escape(partition.title())} partition · Paired comparisons of original Qwen, FineWeb training, Heretic, and Heretic++. Scores appear only when the full task is scored. Intervals use prompt-group bootstrap; public benchmark contamination remains possible.</p>"
    document += '<div class="scroll"><table><thead><tr><th>Accuracy ↑</th>' + headers + '<th>++ − Heretic</th></tr></thead><tbody>' + "".join(table) + '</tbody></table></div>'
    document += '<h2>Matched Heretic searches</h2><p>Both variants start from the same FineWeb parent and receive four hours of accumulated GPU-job wall time. Completed trials are also shown because the additional Heretic++ scorers consume part of that matched budget.</p><div class="scroll"><table><thead><tr><th>Metric</th><th>Heretic</th><th>Heretic++</th></tr></thead><tbody>'+''.join(search_rows)+'</tbody></table></div>'
    audit=report['review_audit']
    audit_text=('Pending independent review' if audit['status']!='complete' else
                f"{audit['reviewed']} audited responses · category agreement {audit['category_agreement']:.1%} · Cohen’s κ {audit['category_cohen_kappa']:.3f} · support agreement {audit['supported_agreement']:.1%}")
    document += '<h2>Independent review audit</h2><p>'+html.escape(audit_text)+'</p>'
    document += '<h2>Operating measurements</h2><p>Observed request latency on the shared GPU. Compare complete, matched workloads; partial runs have different task mixes.</p><div class="scroll"><table><thead><tr><th>Metric</th>'+headers+'</tr></thead><tbody>'+''.join(performance_rows)+'</tbody></table></div>'
    document += '<h2>Counterbalanced latency</h2><p>Four rotating model-order blocks; three repetitions per scenario per block. Mean request seconds with 95% intervals clustered by block. Warmups are excluded.</p><div class="scroll"><table><thead><tr><th>Scenario</th>'+headers+'</tr></thead><tbody>'
    for scenario in ['performance/short','performance/prefill','performance/decode']:
        cells=[]
        for arm in ARMS:
            value=report['counterbalanced_performance'][arm][scenario]
            text='Pending' if value['mean'] is None else f'{value["mean"]:.3f} s<small>{value["ci95"][0]:.3f}–{value["ci95"][1]:.3f}</small>'
            cells.append('<td>'+text+'</td>')
        document+='<tr><th>'+scenario.removeprefix('performance/')+'</th>'+''.join(cells)+'</tr>'
    document+='</tbody></table></div>'
    document += "<h2>Failure browser</h2>" + ("".join(failures) or "<p>No recorded failures yet. Pending models have not been evaluated.</p>")
    document += "<footer>Separate capability and operating-cost metrics; no composite score. Local subset results, not official leaderboard submissions.</footer></html>"
    output.write_text(document)
    atomic_json(root / f"report-{partition}.json", report)
    return output


def publish(root: Path, *, partition='final'):
    """Publish aggregate metrics only; response-bearing HTML stays local."""
    render(root,partition=partition)
    import wandb
    config=json.loads((root/'execution.json').read_text())
    summary=json.loads((root/f'report-{partition}.json').read_text())
    with wandb.init(**config['wandb'],id='qcomp-scorecard-'+partition,resume='allow',dir=str(root),
                    config={'partition':partition, **summary['provenance']},
                    settings=wandb.Settings(disable_git=True)) as tracking:
        data={}
        for task,record in summary['tasks'].items():
            for arm,values in record['arms'].items():
                prefix=arm+'/'+task
                data[prefix+'/scored']=values['scored_metric']
                data[prefix+'/expected']=values['expected']
                if values['mean'] is not None:data[prefix+'/accuracy']=values['mean']
            for name,delta in record['contrasts'].items():
                if delta['mean'] is not None:data[task+'/'+name]=delta['mean']
            for subgroup,subrecord in record.get('subgroups',{}).items():
                for arm,values in subrecord['arms'].items():
                    if values['mean'] is not None:data[arm+'/'+task+'/subgroup/'+subgroup+'/accuracy']=values['mean']
            if 'false_refusal' in record:
                for arm,values in record['false_refusal']['arms'].items():
                    if values['mean'] is not None:data[arm+'/'+task+'/false_refusal_rate']=values['mean']
        for arm,metrics in summary['performance'].items():
            data.update({arm+'/'+k:v for k,v in metrics.items() if v is not None})
        for arm,metrics in summary['search'].items():
            data.update({arm+'/search/'+k:v for k,v in metrics.items()
                         if isinstance(v,(int,float)) and not isinstance(v,bool)})
        data.update({'review_audit/'+key:value for key,value in summary['review_audit'].items()
                     if isinstance(value,(int,float)) and not isinstance(value,bool)})
        for key, value in summary['provenance'].items():
            if value is not None:
                tracking.summary['provenance/' + key] = value
        for arm,scenarios in summary['counterbalanced_performance'].items():
            for name,metric in scenarios.items():
                if metric['mean'] is not None:data[arm+'/'+name+'/mean_latency_seconds']=metric['mean']
        tracking.log(data)
        artifact=wandb.Artifact('qwen-comparison-aggregate-'+partition,type='evaluation')
        artifact.add_file(str(root/f'report-{partition}.json'))
        tracking.log_artifact(artifact)
        atomic_json(root/f'report-{partition}-tracking.json',{'url':tracking.url})
        return tracking.url
