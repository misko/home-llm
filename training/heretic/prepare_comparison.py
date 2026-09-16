"""Prepare matched upstream/++ profiles without launching a GPU process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_lab.comparison.contracts import atomic_json, file_hash, load_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream_launcher', type=Path)
    parser.add_argument('comparison', type=Path)
    args = parser.parse_args()
    upstream = json.loads(args.upstream_launcher.read_text())
    repo = Path(upstream.get('code_root', upstream['repo_root']))
    calibration = Path(upstream['data_root']) / 'tools/comparison/nli-judge-calibration.json'
    if not json.loads(calibration.read_text())['calibration_passed']:
        raise ValueError('Semantic judge calibration has not passed')
    upstream_root = Path(upstream['run_dir'])
    plus_root = upstream_root.with_name(upstream_root.name + '-plus')
    if (upstream_root / 'study').exists() or (plus_root / 'study').exists():
        raise ValueError('Do not rewrite profiles after a search has started')
    upstream.update(variant='upstream', search_budget_seconds=14400, capability_tolerance=.05,
                    environment_lock_sha256=file_hash(Path(upstream['environment_lock'])))
    execution = json.loads((args.comparison / 'execution.json').read_text())
    upstream.update(humandescent=execution['humandescent'], sharing_policy=execution['sharing_policy'])
    upstream['wandb']['group'] = 'qwen-comparison-v0.1'
    atomic_json(args.upstream_launcher, upstream)
    plus = json.loads(json.dumps(upstream))
    plus.update(run_dir=str(plus_root), variant='plus')
    plus['wandb'].update(id='qwen38-fw-heretic-plus-20260913', name='qwen3.8-27b-fineweb-heretic-plus')
    plus_root.mkdir(parents=True, exist_ok=True)
    atomic_json(plus_root / 'launcher.json', plus)
    base = (upstream_root / 'config.toml').read_text()
    # Keep the exact FineWeb input path, change only output/study paths.
    profile = base.replace(str(upstream_root / 'study'), str(plus_root / 'study')).replace(str(upstream_root / 'heretic-adapter'), str(plus_root / 'heretic-adapter'))
    plugin = repo / 'training/heretic/plus.py'
    refs = repo / 'benchmarks/qwen-comparison/semantic-development.json'
    cases_file = args.comparison / 'cases.jsonl'
    cases = load_cases(cases_file)
    selected = []
    for task in ['ifeval', 'gsm8k', 'grounded-local', 'tool-json-local']:
        eligible = [c for c in cases if c.task == task and c.partition == 'development' and not c.tools and len(c.messages) == 1]
        selected.extend(c.id for c in sorted(eligible, key=lambda c:(len(c.messages[0]['content']), c.id))[:8])
    if len(selected) != 32:
        raise ValueError('Expected 32 capability development cases')
    scorers = '\nscorers = [\n' + ',\n'.join([
        '  { plugin = ' + json.dumps(str(plugin) + ':SemanticTaskLoss') + ', optimization = "minimize" }',
        '  { plugin = ' + json.dumps(str(plugin) + ':MultiPositionKL') + ', optimization = "minimize" }',
        '  { plugin = ' + json.dumps(str(plugin) + ':CapabilityRegression') + ', optimization = "none" }',
        '  { plugin = "heretic.scorers.keyword_rate.KeywordRate", optimization = "none" }',
    ]) + '\n]\n'
    profile = profile.replace('\n[good_prompts]', scorers + '\n[good_prompts]', 1)
    settings = {
        'SemanticTaskLoss': {'references_file':str(refs), 'references_sha256':file_hash(refs),
            'judge_path':str(calibration.parent / 'nli-judge'), 'judge_manifest':str(calibration),
            'judge_manifest_sha256':file_hash(calibration), 'threshold':.7},
        'MultiPositionKL': {'references_file':str(refs), 'references_sha256':file_hash(refs), 'max_positions':4, 'max_context':512},
        'CapabilityRegression': {'cases_file':str(cases_file.resolve()), 'cases_sha256':file_hash(cases_file), 'case_ids':selected},
    }
    for name, values in settings.items():
        profile += '\n[scorer.' + name + ']\n'
        profile += '\n'.join(key + ' = ' + json.dumps(value) for key, value in values.items()) + '\n'
    (plus_root / 'config.toml').write_text(profile)
    atomic_json(args.comparison / 'search-profiles.json', {'upstream_launcher':str(args.upstream_launcher.resolve()),
        'plus_launcher':str(plus_root / 'launcher.json'), 'budget_seconds_each':14400,
        'max_trials_each':32, 'capability_tolerance':.05,
        'semantic_scope':'Calibrated entailment on 12 bounded English development tasks; not a universal quality judge.',
        'plus_plugin_sha256':file_hash(plugin), 'development_case_ids':selected})
    print(plus_root / 'launcher.json')


if __name__ == '__main__':
    main()
