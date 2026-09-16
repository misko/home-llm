"""Local Heretic integration: shared GPU lease, CPU offload, throttle and W&B."""
import functools
import json
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import atomic_json, digest, identity
from throttle import read_duty_cycle, yield_gpu
from run import (can_finalize_exhausted_study, completed_checkpoint,
                 experiment_fingerprint, validate_install)
from monitoring import SearchMonitors, summarize_search, trial_summary
from llm_lab.paths import LabPaths
from llm_lab.training_runtime import training_lease


def main():
    config = json.loads(Path(sys.argv[1]).read_text()); mode = sys.argv[2]
    validate_install(config); checkpoint = completed_checkpoint(config)
    root = Path(config['run_dir']); source = Path(config['heretic_source'])
    input_model = Path(config['input_model'])
    manifest = json.loads((input_model / 'fineweb-input-manifest.json').read_text())
    if manifest['adapter_sha256'] != digest(checkpoint / 'adapter_model.safetensors'):
        raise RuntimeError('Heretic input does not contain the final FineWeb adapter.')
    fingerprint = experiment_fingerprint(config)
    paths = LabPaths.discover(repo_root=config['repo_root'], data_root=config['data_root'])
    with training_lease(paths):
        import torch
        import optuna
        from heretic.config import Settings
        from heretic.model import Model, AbliterationParameters
        from heretic.utils import Prompt
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        os.chdir(root)
        # Heretic reads config.toml and parses argv; do not leak wrapper arguments.
        sys.argv = ['heretic']
        settings = Settings()
        if Path(settings.model) != input_model:
            raise RuntimeError('Heretic profile points at the wrong input model.')
        original_quantization = Model._get_quantization_config
        def quantization(self, dtype):
            result = original_quantization(self, dtype)
            if result is not None:
                result.llm_int8_enable_fp32_cpu_offload = True
            return result
        Model._get_quantization_config = quantization
        original_init = Model.__init__
        budget_tick = [lambda: None]
        @functools.wraps(original_init)
        def initialize(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Match the benchmark's nonthinking track in both search variants.
            original_template = self.tokenizer.apply_chat_template
            def template(*args, **kwargs):
                kwargs.setdefault('enable_thinking', False)
                return original_template(*args, **kwargs)
            self.tokenizer.apply_chat_template = template
            base = self.model.get_base_model()
            started = [None]
            def before(*_):
                started[0] = time.monotonic()
            def after(*_):
                if started[0] is not None:
                    torch.cuda.synchronize()
                    duty = read_duty_cycle(Path(config['training_run']))
                    yield_gpu(time.monotonic() - started[0], duty)
                    budget_tick[0]()
            base.register_forward_pre_hook(before)
            base.register_forward_hook(after)
        Model.__init__ = initialize
        original_svd = torch.svd_lowrank
        def shared_svd(matrix, *args, **kwargs):
            started = time.monotonic()
            result = original_svd(matrix, *args, **kwargs)
            if matrix.is_cuda:
                torch.cuda.synchronize()
                yield_gpu(time.monotonic() - started, read_duty_cycle(Path(config['training_run'])))
            return result
        torch.svd_lowrank = shared_svd
        if mode == 'probe':
            model = Model(settings)
            prompts = [Prompt(system='You are a helpful assistant.', user='What is 2+2?')]
            inputs, output = model.generate(prompts, max_new_tokens=8)
            if output.shape[1] <= inputs['input_ids'].shape[1]:
                raise RuntimeError('Probe generated no new tokens.')
            residuals = model.get_residuals(prompts)
            if not torch.isfinite(residuals).all():
                raise RuntimeError('Nonfinite probe residuals.')
            if config.get('variant') == 'plus':
                from plus import position_logprobs, kl_from_logprobs
                reference = position_logprobs(model.model, model.tokenizer,
                    [{'role':'user','content':'What is two plus two?'}], 'The answer is four.', 2, 512)
                if abs(kl_from_logprobs(reference, reference)) > 1e-6:
                    raise RuntimeError('Heretic++ teacher-forced KL identity probe failed.')
            # Exercise actual intervention workspaces on disposable probe weights.
            directions = torch.nn.functional.normalize(residuals[0], dim=-1)
            parameters = {name:AbliterationParameters(0.02,32,0.01,64)
                for name in model.get_abliterable_components()}
            model.abliterate(directions, None, parameters)
            model.generate(prompts, max_new_tokens=8)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0] < 1_500_000_000:
                raise RuntimeError('GPU probe left less than 1.5 GB free; reduce max_memory before running.')
            atomic_json(root / 'gpu-probe.json', {'fingerprint':fingerprint,
                'layers':len(model.get_layers()), 'generation_type':type(output).__name__,
                'residual_shape':list(residuals.shape), 'free_vram_bytes':torch.cuda.mem_get_info()[0],
                'allocated_bytes':torch.cuda.memory_allocated(), 'heretic_commit':config['heretic_commit']})
            print('Heretic GPU load, generation, residual and intervention probe passed', flush=True)
            return
        probe = json.loads((root / 'gpu-probe.json').read_text())
        if probe['fingerprint'] != fingerprint:
            raise RuntimeError('Heretic profile changed since the GPU probe.')
        import wandb
        import heretic.main
        if config.get('variant') == 'plus':
            from plus import feasible_pareto
            tolerance = float(config.get('capability_tolerance', 0.05))
            # Filter all complete trials before constructing a feasible Pareto front.
            # Filtering only the unconstrained Pareto front can lose valid candidates.
            optuna.Study.best_trials = property(lambda study: feasible_pareto(study.trials, study.directions, tolerance))
        budget_path = root / 'search-budget.json'
        previous = json.loads(budget_path.read_text()) if budget_path.exists() else {}
        if previous and previous['fingerprint'] != fingerprint:
            raise RuntimeError('Search configuration changed since prior budget checkpoint.')
        consumed_before = float(previous.get('consumed_seconds', 0))
        budget_seconds = float(config.get('search_budget_seconds', 14400))
        started = time.monotonic()
        stop_requested = False
        def stop(*_):
            nonlocal stop_requested
            stop_requested = True
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        def save_budget():
            atomic_json(budget_path, {'fingerprint':fingerprint, 'budget_seconds':budget_seconds,
                'consumed_seconds':consumed_before + time.monotonic() - started})
        last_budget_save=[0.0]
        def heartbeat():
            now=time.monotonic()
            if now-last_budget_save[0]>=30:
                save_budget();last_budget_save[0]=now
        budget_tick[0]=heartbeat
        tracking = wandb.init(**config['wandb'], resume='allow', dir=str(root),
            config={**config, 'input_fingerprint':manifest['fingerprint']},
            settings=wandb.Settings(disable_git=True))
        original_optimize = optuna.Study.optimize
        active_study = [None]
        def on_trial(study, trial):
            record = {'heretic/trial':trial.number, 'heretic/state':trial.state.name,
                'heretic/gpu_duty_cycle_target':read_duty_cycle(Path(config['training_run']))}
            for score in trial.user_attrs.get('scores', []):
                value = score.get('score', {}).get('value')
                if isinstance(value, (int, float)):
                    record['heretic/' + score['name']] = value
            tracking.log(record)
            save_budget()
        def optimize(self, func, *args, **kwargs):
            active_study[0] = self
            kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), on_trial]
            remaining = budget_seconds - consumed_before - (time.monotonic() - started)
            if remaining <= 0:
                if can_finalize_exhausted_study(self.trials):
                    # A prior process may have exhausted the timer and died
                    # before exporting. Let upstream select and save from the
                    # persisted completed trials without launching another.
                    save_budget()
                    return None
                raise RuntimeError('Search budget exhausted without a completed trial to export.')
            def exhausted(*_):
                # Upstream catches this, prunes the in-flight trial, and exports
                # the best completed candidate. Idle downtime is never charged.
                raise KeyboardInterrupt
            previous_handler = signal.signal(signal.SIGALRM, exhausted)
            signal.setitimer(signal.ITIMER_REAL, remaining)
            try:
                return original_optimize(self, func, *args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
                save_budget()
        optuna.Study.optimize = optimize
        monitors = SearchMonitors(config, root)
        try:
            monitors.start()
            tracking.summary['phase'] = 'heretic'
            heretic.main.main()
            if stop_requested:
                raise RuntimeError('Heretic was paused; completed trials remain resumable.')
            adapter = root / 'heretic-adapter/adapter_model.safetensors'
            if not adapter.exists():
                raise RuntimeError('Heretic finished without saving the configured adapter.')
            if active_study[0] is None:
                raise RuntimeError('Heretic finished without exposing its study for provenance.')
            monitors.stop()
            shared_gpu = summarize_search(root, config['sharing_policy'])
            if not shared_gpu['passed']:
                raise RuntimeError('Heretic search did not meet the shared-GPU/Humandescent policy.')
            trials = trial_summary(active_study[0])
            save_budget()
            result = {'phase':'complete', 'input_model':str(input_model), 'adapter':str(adapter.parent),
                'adapter_sha256':digest(adapter), 'fingerprint':fingerprint, 'wandb_url':tracking.url,
                'search_budget':json.loads(budget_path.read_text()), 'variant':config.get('variant', 'upstream'),
                'trials':trials, 'shared_gpu':shared_gpu}
            atomic_json(root / 'result.json', result)
            tracking.log({'search/completed_trials':trials['completed_trials'],
                'search/total_trials':trials['total_trials'],
                'search/consumed_seconds':result['search_budget']['consumed_seconds'],
                'search/gpu_energy_joules_shared':shared_gpu['gpu_energy_joules_shared'],
                'search/mean_gpu_utilization_percent':shared_gpu['mean_gpu_utilization_percent'],
                'search/minimum_free_gpu_mib':shared_gpu['minimum_free_gpu_mib'],
                'search/humandescent_p95_seconds':shared_gpu['humandescent_p95_seconds']})
            tracking.summary.update(result)
        finally:
            try:
                monitors.stop()
            finally:
                save_budget()
                if sys.exc_info()[0]:
                    tracking.summary['phase'] = 'failed'
                tracking.finish(exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == '__main__':
    main()
