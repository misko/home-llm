"""Continuous shared-GPU and Humandescent monitoring for Heretic searches."""
from __future__ import annotations

import json
import statistics
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

from llm_lab.comparison.contracts import atomic_json
from llm_lab.comparison.humandescent import probe, read_samples
from llm_lab.telemetry import NvidiaTelemetrySampler


class SearchMonitors:
    def __init__(self, config: dict, root: Path, *, interval_seconds: float = 1):
        self.config = config
        self.root = root
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = None
        self.probe_context = None
        self.telemetry_path = None

    def start(self):
        if self.thread is not None:
            return self
        index = max(len(list(self.root.glob("telemetry-search-*.json"))),
                    len(list(self.root.glob("humandescent-search-*.jsonl")))) + 1
        self.telemetry_path = self.root / f"telemetry-search-{index:04d}.json"
        sampler = NvidiaTelemetrySampler(interval_seconds=self.interval_seconds)

        def collect():
            while not self.stop_event.is_set():
                self.samples.extend(sample.to_dict() for sample in sampler.sample_once())
                self.stop_event.wait(self.interval_seconds)

        self.probe_context = probe(
            self.config, self.root / f"humandescent-search-{index:04d}.jsonl", interval_seconds=30
        )
        self.probe_context.__enter__()
        self.thread = threading.Thread(target=collect, name="heretic-gpu-telemetry", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.thread is None:
            return
        probe_error = None
        try:
            if self.probe_context is not None:
                self.probe_context.__exit__(None, None, None)
        except BaseException as exc:
            probe_error = exc
        finally:
            self.stop_event.set()
            self.thread.join(timeout=max(10, self.interval_seconds * 3))
            atomic_json(self.telemetry_path, self.samples)
            self.thread = None
            self.probe_context = None
        if probe_error is not None:
            raise probe_error


def summarize_search(root: Path, policy: dict) -> dict:
    energy = 0.0
    observed = 0.0
    utilizations = []
    free_memory = []
    available_samples = 0
    for path in sorted(root.glob("telemetry-search-*.json")):
        samples = [sample for sample in json.loads(path.read_text())
                   if sample.get("available") and sample.get("index") == 0]
        available_samples += len(samples)
        for sample in samples:
            if sample.get("gpu_utilization_percent") is not None:
                utilizations.append(sample["gpu_utilization_percent"])
            if sample.get("memory_total_mib") is not None and sample.get("memory_used_mib") is not None:
                free_memory.append(sample["memory_total_mib"] - sample["memory_used_mib"])
        for before, after in zip(samples, samples[1:]):
            seconds = (datetime.fromisoformat(after["timestamp"].replace("Z", "+00:00")) -
                       datetime.fromisoformat(before["timestamp"].replace("Z", "+00:00"))).total_seconds()
            powers = [before.get("power_draw_w"), after.get("power_draw_w")]
            if 0 < seconds <= 5 and all(value is not None for value in powers):
                energy += seconds * sum(powers) / 2
                observed += seconds
    hum = sorted(record["latency_seconds"] for record in read_samples(root)
                 if "latency_seconds" in record)
    p95 = hum[min(len(hum) - 1, int(len(hum) * .95))] if hum else None
    mean_utilization = statistics.mean(utilizations) if utilizations else None
    minimum_free = min(free_memory) if free_memory else None
    passed = (
        len(hum) >= policy["minimum_humandescent_samples"]
        and p95 is not None and p95 <= policy["max_humandescent_p95_seconds"]
        and mean_utilization is not None and mean_utilization <= policy["max_mean_gpu_utilization_percent"]
        and minimum_free is not None and minimum_free >= policy["minimum_free_gpu_mib"]
    )
    return {
        "passed": passed,
        "policy": policy,
        "telemetry_samples": available_samples,
        "gpu_energy_joules_shared": energy if observed else None,
        "gpu_energy_observed_seconds": observed,
        "mean_gpu_utilization_percent": mean_utilization,
        "minimum_free_gpu_mib": minimum_free,
        "humandescent_samples": len(hum),
        "humandescent_p50_seconds": statistics.median(hum) if hum else None,
        "humandescent_p95_seconds": p95,
        "note": "System-level shared-GPU measurements; energy is not isolated to Heretic alone.",
    }


def trial_summary(study) -> dict:
    states = Counter(trial.state.name.lower() for trial in study.trials)
    selected = sorted(study.best_trials, key=lambda trial: trial.number)
    return {
        "total_trials": len(study.trials),
        "trial_states": dict(sorted(states.items())),
        "completed_trials": states.get("complete", 0),
        "selected_trial_numbers": [trial.number for trial in selected],
        "selected_objective_values": [list(trial.values or ()) for trial in selected],
    }
