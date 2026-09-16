"""Cooperative GPU time sharing; this does not partition GPU memory or SMs."""
import json
import math
import time


def read_duty_cycle(run):
    path = run / 'gpu-duty-cycle.json'
    value = json.loads(path.read_text())['duty_cycle'] if path.exists() else 1.0
    value = float(value)
    if not math.isfinite(value) or not 0.1 <= value <= 1.0:
        raise ValueError('GPU duty cycle must be between 0.1 and 1.0')
    return value


def yield_gpu(active_seconds, duty_cycle):
    """Call only after CUDA synchronization so the pause leaves no queued work."""
    delay = max(0.0, active_seconds) * (1.0 / duty_cycle - 1.0)
    if delay > 0:
        time.sleep(delay)
    return delay
