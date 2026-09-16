"""Cooperative process scheduling for a dedicated llama.cpp server.

The watchdog resumes the server if its controlling process exits. It only signals
the specified PID while its creation time matches, never other GPU processes.
Already queued kernels can finish during pauses; this is not an SM or VRAM cap.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

import psutil


def same_process(pid: int, created: float) -> bool:
    try:
        return psutil.Process(pid).create_time() == created and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def watch(pid: int, created: float, parent: int, parent_created: float, duty: float, period: float) -> None:
    if not 0.1 <= duty <= 1 or not 0.02 <= period <= 1:
        raise ValueError("Invalid duty cycle or scheduling period")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping and same_process(pid, created) and same_process(parent, parent_created):
            os.kill(pid, signal.SIGCONT)
            time.sleep(period * duty)
            if duty < 1 and not stopping and same_process(pid, created):
                os.kill(pid, signal.SIGSTOP)
                time.sleep(period * (1 - duty))
    except ProcessLookupError:
        pass
    finally:
        if same_process(pid, created):
            os.kill(pid, signal.SIGCONT)


@contextmanager
def process_duty_cycle(pid: int, duty: float = 0.85, period: float = 0.2, *, allow_target_exit: bool = False):
    if not 0.1 <= duty <= 1 or not 0.02 <= period <= 1:
        raise ValueError("Invalid duty cycle or scheduling period")
    created = psutil.Process(pid).create_time()
    parent = psutil.Process()
    command = [sys.executable, "-m", __name__, str(pid), str(created), str(parent.pid), str(parent.create_time()), str(duty), str(period)]
    watchdog = subprocess.Popen(command)
    try:
        time.sleep(min(period, 0.05))
        if watchdog.poll() is not None:
            raise RuntimeError("Inference-sharing watchdog failed to start")
        yield watchdog
        if watchdog.poll() is not None and not (allow_target_exit and not same_process(pid, created)):
            raise RuntimeError("Inference-sharing watchdog exited during evaluation")
    finally:
        watchdog.terminate()
        try:
            watchdog.wait(timeout=3)
        except subprocess.TimeoutExpired:
            watchdog.kill()
            watchdog.wait()
        if same_process(pid, created):
            os.kill(pid, signal.SIGCONT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("created", type=float)
    parser.add_argument("parent", type=int)
    parser.add_argument("parent_created", type=float)
    parser.add_argument("duty", type=float)
    parser.add_argument("period", type=float)
    watch(**vars(parser.parse_args()))
