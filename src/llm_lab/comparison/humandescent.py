"""Bounded lifecycle for the independent Humandescent responsiveness probe."""
from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def probe(config: dict, output: Path, *, interval_seconds: float=30):
    settings=config['humandescent']
    command=[settings['python'],str(Path(config.get('code_root',config['repo_root']))/'benchmarks/qwen-comparison/humandescent-probe.py'),
             '--hudes-root',settings['root'],'--url',settings['url'],'--samples','0','--interval',str(interval_seconds)]
    with output.open('w') as stream, output.with_suffix('.log').open('w') as errors:
        child=subprocess.Popen(command,stdout=stream,stderr=errors)
        try:
            yield child
            if child.poll() is not None:
                raise RuntimeError('Humandescent task probe failed; inspect '+str(output.with_suffix('.log')))
        finally:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()


def read_samples(root: Path) -> list[dict]:
    samples=[]
    for path in sorted(root.glob('humandescent-*.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():samples.append(json.loads(line))
    return samples
