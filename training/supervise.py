"""Systemd entry point: resume training, then export/register the candidate."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from common import atomic_json


def main():
    config_path=Path(sys.argv[1]);config=json.loads(config_path.read_text())
    run=Path(config['run_dir']);source=Path(__file__).parent
    child=[None];stopping=[False]
    def stop(*_):
        stopping[0]=True
        if child[0] is not None:
            child[0].terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    commands=[
        [config['training_python'],str(source/'worker.py'),str(config_path)],
        [config['control_python'],str(source/'export.py'),str(config_path)],
        [config['training_python'],str(source/'report.py'),str(config_path)],
    ]
    for command in commands:
        if stopping[0]:return
        child[0]=subprocess.Popen(command)
        code=child[0].wait();child[0]=None
        if stopping[0]:return
        if code:
            raise SystemExit(code)
        if command[1].endswith('worker.py') and not (run/'trained.json').exists():
            return


if __name__=='__main__':
    main()
