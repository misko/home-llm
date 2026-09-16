"""Execute pinned expanded HumanEval tests exclusively inside a CPU container."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from .contracts import Outcome, atomic_json, fingerprint
from .runner import latest_outcomes


def extract_source(response: str, entry_point: str) -> str:
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", response, flags=re.S)
    source = fences[0] if len(fences) == 1 else response
    tree = ast.parse(source)
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point for node in tree.body):
        raise ValueError("Response does not define the required function")
    return source


def container_command(image: str, name: str) -> list[str]:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Code evaluator image must be an immutable local image ID")
    return ["docker", "run", "--rm", "-i", "--name", name, "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64",
            "--memory=1g", "--memory-swap=1g", "--cpus=1", "--ulimit", "nofile=64:64",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", image]


def evaluate(source: str, test: str, entry_point: str, image: str, *, timeout: float = 150, problem: dict | None = None) -> dict:
    name = "llm-lab-code-" + uuid.uuid4().hex
    # The reference tests execute outside the candidate's global namespace.
    # A random completion marker prevents a normal premature exit being a pass.
    marker = uuid.uuid4().hex
    script = "import json\nnamespace = {}\nexec(" + repr(source) + ", namespace)\n" + test + "\ncheck(namespace[" + repr(entry_point) + "])\nprint(" + repr(marker) + ")\n"
    if problem is not None:
        script = "import sys, json\nsys.path.insert(0, '/opt/vendor')\nfrom evalplus.eval import untrusted_check, PASS\nfrom evalplus.gen.util import trusted_exec\n"
        script += 'problem = json.loads(' + repr(json.dumps(problem)) + ')\ncode = ' + repr(source) + '\n'
        script += "results = []\nfor split in ['base_input', 'plus_input']:\n    expected, times = trusted_exec(problem['prompt'] + problem['canonical_solution'], problem[split], problem['entry_point'], record_time=True)\n    status, details = untrusted_check('humaneval', code, problem[split], problem['entry_point'], expected, problem['atol'], times, fast_check=True)\n    results.append(status == PASS)\nif all(results):\n    print(" + repr(marker) + ")\nelse:\n    raise SystemExit(1)\n"
    with tempfile.TemporaryFile(mode='w+b') as input_file, tempfile.TemporaryFile(mode='w+b') as stdout, tempfile.TemporaryFile(mode='w+b') as stderr:
        input_file.write(script.encode()); input_file.seek(0)
        process = subprocess.Popen(container_command(image, name), stdin=input_file, stdout=stdout, stderr=stderr)
        started = time.monotonic()
        reason = None
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    reason = 'timeout'
                    break
                if stdout.tell() + stderr.tell() > 2_000_000:
                    reason = 'output_limit'
                    break
                time.sleep(.1)
        finally:
            # Terminating Docker CLI alone does not stop the container.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        stdout.seek(0); stderr.seek(0)
        out = stdout.read(2_000_000).decode(errors='replace')
        err = stderr.read(2_000_000).decode(errors='replace')
        if reason:
            return {"passed":False, "reason":reason, "timeout_seconds":timeout}
        if process.returncode in (125,126,127):
            raise RuntimeError("Code evaluator infrastructure failed: " + err[-1500:])
        return {"passed": process.returncode == 0 and marker in out.splitlines(),
                "reason": "completed" if process.returncode == 0 else "test_failure",
                "returncode": process.returncode, "stderr_tail": err[-2000:]}


def score_saved(root: Path, arm: str, partition: str, image: str) -> dict:
    from .contracts import file_hash
    lock = json.loads((root / "dataset-lock.json").read_text())
    path = root / 'sources' / lock['sources']['humaneval-plus']['file']
    if file_hash(path) != lock["sources"]["humaneval-plus"]["sha256"]:
        raise ValueError("Expanded coding tests no longer match the frozen source")
    tasks = {row["task_id"]:row for row in map(json.loads, path.read_text().splitlines())}
    run_root = root / "arms" / arm / partition
    outcomes = latest_outcomes(run_root)
    results = {}
    for case_id, row in outcomes.items():
        if row.status != "pending_judgment" or row.details.get("required_evaluator") != "humaneval_plus":
            continue
        task = tasks[case_id.removeprefix("humaneval-plus/")]
        try:
            source = extract_source(row.response, task["entry_point"])
        except (SyntaxError, ValueError) as exc:
            result = {"passed":False, "reason":f"invalid_source: {exc}"}
        else:
            result = evaluate(source, task["test"], task["entry_point"], image, problem=task)
        judgment = {"kind":"code", "outcome_sha256": fingerprint(row.model_dump()), "image": image,
                    "source_sha256":lock["sources"]["humaneval-plus"]["sha256"], "result":result}
        destination = run_root / "judgments" / (fingerprint(case_id) + ".json")
        if destination.exists() and json.loads(destination.read_text()) != judgment:
            raise ValueError("Refusing to replace a different code judgment")
        atomic_json(destination, judgment)
        results[case_id] = result
    return results
