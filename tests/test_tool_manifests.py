from __future__ import annotations

from pathlib import Path

import yaml

from llm_lab.tooling.builtins import BuiltinToolSettings, create_builtin_registry
from llm_lab.tooling.orchestrator import AgentLimits


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAMES = {
    "web_search": "searxng-search.yaml",
    "web_fetch": "web-fetch.yaml",
    "calculator": "calculator.yaml",
    "current_time": "current-time.yaml",
}


def _yaml(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_reviewed_builtin_schemas_match_runtime_registration() -> None:
    registry = create_builtin_registry(BuiltinToolSettings())

    for definition in registry.resolve("standard-readonly"):
        manifest = _yaml(ROOT / "catalog" / "tools" / MANIFEST_NAMES[definition.name])
        assert manifest["id"] == definition.name
        assert manifest["input_schema"] == definition.parameters
        assert manifest["output_schema"] == definition.output_schema


def test_reviewed_toolset_limits_match_runtime_defaults() -> None:
    manifest = _yaml(
        ROOT / "catalog" / "toolsets" / "standard-readonly.yaml"
    )
    policy = manifest["policy"]
    limits = AgentLimits()
    settings = BuiltinToolSettings()

    assert policy["maximum_calls_per_round"] == limits.max_tool_calls_per_round
    assert policy["tool_calls_execute_serially"] is True
    assert policy["maximum_rounds"] == limits.max_rounds
    assert policy["maximum_concurrent_turns"] == limits.max_concurrent_turns
    assert policy["maximum_queued_turns"] == limits.max_queued_turns
    assert policy["tool_call_deadline_seconds"] == limits.tool_timeout_seconds
    assert policy["model_round_deadline_seconds"] == limits.model_timeout_seconds
    assert policy["total_turn_deadline_seconds"] == limits.total_timeout_seconds
    assert (
        policy["maximum_reserved_generation_tokens"]
        == limits.max_cumulative_generation_tokens
    )
    assert (
        policy["maximum_model_round_response_bytes"]
        == limits.max_model_response_bytes
    )
    assert policy["maximum_assistant_characters"] == limits.max_assistant_characters

    for filename in ("searxng-search.yaml", "web-fetch.yaml"):
        adapter = _yaml(ROOT / "catalog" / "tools" / filename)["adapter"]
        assert adapter["request_timeout_seconds"] == settings.request_timeout_seconds
        assert adapter["execution_deadline_seconds"] == limits.tool_timeout_seconds
        assert adapter["maximum_response_bytes"] == 1024 * 1024
        assert adapter["maximum_serialized_result_bytes"] == 60 * 1024
