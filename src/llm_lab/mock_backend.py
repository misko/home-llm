"""Deterministic OpenAI-compatible backend used by smoke and lifecycle tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


JsonObject = dict[str, Any]


def _canonical_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return "" if value is None else str(value)
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, Mapping):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif item.get("type") in {"input_text", "output_text", "text"}:
                parts.append(str(item.get("content", "")))
    return " ".join(part for part in parts if part)


def _chat_prompt(payload: Mapping[str, Any]) -> str:
    messages = payload.get("messages", ())
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return ""
    fallback = ""
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = _text_content(message.get("content"))
        if content:
            fallback = content
        if message.get("role") == "user" and content:
            fallback = content
    return fallback


def _responses_prompt(payload: Mapping[str, Any]) -> str:
    value = payload.get("input", "")
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return _text_content(value)
    fallback = ""
    for item in value:
        if isinstance(item, Mapping):
            content = _text_content(item.get("content"))
            if content:
                fallback = content
        else:
            content = _text_content(item)
            if content:
                fallback = content
    return fallback


def deterministic_text(prompt: str) -> str:
    """Stable mock generation shared by all endpoint shapes."""

    normalized = " ".join(prompt.split())
    return f"mock response: {normalized}" if normalized else "mock response"


def _schema_value(schema: Mapping[str, Any]) -> Any:
    if "const" in schema:
        return schema["const"]
    if "default" in schema:
        return schema["default"]
    value_type = schema.get("type")
    if value_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", properties.keys())
        if isinstance(properties, Mapping) and isinstance(required, Sequence):
            return {
                str(key): _schema_value(properties[key])
                for key in required
                if key in properties and isinstance(properties[key], Mapping)
            }
        return {}
    if value_type == "array":
        return []
    if value_type == "integer":
        return 2
    if value_type == "number":
        return 2.0
    if value_type == "boolean":
        return True
    return "ok"


def _structured_result(payload: Mapping[str, Any]) -> str | None:
    response_format = payload.get("response_format")
    if not isinstance(response_format, Mapping):
        return None
    if response_format.get("type") not in {"json_schema", "json_object"}:
        return None
    schema_wrapper = response_format.get("json_schema", response_format.get("schema"))
    if isinstance(schema_wrapper, Mapping) and isinstance(
        schema_wrapper.get("schema"), Mapping
    ):
        schema_wrapper = schema_wrapper["schema"]
    if not isinstance(schema_wrapper, Mapping):
        value: Any = {"status": "ok", "count": 2}
    else:
        value = _schema_value(schema_wrapper)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _tool_argument_value(name: str, schema: Mapping[str, Any], prompt: str) -> Any:
    lowered = name.lower()
    if lowered in {"city", "location", "place"}:
        match = re.search(
            r"\bin\s+([A-Za-z][A-Za-z .'-]*?)(?:[?.!,]|\s+use\b|$)",
            prompt,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip().title()
        return "Seattle"
    if "const" in schema:
        return schema["const"]
    if "default" in schema:
        return schema["default"]
    value_type = schema.get("type")
    if value_type == "integer":
        return 2
    if value_type == "number":
        return 2.0
    if value_type == "boolean":
        return True
    if value_type == "array":
        return []
    if value_type == "object":
        return _schema_value(schema)
    return "mock"


def _forced_tool_call(
    payload: Mapping[str, Any],
    prompt: str,
    request_id: str,
) -> JsonObject | None:
    tool_choice = payload.get("tool_choice")
    if not isinstance(tool_choice, Mapping):
        return None
    function_choice = tool_choice.get("function")
    if not isinstance(function_choice, Mapping) or not function_choice.get("name"):
        return None
    name = str(function_choice["name"])
    parameters: Mapping[str, Any] = {}
    tools = payload.get("tools", ())
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            function = tool.get("function")
            if not isinstance(function, Mapping) or function.get("name") != name:
                continue
            candidate = function.get("parameters")
            if isinstance(candidate, Mapping):
                parameters = candidate
            break
    properties = parameters.get("properties", {})
    required = parameters.get("required", ())
    arguments: dict[str, Any] = {}
    if isinstance(properties, Mapping) and isinstance(required, Sequence):
        for key in required:
            schema = properties.get(key, {})
            if isinstance(schema, Mapping):
                arguments[str(key)] = _tool_argument_value(
                    str(key), schema, prompt
                )
    return {
        "id": f"call_{request_id.removeprefix('chatcmpl_')}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    }


def _chat_result(payload: Mapping[str, Any], prompt: str) -> str:
    structured = _structured_result(payload)
    if structured is not None:
        return structured
    if "planet humans live on" in prompt.lower():
        return "Earth"
    return deterministic_text(prompt)


def _usage(prompt: str, result: str) -> JsonObject:
    prompt_tokens = len(prompt.split())
    completion_tokens = len(result.split())
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }


def _sse(value: Mapping[str, Any], *, event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return f"{prefix}data: {data}\n\n".encode()


async def _chat_stream(
    request_id: str,
    model: str,
    result: str,
    tool_call: JsonObject | None = None,
) -> AsyncIterator[bytes]:
    yield _sse(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
    )
    if tool_call is None:
        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": result},
                        "finish_reason": None,
                    }
                ],
            }
        )
    else:
        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, **tool_call}]},
                        "finish_reason": None,
                    }
                ],
            }
        )
    yield _sse(
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool_call else "stop",
                }
            ],
        }
    )
    yield b"data: [DONE]\n\n"


async def _completion_stream(
    request_id: str,
    model: str,
    result: str,
) -> AsyncIterator[bytes]:
    yield _sse(
        {
            "id": request_id,
            "object": "text_completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": result,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        }
    )
    yield b"data: [DONE]\n\n"


def _response_object(
    request_id: str,
    model: str,
    prompt: str,
    result: str,
) -> JsonObject:
    return {
        "id": request_id,
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": model,
        "output_text": result,
        "output": [
            {
                "id": f"msg_{request_id.removeprefix('resp_')}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result,
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": len(prompt.split()),
            "output_tokens": len(result.split()),
            "total_tokens": len(prompt.split()) + len(result.split()),
        },
    }


async def _responses_stream(response: JsonObject, result: str) -> AsyncIterator[bytes]:
    created = dict(response)
    created["status"] = "in_progress"
    created["output"] = []
    created.pop("output_text", None)
    yield _sse(
        {"type": "response.created", "response": created},
        event="response.created",
    )
    yield _sse(
        {
            "type": "response.output_text.delta",
            "item_id": response["output"][0]["id"],
            "output_index": 0,
            "content_index": 0,
            "delta": result,
        },
        event="response.output_text.delta",
    )
    yield _sse(
        {"type": "response.completed", "response": response},
        event="response.completed",
    )
    yield b"data: [DONE]\n\n"


async def _json_body(request: Request) -> JsonObject:
    try:
        value = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        value = None
    if not isinstance(value, dict):
        return {}
    return value


def create_mock_app(model: str = "mock-model") -> FastAPI:
    """Create the deterministic backend without binding a network socket."""

    app = FastAPI(title="LLM Lab deterministic mock backend")
    app.state.model = model

    @app.get("/health")
    async def health() -> JsonObject:
        return {"status": "ok", "model": app.state.model}

    @app.get("/v1/models")
    async def models() -> JsonObject:
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "llm-lab-mock",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await _json_body(request)
        prompt = _chat_prompt(payload)
        model_name = str(payload.get("model") or app.state.model)
        request_id = _canonical_id("chatcmpl", payload)
        result = _chat_result(payload, prompt)
        tool_call = _forced_tool_call(payload, prompt, request_id)
        if payload.get("stream") is True:
            return StreamingResponse(
                _chat_stream(request_id, model_name, result, tool_call),
                media_type="text/event-stream",
                headers={"X-LLM-Lab-Mock": "true"},
            )
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": 0,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None if tool_call else result,
                            **(
                                {"tool_calls": [tool_call]}
                                if tool_call is not None
                                else {}
                            ),
                        },
                        "logprobs": None,
                        "finish_reason": "tool_calls" if tool_call else "stop",
                    }
                ],
                "usage": _usage(prompt, result),
            },
            headers={"X-LLM-Lab-Mock": "true"},
        )

    @app.post("/v1/completions")
    async def completions(request: Request):
        payload = await _json_body(request)
        prompt_value = payload.get("prompt", "")
        if isinstance(prompt_value, Sequence) and not isinstance(prompt_value, str):
            prompt = " ".join(str(value) for value in prompt_value)
        else:
            prompt = str(prompt_value)
        result = deterministic_text(prompt)
        model_name = str(payload.get("model") or app.state.model)
        request_id = _canonical_id("cmpl", payload)
        if payload.get("stream") is True:
            return StreamingResponse(
                _completion_stream(request_id, model_name, result),
                media_type="text/event-stream",
                headers={"X-LLM-Lab-Mock": "true"},
            )
        return JSONResponse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": 0,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": result,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(prompt, result),
            },
            headers={"X-LLM-Lab-Mock": "true"},
        )

    @app.post("/v1/responses")
    async def responses(request: Request):
        payload = await _json_body(request)
        prompt = _responses_prompt(payload)
        result = deterministic_text(prompt)
        model_name = str(payload.get("model") or app.state.model)
        request_id = _canonical_id("resp", payload)
        response = _response_object(request_id, model_name, prompt, result)
        if payload.get("stream") is True:
            return StreamingResponse(
                _responses_stream(response, result),
                media_type="text/event-stream",
                headers={"X-LLM-Lab-Mock": "true"},
            )
        return JSONResponse(response, headers={"X-LLM-Lab-Mock": "true"})

    return app


app = create_mock_app(os.environ.get("LLM_LAB_MOCK_MODEL", "mock-model"))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18089)
    parser.add_argument("--model", default="mock-model")
    arguments = parser.parse_args(argv)
    uvicorn.run(
        create_mock_app(arguments.model),
        host=arguments.host,
        port=arguments.port,
        log_level="warning",
    )


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    main()
