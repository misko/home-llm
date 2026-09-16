"""HTTP adapter for bounded model-managed read-only tool turns."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .runtime import RuntimeState
from .tooling.builtins import create_builtin_registry
from .tooling.errors import ToolPolicyError, ToolingError
from .tooling.orchestrator import AgentRunner, OpenAIChatBackend
from .tooling.registry import ToolRegistry
from .tooling.schema import AgentTurnRequest, ErrorEvent, ToolErrorDTO, ToolsetsResponse

if TYPE_CHECKING:
    from fastapi.responses import Response


LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_AGENT_REQUEST_BYTES = 64 * 1024 * 1024
ActiveStateResolver = Callable[[], Awaitable[RuntimeState | JSONResponse]]
ClientGetter = Callable[[], httpx.AsyncClient]


class _AgentBodyTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=413, detail="agent request body too large")


class AgentRequestBodyLimitMiddleware:
    """Admit bounded turns before parsing and cap their actual ASGI bodies."""

    def __init__(
        self,
        app: Any,
        *,
        maximum_bytes: int,
        maximum_inflight_requests: int,
    ) -> None:
        if maximum_inflight_requests < 1:
            raise ValueError("agent request admission capacity must be positive")
        self.app = app
        self.maximum_bytes = maximum_bytes
        self.maximum_inflight_requests = maximum_inflight_requests
        self._admission_guard = threading.Lock()
        self._inflight_requests = 0

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/api/v1/agent/turns"
        ):
            await self.app(scope, receive, send)
            return
        with self._admission_guard:
            admitted = self._inflight_requests < self.maximum_inflight_requests
            if admitted:
                self._inflight_requests += 1
        if not admitted:
            response = _agent_error(
                429,
                "agent_busy",
                "The agent has reached its bounded request capacity",
                retryable=True,
            )
            response.headers["Retry-After"] = "1"
            await response(scope, receive, send)
            return

        try:
            headers = {
                key.lower(): value
                for key, value in scope.get("headers", ())
            }
            declared = headers.get(b"content-length")
            if declared is not None:
                try:
                    if int(declared) > self.maximum_bytes:
                        await _agent_error(
                            413,
                            "agent_request_too_large",
                            "Agent request body exceeds the configured limit",
                        )(scope, receive, send)
                        return
                except ValueError:
                    pass
            received = 0

            async def limited_receive() -> dict[str, Any]:
                nonlocal received
                message = await receive()
                if message.get("type") == "http.request":
                    received += len(message.get("body", b""))
                    if received > self.maximum_bytes:
                        raise _AgentBodyTooLarge()
                return message

            try:
                await self.app(scope, limited_receive, send)
            except _AgentBodyTooLarge:
                await _agent_error(
                    413,
                    "agent_request_too_large",
                    "Agent request body exceeds the configured limit",
                )(scope, receive, send)
        finally:
            with self._admission_guard:
                self._inflight_requests -= 1


def _agent_error(
    status: int,
    code: str,
    message: str,
    *,
    retryable: bool = False,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            }
        },
    )


def _sse(event: object) -> str:
    assert hasattr(event, "model_dump")
    document = event.model_dump(mode="json")  # type: ignore[attr-defined]
    return (
        f"id: {document['sequence']}\n"
        f"event: {document['type']}\n"
        "data: "
        + json.dumps(document, ensure_ascii=True, separators=(",", ":"))
        + "\n\n"
    )


def install_agent_api(
    app: FastAPI,
    *,
    active_state: ActiveStateResolver,
    client_getter: ClientGetter,
    registry: ToolRegistry | None = None,
    runner: AgentRunner | None = None,
    maximum_request_bytes: int = DEFAULT_MAX_AGENT_REQUEST_BYTES,
    body_limit_installed: bool = False,
) -> AgentRunner:
    """Install the agent routes without changing the raw ``/v1`` API."""

    if not 1024 <= maximum_request_bytes <= 64 * 1024 * 1024:
        raise ValueError("agent request body limit must be between 1 KiB and 64 MiB")
    if runner is not None and registry is not None and runner.registry is not registry:
        raise ValueError("runner and registry must refer to the same tool registry")
    selected_registry = registry or (runner.registry if runner else create_builtin_registry())
    selected_runner = runner or AgentRunner(
        selected_registry, OpenAIChatBackend(client_getter)
    )
    router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

    prior_validation_handler = app.exception_handlers.get(RequestValidationError)

    @app.exception_handler(RequestValidationError)
    async def sanitized_agent_validation(request: Request, exc: RequestValidationError):
        if request.url.path == "/api/v1/agent/turns":
            return _agent_error(
                422,
                "invalid_agent_request",
                "Agent request did not match the reviewed input contract",
            )
        if prior_validation_handler is not None:
            return await prior_validation_handler(request, exc)
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(_AgentBodyTooLarge)
    async def sanitized_agent_body_limit(
        request: Request, exc: _AgentBodyTooLarge
    ) -> JSONResponse:
        del request, exc
        return _agent_error(
            413,
            "agent_request_too_large",
            "Agent request body exceeds the configured limit",
        )

    @router.get("/toolsets", response_model=ToolsetsResponse)
    async def toolsets() -> ToolsetsResponse:
        return selected_registry.describe()

    @router.post("/turns", response_model=None)
    async def turns(request: AgentTurnRequest) -> "Response":
        try:
            selected_registry.resolve(request.toolset)
        except ToolPolicyError as exc:
            return _agent_error(
                404,
                exc.code,
                exc.message,
                retryable=exc.retryable,
            )
        state_or_error = await active_state()
        if isinstance(state_or_error, JSONResponse):
            return state_or_error
        state = state_or_error

        async def stream() -> AsyncIterator[str]:
            run_id = f"run_{uuid.uuid4().hex}"
            sequence = 1
            try:
                async with asyncio.timeout(
                    selected_runner.limits.total_timeout_seconds
                ):
                    async for event in selected_runner.run(
                        request,
                        model=state.public_alias,
                        deployment=state.deployment_id,
                        base_url=state.base_url,
                        context_size=state.deployment.context_size,
                    ):
                        run_id = event.run_id
                        sequence = event.sequence + 1
                        yield _sse(event)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                yield _sse(
                    ErrorEvent(
                        run_id=run_id,
                        sequence=sequence,
                        error=ToolErrorDTO(
                            code="agent_timeout",
                            message="The agent turn exceeded its total deadline",
                            retryable=True,
                        ),
                    )
                )
            except ToolingError as exc:
                yield _sse(
                    ErrorEvent(
                        run_id=run_id,
                        sequence=sequence,
                        error=ToolErrorDTO(
                            code=exc.code,
                            message=exc.message[:1024],
                            retryable=exc.retryable,
                        ),
                    )
                )
            except Exception:
                LOGGER.exception("Unhandled agent turn failure")
                yield _sse(
                    ErrorEvent(
                        run_id=run_id,
                        sequence=sequence,
                        error=ToolErrorDTO(
                            code="agent_failed",
                            message="The agent turn failed",
                            retryable=False,
                        ),
                    )
                )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    app.include_router(router)
    if not body_limit_installed:
        app.add_middleware(
            AgentRequestBodyLimitMiddleware,
            maximum_bytes=maximum_request_bytes,
            maximum_inflight_requests=(
                selected_runner.limits.max_concurrent_turns
                + selected_runner.limits.max_queued_turns
            ),
        )
    app.state.agent_registry = selected_registry
    app.state.owns_agent_registry = runner is None and registry is None
    app.state.agent_runner = selected_runner
    return selected_runner


__all__ = [
    "AgentRequestBodyLimitMiddleware",
    "DEFAULT_MAX_AGENT_REQUEST_BYTES",
    "install_agent_api",
]
