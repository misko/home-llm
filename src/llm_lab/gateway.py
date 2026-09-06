"""Small OpenAI-compatible gateway for the currently active deployment."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .attestation import configured_gateway_origin, create_gateway_attestation
from .errors import DeploymentError, IntegrityError
from .paths import LabPaths
from .runtime import RuntimeManager, RuntimeState, read_active_state


_REQUEST_HEADER_DENYLIST = {
    "authorization",
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_RESPONSE_HEADER_DENYLIST = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _error(status_code: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


def _upstream_url(base_url: str, request: Request) -> str:
    """Join a route without duplicating ``/v1`` on external base URLs."""

    parsed = urlsplit(base_url)
    base_path = parsed.path.rstrip("/")
    route = request.url.path
    if base_path.endswith("/v1") and route.startswith("/v1"):
        path = f"{base_path}{route.removeprefix('/v1')}"
    else:
        path = f"{base_path}{route}"
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            request.url.query,
            "",
        )
    )


def _request_headers(headers: Mapping[str, str], state: RuntimeState) -> dict[str, str]:
    forwarded = {
        key: value
        for key, value in headers.items()
        if key.lower() not in _REQUEST_HEADER_DENYLIST
    }
    forwarded["x-llm-lab-deployment"] = state.deployment_id
    return forwarded


def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _RESPONSE_HEADER_DENYLIST
    }


def _requests_stream(body: bytes) -> bool:
    if not body:
        return False
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


def _validate_model_identity(body: bytes, state: RuntimeState) -> JSONResponse | None:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error(400, "Request body must be valid JSON", "invalid_request_error")
    if not isinstance(payload, dict):
        return _error(400, "Request body must be a JSON object", "invalid_request_error")
    requested = payload.get("model")
    if requested != state.public_alias:
        return _error(
            400,
            f"Requested model must be the active alias {state.public_alias!r}",
            "invalid_request_error",
        )
    return None


async def _stream_and_close(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        # httpx decodes content-encoding in ``aiter_bytes``.  The gateway strips
        # that header, so forwarding raw compressed bytes would mislabel them.
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()


def create_app(
    paths: LabPaths | None = None,
    *,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
    advertised_origin: str | None = None,
    runtime_manager: RuntimeManager | None = None,
) -> FastAPI:
    """Create a gateway whose target is resolved from active state per request."""

    resolved_paths = paths or LabPaths.discover()
    configured_key = (
        api_key
        if api_key is not None
        else os.environ.get("LLM_LAB_GATEWAY_API_KEY")
    )
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        upstream_client = application.state.upstream_client
        if application.state.owns_upstream_client and upstream_client is not None:
            await upstream_client.aclose()
            application.state.upstream_client = None

    app = FastAPI(
        title="LLM Lab OpenAI-compatible gateway",
        lifespan=lifespan,
    )
    app.state.paths = resolved_paths
    app.state.api_key = configured_key
    app.state.advertised_origin = advertised_origin or configured_gateway_origin()
    app.state.upstream_client = client
    app.state.owns_upstream_client = client is None
    app.state.runtime_manager = runtime_manager or RuntimeManager(resolved_paths)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        expected = app.state.api_key
        if expected and request.url.path.startswith("/v1/"):
            authorization = request.headers.get("authorization", "")
            scheme, separator, token = authorization.partition(" ")
            valid = (
                bool(separator)
                and scheme.lower() == "bearer"
                and secrets.compare_digest(token, expected)
            )
            if not valid:
                response = _error(
                    401,
                    "Missing or invalid bearer token",
                    "authentication_error",
                )
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
        return await call_next(request)

    async def active_state() -> RuntimeState | JSONResponse:
        try:
            state = read_active_state(app.state.paths)
        except DeploymentError as exc:
            return _error(503, str(exc), "service_unavailable")
        if state is None:
            return _error(
                503,
                "No active model deployment",
                "service_unavailable",
            )
        status = await asyncio.to_thread(
            app.state.runtime_manager.inspect_state,
            state,
            check_health=False,
        )
        if not status.ready:
            return _error(
                503,
                f"Active deployment is not ready (phase={state.phase}, "
                f"running={status.running})",
                "service_unavailable",
            )
        return state

    async def proxy(request: Request):
        state_or_error = await active_state()
        if isinstance(state_or_error, JSONResponse):
            return state_or_error
        state = state_or_error
        body = await request.body()
        if request.url.path in {
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
        }:
            identity_error = _validate_model_identity(body, state)
            if identity_error is not None:
                return identity_error
        target = _upstream_url(state.base_url, request)
        upstream_client = app.state.upstream_client
        if upstream_client is None:
            upstream_client = httpx.AsyncClient(
                follow_redirects=False,
                timeout=httpx.Timeout(300.0, connect=5.0),
            )
            app.state.upstream_client = upstream_client
        upstream_request = upstream_client.build_request(
            request.method,
            target,
            content=body or None,
            headers=_request_headers(request.headers, state),
        )
        try:
            upstream = await upstream_client.send(upstream_request, stream=True)
        except httpx.RequestError as exc:
            return _error(
                502,
                f"Active backend is unreachable: {exc}",
                "upstream_error",
            )

        response_headers = _response_headers(upstream.headers)
        content_type = upstream.headers.get("content-type", "").lower()
        if _requests_stream(body) or content_type.startswith("text/event-stream"):
            return StreamingResponse(
                _stream_and_close(upstream),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=None,
            )
        try:
            content = await upstream.aread()
        finally:
            await upstream.aclose()
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=response_headers,
        )

    @app.get("/health", response_model=None)
    async def gateway_health(challenge: str | None = None) -> dict[str, Any] | JSONResponse:
        try:
            state = read_active_state(app.state.paths)
            state_error = None
        except DeploymentError as exc:
            state = None
            state_error = str(exc)
        status = None
        if state is not None:
            status = await asyncio.to_thread(
                app.state.runtime_manager.inspect_state,
                state,
                check_health=True,
            )
        document: dict[str, Any] = {
            "status": "ok",
            "active": state is not None,
            "ready": status is not None and status.ready,
            "deployment": None if state is None else state.deployment_id,
            "model": None if state is None else state.public_alias,
            "state_error": state_error,
        }
        if challenge is not None:
            try:
                document["attestation"] = create_gateway_attestation(
                    app.state.paths,
                    challenge=challenge,
                    origin=app.state.advertised_origin,
                    active=bool(document["active"]),
                    ready=bool(document["ready"]),
                    deployment=document["deployment"],
                    model=document["model"],
                )
            except IntegrityError as exc:
                return _error(400, str(exc), "invalid_request_error")
        return document

    app.add_api_route("/v1/models", proxy, methods=["GET"])
    app.add_api_route("/v1/chat/completions", proxy, methods=["POST"])
    app.add_api_route("/v1/completions", proxy, methods=["POST"])
    app.add_api_route("/v1/responses", proxy, methods=["POST"])
    return app


app = create_app()
