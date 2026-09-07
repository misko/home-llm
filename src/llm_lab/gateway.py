"""Small OpenAI-compatible gateway for the currently active deployment."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import secrets
import socket
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .attestation import configured_gateway_origin, create_gateway_attestation
from .errors import DeploymentError, IntegrityError
from .paths import LabPaths
from .runtime import RuntimeManager, RuntimeState, read_active_state

if TYPE_CHECKING:
    from .console_api import ConsoleController
    from .tooling.orchestrator import AgentRunner
    from .tooling.registry import ToolRegistry


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
LOGGER = logging.getLogger(__name__)
_RESERVED_TEST_HOST_SUFFIXES = (".invalid", ".test")


def _canonical_host(value: str) -> str:
    candidate = value.strip().lower().rstrip(".")
    if not candidate or len(candidate) > 253 or "%" in candidate:
        raise ValueError("invalid allowed host")
    try:
        return ipaddress.ip_address(candidate).compressed
    except ValueError:
        try:
            canonical = candidate.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("invalid allowed host") from exc
        labels = canonical.split(".")
        if any(
            not label
            or len(label) > 63
            or not label.replace("-", "a").isalnum()
            or label.startswith("-")
            or label.endswith("-")
            for label in labels
        ):
            raise ValueError("invalid allowed host")
        return canonical


def _configured_allowed_hosts(configured: Sequence[str] | None) -> frozenset[str]:
    if configured is None:
        environment = os.environ.get("LLM_LAB_ALLOWED_HOSTS")
        if environment is not None:
            values = tuple(
                part.strip() for part in environment.split(",") if part.strip()
            )
        else:
            values = ("127.0.0.1", "::1", "localhost", socket.gethostname())
    else:
        if isinstance(configured, (str, bytes)):
            raise ValueError("allowed_hosts must be a sequence of host names")
        values = tuple(configured)
    if not values:
        raise ValueError("at least one allowed gateway host is required")
    try:
        return frozenset(_canonical_host(value) for value in values)
    except (AttributeError, TypeError) as exc:
        raise ValueError("allowed gateway hosts must be strings") from exc


def _request_host(scope: Mapping[str, Any]) -> str | None:
    values = [
        value
        for key, value in scope.get("headers", ())
        if key.lower() == b"host"
    ]
    if len(values) != 1:
        return None
    try:
        authority = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    if (
        not authority
        or authority != authority.strip()
        or any(character in authority for character in "\r\n\0/@\\")
    ):
        return None
    if authority.startswith("["):
        closing = authority.find("]")
        if closing < 0:
            return None
        host = authority[1:closing]
        remainder = authority[closing + 1 :]
        if remainder and (not remainder.startswith(":") or not remainder[1:].isdigit()):
            return None
        port_text = remainder[1:] if remainder else ""
    else:
        if authority.count(":") > 1:
            return None
        host, separator, port_text = authority.partition(":")
        if separator and not port_text.isdigit():
            return None
    if port_text and not 1 <= int(port_text) <= 65_535:
        return None
    try:
        return _canonical_host(host)
    except ValueError:
        return None


class GatewayHostMiddleware:
    """Reject unreviewed Host values before request bodies reach the gateway."""

    def __init__(self, app: Any, *, allowed_hosts: Sequence[str]) -> None:
        self.app = app
        self.allowed_hosts = _configured_allowed_hosts(allowed_hosts)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        host = _request_host(scope)
        reserved_test_host = bool(
            host
            and (
                host in {"testserver", "invalid", "test"}
                or host.endswith(_RESERVED_TEST_HOST_SUFFIXES)
            )
        )
        if host not in self.allowed_hosts and not reserved_test_host:
            if scope.get("type") == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            response = _error(
                400,
                "Host header is not permitted",
                "invalid_host",
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


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
    console_controller: "ConsoleController | None" = None,
    console_static_directory: str | Path | None = None,
    enable_console: bool = True,
    enable_agent: bool = True,
    agent_registry: "ToolRegistry | None" = None,
    agent_runner: "AgentRunner | None" = None,
    agent_max_request_bytes: int = 64 * 1024 * 1024,
    allowed_hosts: Sequence[str] | None = None,
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
        console = getattr(application.state, "console_controller", None)
        if application.state.owns_console_controller and console is not None:
            console.close()
        tools = getattr(application.state, "agent_registry", None)
        if getattr(application.state, "owns_agent_registry", False) and tools is not None:
            await tools.aclose()
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
    app.state.owns_console_controller = console_controller is None
    reviewed_hosts = _configured_allowed_hosts(allowed_hosts)

    # Install the receive-stream limiter before the decorator-based auth
    # middleware below. Starlette then places authentication outside the body
    # reader, while the limiter remains directly outside FastAPI routing so its
    # typed 413 cannot be wrapped by BaseHTTPMiddleware task boundaries.
    if enable_agent:
        from .agent_api import AgentRequestBodyLimitMiddleware
        from .tooling.orchestrator import AgentLimits

        if not 1024 <= agent_max_request_bytes <= 64 * 1024 * 1024:
            raise ValueError(
                "agent request body limit must be between 1 KiB and 64 MiB"
            )
        admission_limits = agent_runner.limits if agent_runner else AgentLimits()
        app.add_middleware(
            AgentRequestBodyLimitMiddleware,
            maximum_bytes=agent_max_request_bytes,
            maximum_inflight_requests=(
                admission_limits.max_concurrent_turns
                + admission_limits.max_queued_turns
            ),
        )
    app.add_middleware(
        GatewayHostMiddleware,
        allowed_hosts=tuple(reviewed_hosts),
    )

    def get_upstream_client() -> httpx.AsyncClient:
        upstream_client = app.state.upstream_client
        if upstream_client is None:
            upstream_client = httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                timeout=httpx.Timeout(300.0, connect=5.0),
            )
            app.state.upstream_client = upstream_client
        return upstream_client

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        expected = app.state.api_key
        protected = request.url.path.startswith(("/v1/", "/api/v1/"))
        if expected and protected:
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
        response = await call_next(request)
        if request.url.path.startswith(("/ui", "/api/v1/")):
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; "
                "style-src 'self' 'unsafe-inline'; script-src 'self'; "
                "font-src 'self'; object-src 'none'; base-uri 'none'; "
                "frame-ancestors 'none'",
            )
        return response

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
        upstream_client = get_upstream_client()
        upstream_request = upstream_client.build_request(
            request.method,
            target,
            content=body or None,
            headers=_request_headers(request.headers, state),
        )
        try:
            upstream = await upstream_client.send(upstream_request, stream=True)
        except httpx.RequestError as exc:
            LOGGER.warning(
                "Active backend request failed for deployment %s (%s)",
                state.deployment_id,
                type(exc).__name__,
            )
            return _error(
                502,
                "Active backend is unreachable",
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
    if enable_agent:
        from .agent_api import install_agent_api

        install_agent_api(
            app,
            active_state=active_state,
            client_getter=get_upstream_client,
            registry=agent_registry,
            runner=agent_runner,
            maximum_request_bytes=agent_max_request_bytes,
            body_limit_installed=True,
        )
    if enable_console:
        from .console_api import install_console

        install_console(
            app,
            resolved_paths,
            runtime_manager=app.state.runtime_manager,
            controller=console_controller,
            static_directory=console_static_directory,
        )
    return app


app = create_app()
