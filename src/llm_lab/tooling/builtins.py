"""Reviewed, read-only tools available to every tool-capable model."""

from __future__ import annotations

import ast
import asyncio
import ipaddress
import json
import logging
import math
import os
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .errors import ToolExecutionError, ToolPolicyError
from .registry import ToolDefinition, ToolProvider, ToolRegistry, ToolsetDefinition
from .python_sandbox import PythonSandboxProvider, PythonSandboxSettings
from .openrouter import OpenRouterProvider, OpenRouterSettings
from .workspace import WorkspaceSettings, WorkspaceToolProvider


DEFAULT_SEARXNG_URL = "http://127.0.0.1:18888"
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_FETCH_CONTENT_TYPES = {
    "application/json",
    "application/xhtml+xml",
    "application/xml",
    "text/html",
    "text/plain",
    "text/xml",
}
_IPV4_TRANSLATION_PREFIXES = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)
Resolver = Callable[[str, int], Awaitable[Sequence[str]]]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuiltinToolSettings:
    searxng_url: str = DEFAULT_SEARXNG_URL
    request_timeout_seconds: float = 12.0
    max_search_response_bytes: int = 1024 * 1024
    max_search_result_bytes: int = 60 * 1024
    max_fetch_response_bytes: int = 1024 * 1024
    max_fetch_characters: int = 60_000
    max_fetch_result_bytes: int = 60 * 1024
    max_redirects: int = 3

    def __post_init__(self) -> None:
        if self.searxng_url:
            try:
                parsed = urlsplit(self.searxng_url)
                port = parsed.port
            except ValueError as exc:
                raise ValueError("LLM_LAB_SEARXNG_URL is malformed") from exc
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or parsed.path.rstrip("/") not in {"", "/search"}
                or port is None
            ):
                raise ValueError(
                    "LLM_LAB_SEARXNG_URL must be an explicit loopback HTTP URL "
                    "with a port and optional /search path"
                )
        if (
            self.request_timeout_seconds <= 0
            or self.max_search_response_bytes < 1024
            or not 1024 <= self.max_search_result_bytes <= 64 * 1024
            or self.max_fetch_response_bytes < 1024
            or self.max_fetch_characters < 1024
            or self.max_fetch_result_bytes < 1024
            or not 0 <= self.max_redirects <= 8
        ):
            raise ValueError("invalid built-in tool limits")

    @classmethod
    def from_environment(cls) -> "BuiltinToolSettings":
        return cls(
            searxng_url=os.environ.get(
                "LLM_LAB_SEARXNG_URL", DEFAULT_SEARXNG_URL
            ).strip()
        )


class _ReadableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._main_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self.main_parts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        lowered = tag.lower()
        if lowered in {"script", "style", "svg", "noscript", "template"}:
            self._ignored_depth += 1
        if lowered in {"article", "main"} and self._ignored_depth == 0:
            self._main_depth += 1
        if lowered == "title" and self._ignored_depth == 0:
            self._in_title = True
        if lowered in {
            "article",
            "br",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "li",
            "main",
            "p",
            "section",
            "td",
            "th",
        }:
            self.parts.append("\n")
            if self._main_depth:
                self.main_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self._in_title = False
        if lowered in {"script", "style", "svg", "noscript", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        if lowered in {"article", "main"} and self._ignored_depth == 0:
            self._main_depth = max(0, self._main_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        self.parts.append(data)
        if self._main_depth:
            self.main_parts.append(data)

    @staticmethod
    def clean(parts: Sequence[str]) -> str:
        lines = []
        for line in "".join(parts).splitlines():
            normalized = " ".join(line.split())
            if normalized:
                lines.append(normalized)
        return "\n".join(lines)

    @property
    def title(self) -> str | None:
        value = " ".join("".join(self.title_parts).split())
        return value or None

    @property
    def text(self) -> str:
        main = self.clean(self.main_parts)
        return main or self.clean(self.parts)


def _json_size(value: Mapping[str, Any]) -> int:
    try:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ToolExecutionError(
            "invalid_search_response",
            "The configured search service returned invalid text",
            retryable=True,
        ) from exc


def _fit_search_result(
    query: str,
    results: Sequence[Mapping[str, Any]],
    maximum: int,
) -> dict[str, Any]:
    """Fit a search result to its exact compact UTF-8 JSON wire budget."""

    bounded_results = [dict(result) for result in results]
    original_snippets = [str(result.get("snippet", "")) for result in results]
    document: dict[str, Any] = {
        "query": query,
        "results": bounded_results,
        "result_count": len(bounded_results),
    }
    if _json_size(document) <= maximum:
        return document

    for result in bounded_results:
        result["snippet"] = ""
    while bounded_results and _json_size(document) > maximum:
        bounded_results.pop()
        original_snippets.pop()
        document["result_count"] = len(bounded_results)
    if _json_size(document) > maximum:
        raise ToolExecutionError(
            "response_too_large",
            "Search result metadata exceeds the serialized result limit",
        )

    # Refill each retained snippet up to the remaining exact byte budget.
    for index, snippet in enumerate(original_snippets):
        lower = 0
        upper = len(snippet)
        while lower < upper:
            midpoint = (lower + upper + 1) // 2
            bounded_results[index]["snippet"] = snippet[:midpoint]
            if _json_size(document) <= maximum:
                lower = midpoint
            else:
                upper = midpoint - 1
        bounded_results[index]["snippet"] = snippet[:lower]
    return document


def _decode(body: bytes, content_type: str) -> str:
    charset = "utf-8"
    for parameter in content_type.split(";")[1:]:
        key, separator, value = parameter.strip().partition("=")
        if separator and key.lower() == "charset":
            charset = value.strip(" \t\"'")[:64] or "utf-8"
            break
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _fit_fetch_result(document: dict[str, Any], maximum: int) -> dict[str, Any]:
    def size(value: Mapping[str, Any]) -> int:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )

    if size(document) <= maximum:
        return document
    text = str(document["text"])
    bounded = {**document, "truncated": True}
    lower = 0
    upper = len(text)
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        bounded["text"] = text[:midpoint]
        if size(bounded) <= maximum:
            lower = midpoint
        else:
            upper = midpoint - 1
    bounded["text"] = text[:lower]
    if size(bounded) > maximum:
        raise ToolExecutionError(
            "response_too_large",
            "Fetched page metadata exceeds the serialized result limit",
        )
    return bounded


async def _default_resolver(host: str, port: int) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    try:
        records = await loop.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ToolExecutionError(
            "fetch_dns_failed",
            "The requested host could not be resolved",
            retryable=True,
        ) from exc
    return tuple(dict.fromkeys(record[4][0] for record in records))


def _validated_public_addresses(addresses: Sequence[str]) -> tuple[ipaddress._BaseAddress, ...]:
    parsed: list[ipaddress._BaseAddress] = []
    for address in addresses:
        try:
            candidate = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError as exc:
            raise ToolPolicyError(
                "fetch_unsafe_destination",
                "The requested host resolved to an invalid address",
            ) from exc
        translated = candidate.version == 6 and any(
            candidate in prefix for prefix in _IPV4_TRANSLATION_PREFIXES
        )
        if not candidate.is_global or candidate.is_reserved or translated:
            raise ToolPolicyError(
                "fetch_unsafe_destination",
                "Requests to private, local, reserved, or special-use networks are blocked",
            )
        parsed.append(candidate)
    if not parsed:
        raise ToolExecutionError(
            "fetch_dns_failed",
            "The requested host did not resolve to an address",
            retryable=True,
        )
    return tuple(sorted(set(parsed), key=lambda value: (value.version, value.packed)))


def _validated_web_url(url: str) -> tuple[str, str, int]:
    if len(url) > 4096 or any(character in url for character in "\r\n\0"):
        raise ToolPolicyError("invalid_url", "URL is malformed or too long")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ToolPolicyError("invalid_url", "URL is malformed") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ToolPolicyError("invalid_url", "Only HTTP and HTTPS URLs are permitted")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ToolPolicyError("invalid_url", "URL must contain a host and no credentials")
    if "%" in parsed.hostname:
        raise ToolPolicyError("invalid_url", "Scoped host addresses are not permitted")
    selected_port = port or (443 if parsed.scheme.lower() == "https" else 80)
    if selected_port not in {80, 443}:
        raise ToolPolicyError(
            "fetch_port_blocked",
            "Web fetch permits only ports 80 and 443",
        )
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise ToolPolicyError("invalid_url", "URL host is invalid") from exc
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ToolPolicyError(
            "fetch_unsafe_destination",
            "Requests to local network names are blocked",
        )
    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )
    return normalized, host, selected_port


def _pinned_url(original: str, address: ipaddress._BaseAddress, port: int) -> str:
    parsed = urlsplit(original)
    ip_literal = f"[{address}]" if address.version == 6 else str(address)
    default = 443 if parsed.scheme == "https" else 80
    authority = ip_literal if port == default else f"{ip_literal}:{port}"
    return urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, ""))


def _host_header(host: str, scheme: str, port: int) -> str:
    default = 443 if scheme == "https" else 80
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        authority = host
    else:
        authority = f"[{literal}]" if literal.version == 6 else str(literal)
    return authority if port == default else f"{authority}:{port}"


def _require_identity_encoding(response: httpx.Response) -> None:
    content_encoding = response.headers.get("content-encoding", "").strip().lower()
    if content_encoding not in {"", "identity"}:
        # httpx decodes content encodings inside aiter_bytes(). Reject before
        # entering that iterator so a tiny compressed body cannot inflate into
        # an unbounded in-memory chunk ahead of our decoded-byte counter.
        raise ToolExecutionError(
            "unsupported_content_encoding",
            "Compressed web responses are not accepted",
        )


async def _read_limited(response: httpx.Response, maximum: int) -> bytes:
    _require_identity_encoding(response)
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError:
            declared = 0
        if declared > maximum:
            raise ToolExecutionError(
                "response_too_large",
                f"Remote response exceeds the {maximum}-byte limit",
            )
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > maximum:
            raise ToolExecutionError(
                "response_too_large",
                f"Remote response exceeds the {maximum}-byte limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_bounded_prefix(
    response: httpx.Response, maximum: int
) -> tuple[bytes, bool, int | None, int]:
    """Read at most ``maximum`` bytes plus one byte of truncation evidence."""

    _require_identity_encoding(response)
    captured = bytearray()
    evidence_limit = maximum + 1
    chunk_size = min(64 * 1024, evidence_limit)
    async for chunk in response.aiter_bytes(chunk_size=chunk_size):
        remaining = evidence_limit - len(captured)
        captured.extend(chunk[:remaining])
        if len(captured) == evidence_limit:
            break
    truncated = len(captured) > maximum
    raw_declared = response.headers.get("content-length")
    try:
        declared = int(raw_declared) if raw_declared is not None else None
    except ValueError:
        declared = None
    if declared is not None and declared < 0:
        declared = None
    return bytes(captured[:maximum]), truncated, declared, len(captured)


def _extraction_quality(text: str) -> str:
    readable_characters = len("".join(text.split()))
    if readable_characters == 0:
        return "empty"
    if readable_characters < 200:
        return "sparse"
    return "usable"


class BuiltinToolProvider(ToolProvider):
    def __init__(
        self,
        settings: BuiltinToolSettings,
        *,
        client: httpx.AsyncClient | None = None,
        resolver: Resolver | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        self._resolver = resolver or _default_resolver
        self._clock = clock or (lambda: datetime.now(UTC))
        self._tools = self._build_tools()

    @property
    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=False,
                trust_env=False,
                timeout=httpx.Timeout(
                    self.settings.request_timeout_seconds, connect=5.0
                ),
                # Requests connect to validated pinned IPs while retaining the
                # original Host/SNI. Pooling by pinned origin could otherwise
                # reuse a socket for distinct hostnames sharing that address.
                limits=httpx.Limits(
                    max_connections=8, max_keepalive_connections=0
                ),
            )
        return self._client

    @property
    def tools(self) -> Sequence[ToolDefinition]:
        return self._tools

    def _build_tools(self) -> tuple[ToolDefinition, ...]:
        closed = {"additionalProperties": False}
        return (
            ToolDefinition(
                name="web_search",
                description=(
                    "Search the public web. Returns concise result titles, URLs, snippets, "
                    "and available publication dates. Use web_fetch to inspect a result."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                    **closed,
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 500},
                        "results": {
                            "type": "array",
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string", "maxLength": 500},
                                    "url": {"type": "string", "maxLength": 4096},
                                    "snippet": {
                                        "type": "string",
                                        "maxLength": 2000,
                                    },
                                    "published_at": {
                                        "type": "string",
                                        "maxLength": 128,
                                    },
                                },
                                "required": ["title", "url", "snippet"],
                                "additionalProperties": False,
                            },
                        },
                        "result_count": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10,
                        },
                    },
                    "required": ["query", "results", "result_count"],
                    **closed,
                },
                handler=self.web_search,
                effect="open_world_search",
                available=bool(self.settings.searxng_url),
            ),
            ToolDefinition(
                name="web_fetch",
                description=(
                    "Fetch readable text from one public HTTP or HTTPS page. Local and "
                    "private network destinations are blocked."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "minLength": 1, "maxLength": 4096}
                    },
                    "required": ["url"],
                    **closed,
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "maxLength": 4096},
                        "status": {"type": "integer", "minimum": 200, "maximum": 299},
                        "content_type": {"type": "string", "maxLength": 128},
                        "title": {
                            "type": ["string", "null"],
                            "maxLength": 500,
                        },
                        "text": {"type": "string", "maxLength": 60_000},
                        "truncated": {"type": "boolean"},
                        "source_host": {"type": "string", "maxLength": 253},
                        "response_bytes_read": {"type": "integer", "minimum": 0},
                        "response_bytes_declared": {
                            "type": ["integer", "null"],
                            "minimum": 0,
                        },
                        "transport_truncated": {"type": "boolean"},
                        "extracted_characters": {"type": "integer", "minimum": 0},
                        "extraction_quality": {
                            "type": "string",
                            "enum": ["empty", "sparse", "usable"],
                        },
                    },
                    "required": [
                        "url",
                        "status",
                        "content_type",
                        "title",
                        "text",
                        "truncated",
                        "source_host",
                        "response_bytes_read",
                        "response_bytes_declared",
                        "transport_truncated",
                        "extracted_characters",
                        "extraction_quality",
                    ],
                    **closed,
                },
                handler=self.web_fetch,
                effect="open_world_fetch",
            ),
            ToolDefinition(
                name="calculator",
                description=(
                    "Evaluate a bounded arithmetic expression using +, -, *, /, //, %, "
                    "powers, unary signs, and parentheses."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                        }
                    },
                    "required": ["expression"],
                    **closed,
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "maxLength": 256},
                        "result": {"type": "number"},
                    },
                    "required": ["expression", "result"],
                    **closed,
                },
                handler=self.calculator,
            ),
            ToolDefinition(
                name="current_time",
                description=(
                    "Return the current time in UTC or in a named IANA timezone such as "
                    "America/Los_Angeles."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 64,
                            "default": "UTC",
                        }
                    },
                    **closed,
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "maxLength": 64},
                        "iso8601": {"type": "string", "maxLength": 128},
                        "utc_offset": {"type": "string", "maxLength": 16},
                    },
                    "required": ["timezone", "iso8601", "utc_offset"],
                    **closed,
                },
                handler=self.current_time,
            ),
        )

    async def web_search(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if not self.settings.searxng_url:
            raise ToolExecutionError(
                "search_not_configured",
                "Web search is not configured",
            )
        query = str(arguments["query"]).strip()
        maximum = int(arguments.get("max_results", 5))
        base = self.settings.searxng_url.rstrip("/")
        endpoint = base if urlsplit(base).path.rstrip("/").endswith("/search") else f"{base}/search"
        try:
            request = self._http_client.build_request(
                "GET",
                endpoint,
                params={"q": query, "format": "json"},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "User-Agent": "llm-lab/0.1",
                },
            )
            response = await self._http_client.send(request, stream=True)
            try:
                if response.status_code >= 400:
                    raise ToolExecutionError(
                        "search_unavailable",
                        "The configured search service returned an error",
                        retryable=response.status_code >= 500,
                    )
                body = await _read_limited(
                    response, self.settings.max_search_response_bytes
                )
            finally:
                await response.aclose()
        except asyncio.CancelledError:
            raise
        except ToolExecutionError:
            raise
        except httpx.RequestError as exc:
            raise ToolExecutionError(
                "search_unavailable",
                "The configured search service is unavailable",
                retryable=True,
            ) from exc
        try:
            document = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ToolExecutionError(
                "invalid_search_response",
                "The configured search service returned invalid JSON",
                retryable=True,
            ) from exc
        candidates = document.get("results", []) if isinstance(document, dict) else []
        results: list[dict[str, Any]] = []
        if isinstance(candidates, list):
            for candidate in candidates[: max(20, maximum * 4)]:
                if not isinstance(candidate, dict):
                    continue
                url = candidate.get("url")
                title = candidate.get("title")
                if not isinstance(url, str) or not isinstance(title, str):
                    continue
                try:
                    normalized_url, result_host, result_port = _validated_web_url(url)
                    await self._resolve_public(result_host, result_port)
                except (ToolPolicyError, ToolExecutionError):
                    continue
                result: dict[str, Any] = {
                    "title": title[:500],
                    "url": normalized_url,
                    "snippet": str(candidate.get("content") or "")[:2000],
                }
                published = candidate.get("publishedDate") or candidate.get(
                    "published_date"
                )
                if published:
                    result["published_at"] = str(published)[:128]
                results.append(result)
                if len(results) >= maximum:
                    break
        return _fit_search_result(
            query,
            results,
            self.settings.max_search_result_bytes,
        )

    async def web_fetch(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        current = str(arguments["url"]).strip()
        redirect_count = 0
        while True:
            original, host, port = _validated_web_url(current)
            addresses = await self._resolve_public(host, port)
            address = addresses[0]
            pinned = _pinned_url(original, address, port)
            parsed = urlsplit(original)
            try:
                request = self._http_client.build_request(
                    "GET",
                    pinned,
                    headers={
                        "Accept": "text/html, text/plain, application/json, application/xml",
                        "Accept-Encoding": "identity",
                        "Connection": "close",
                        "Host": _host_header(host, parsed.scheme, port),
                        "User-Agent": "llm-lab-web-fetch/0.1",
                    },
                    extensions={"sni_hostname": host},
                )
                response = await self._http_client.send(request, stream=True)
            except asyncio.CancelledError:
                raise
            except httpx.RequestError as exc:
                raise ToolExecutionError(
                    "fetch_unavailable",
                    "The requested page could not be reached",
                    retryable=True,
                ) from exc
            try:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise ToolExecutionError(
                            "invalid_redirect",
                            "The requested page returned an invalid redirect",
                        )
                    if redirect_count >= self.settings.max_redirects:
                        raise ToolExecutionError(
                            "redirect_limit",
                            "The requested page exceeded the redirect limit",
                        )
                    redirect_count += 1
                    current = urljoin(original, location)
                    continue
                if response.status_code >= 400:
                    raise ToolExecutionError(
                        "fetch_http_error",
                        f"The requested page returned HTTP {response.status_code}",
                        retryable=response.status_code >= 500,
                    )
                content_type = response.headers.get("content-type", "")
                media_type = content_type.partition(";")[0].strip().lower()
                if media_type not in _FETCH_CONTENT_TYPES:
                    raise ToolExecutionError(
                        "unsupported_content_type",
                        "The requested page is not a supported text document",
                    )
                (
                    body,
                    transport_truncated,
                    response_bytes_declared,
                    response_bytes_read,
                ) = await _read_bounded_prefix(
                    response, self.settings.max_fetch_response_bytes
                )
            finally:
                await response.aclose()
            decoded = _decode(body, content_type)
            title: str | None = None
            if media_type in {"text/html", "application/xhtml+xml"}:
                parser = _ReadableHTML()
                parser.feed(decoded)
                parser.close()
                title = parser.title
                if title is not None:
                    title = title[:500]
                decoded = parser.text
            elif media_type == "application/json":
                try:
                    decoded = json.dumps(
                        json.loads(decoded), ensure_ascii=False, indent=2
                    )
                except json.JSONDecodeError:
                    pass
            truncated = (
                transport_truncated
                or len(decoded) > self.settings.max_fetch_characters
            )
            text = decoded[: self.settings.max_fetch_characters]
            quality = _extraction_quality(text)
            result = _fit_fetch_result(
                {
                    "url": original,
                    "status": response.status_code,
                    "content_type": media_type,
                    "title": title,
                    "text": text,
                    "truncated": truncated,
                    "source_host": host,
                    "response_bytes_read": response_bytes_read,
                    "response_bytes_declared": response_bytes_declared,
                    "transport_truncated": transport_truncated,
                    "extracted_characters": len(decoded),
                    "extraction_quality": quality,
                },
                self.settings.max_fetch_result_bytes,
            )
            log_fetch = LOGGER.warning if transport_truncated else LOGGER.info
            log_fetch(
                "web_fetch_completed host=%s status=%d bytes_read=%d "
                "bytes_declared=%s transport_truncated=%s extracted_characters=%d "
                "quality=%s result_truncated=%s",
                host,
                response.status_code,
                response_bytes_read,
                response_bytes_declared,
                transport_truncated,
                len(decoded),
                quality,
                result["truncated"],
            )
            return result

    async def _resolve_public(
        self, host: str, port: int
    ) -> tuple[ipaddress._BaseAddress, ...]:
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            try:
                resolved = await self._resolver(host, port)
            except asyncio.CancelledError:
                raise
            except ToolExecutionError:
                raise
            except Exception as exc:
                raise ToolExecutionError(
                    "fetch_dns_failed",
                    "The requested host could not be resolved",
                    retryable=True,
                ) from exc
        else:
            resolved = (str(literal),)
        return _validated_public_addresses(resolved)

    async def calculator(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        expression = str(arguments["expression"]).strip()
        try:
            tree = ast.parse(expression, mode="eval")
            result = _evaluate_arithmetic(tree)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ToolExecutionError(
                "invalid_expression",
                "Expression is invalid or exceeds calculator limits",
            ) from exc
        return {"expression": expression, "result": result}

    async def current_time(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        timezone_name = str(arguments.get("timezone", "UTC"))
        try:
            timezone = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolExecutionError(
                "invalid_timezone",
                f"Unknown IANA timezone {timezone_name!r}",
            ) from exc
        current = self._clock().astimezone(timezone)
        return {
            "timezone": timezone_name,
            "iso8601": current.isoformat(),
            "utc_offset": current.strftime("%z"),
        }

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()


def _checked_number(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not a number")
    if isinstance(value, int) and value.bit_length() > 256:
        raise OverflowError("integer exceeds 256 bits")
    if isinstance(value, float) and not math.isfinite(value):
        raise OverflowError("non-finite result")
    if abs(value) > 1e100:
        raise OverflowError("result magnitude exceeds limit")
    return value


def _evaluate_arithmetic(node: ast.AST, *, depth: int = 0) -> int | float:
    if depth > 32:
        raise ValueError("expression is too deep")
    if isinstance(node, ast.Expression):
        return _evaluate_arithmetic(node.body, depth=depth + 1)
    if isinstance(node, ast.Constant):
        return _checked_number(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_arithmetic(node.operand, depth=depth + 1)
        return _checked_number(value if isinstance(node.op, ast.UAdd) else -value)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ):
        left = _evaluate_arithmetic(node.left, depth=depth + 1)
        right = _evaluate_arithmetic(node.right, depth=depth + 1)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 12 or (left == 0 and right < 0):
                raise OverflowError("power exceeds limit")
            return _checked_number(left**right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = left // right
        else:
            result = left % right
        return _checked_number(result)
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def create_builtin_registry(
    settings: BuiltinToolSettings | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    resolver: Resolver | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    provider = BuiltinToolProvider(
        settings or BuiltinToolSettings.from_environment(),
        client=client,
        resolver=resolver,
        clock=clock,
    )
    registry.register_provider(provider)
    registry.register_toolset(
        ToolsetDefinition(
            id="standard-readonly",
            name="Standard read-only",
            description=(
                "Public web research plus deterministic arithmetic and current time. "
                "No tool can mutate local or remote state."
            ),
            tools=("web_search", "web_fetch", "calculator", "current_time"),
        )
    )
    workspace = WorkspaceToolProvider(WorkspaceSettings.from_environment())
    registry.register_provider(workspace)
    registry.workspace_provider = workspace  # type: ignore[attr-defined]
    if workspace.enabled:
        registry.register_toolset(
            ToolsetDefinition(
                id="workspace-files",
                name="Workspace files",
                description=(
                    "Bounded access to one operator-approved workspace. "
                    "Writes are proposals and require explicit approval."
                ),
                tools=("workspace_list", "workspace_read", "workspace_write_proposal"),
            )
        )
    sandbox = PythonSandboxProvider(PythonSandboxSettings.from_environment())
    registry.register_provider(sandbox)
    if sandbox.enabled:
        registry.register_toolset(
            ToolsetDefinition(
                id="python-sandbox",
                name="Python sandbox",
                description=(
                    "Disposable, networkless Python execution. Host workspace changes "
                    "are staged and never committed automatically."
                ),
                tools=("python_sandbox",),
            )
        )
    openrouter = OpenRouterProvider(OpenRouterSettings.from_environment())
    registry.register_provider(openrouter)
    if openrouter.enabled:
        registry.register_toolset(
            ToolsetDefinition(
                id="openrouter-delegation",
                name="OpenRouter delegation",
                description="One bounded call to an operator-approved remote model. Prompts leave this machine.",
                tools=("openrouter_delegate",),
            )
        )
    combined_tools = ["web_search", "web_fetch", "calculator", "current_time"]
    if workspace.enabled:
        combined_tools.extend(("workspace_list", "workspace_read", "workspace_write_proposal"))
    if sandbox.enabled:
        combined_tools.append("python_sandbox")
    if openrouter.enabled:
        combined_tools.append("openrouter_delegate")
    registry.register_toolset(ToolsetDefinition(
        id="assistant-tools", name="Assistant tools",
        description="Operator-configured tools; each client request may enable only a reviewed subset.",
        tools=tuple(combined_tools),
    ))
    return registry


__all__ = [
    "DEFAULT_SEARXNG_URL",
    "BuiltinToolProvider",
    "BuiltinToolSettings",
    "Resolver",
    "create_builtin_registry",
]
