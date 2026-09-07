"""Stable, non-sensitive failures crossing the tool/agent boundary."""

from __future__ import annotations


class ToolingError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ToolPolicyError(ToolingError):
    pass


class ToolExecutionError(ToolingError):
    pass


class AgentUpstreamError(ToolingError):
    pass


__all__ = [
    "AgentUpstreamError",
    "ToolExecutionError",
    "ToolPolicyError",
    "ToolingError",
]
