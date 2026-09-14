from __future__ import annotations


class AgentError(Exception):
    """Base type for a failure raised deliberately by this service's own logic."""


class GuardrailBlockedError(AgentError):
    """The Guardrail refused the query. Carries the reason shown to the caller."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class UpstreamProviderError(AgentError):
    """Groq's API failed in a way no retry or fallback in this service could recover from."""
