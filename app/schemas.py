from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequestPayload(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")


class GuardrailVerdict(BaseModel):
    allowed: bool = Field(description="False if the query violates policy and must be refused.")
    reason: str = Field(description="Why the query was blocked. Empty string if allowed.")


class SupervisorRoutingDecision(BaseModel):
    next: Literal["researcher", "critic", "finish"] = Field(description="Which agent should act next.")
    reason: str = Field(description="One short sentence explaining the choice.")


class CritiqueVerdict(BaseModel):
    verdict: Literal["approved", "needs_revision"] = Field(
        description="approved if the draft fully and accurately answers the "
        "question; needs_revision otherwise."
    )
    critique: str = Field(
        description="Specific, actionable feedback the researcher can act on. "
        "Empty string if approved."
    )


class AgentEvent(BaseModel):
    """One line of the streamed response. Every chunk sent to the client has this shape."""

    agent: Literal["guardrail", "researcher", "critic", "supervisor"]
    type: Literal["status", "draft", "verdict", "token", "final", "refusal", "error"]
    content: str
