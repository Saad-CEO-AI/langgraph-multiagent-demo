from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ConversationRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's message.")
    conversation_id: Optional[str] = Field(default=None, description="Caller-supplied id for correlating turns.")


class GuardrailVerdict(BaseModel):
    allowed: bool = Field(description="False if the request violates policy and must be refused.")
    reason: str = Field(description="Why the request was blocked. Empty string if allowed.")


class RoutingDecision(BaseModel):
    next: Literal["knowledge_base", "structured_query", "finish"] = Field(
        description="Which agent should act next, once Guardrail has already cleared the request."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


class RetrievalIntent(BaseModel):
    """What a real hybrid-search call would do. No search is actually performed yet."""

    rewritten_query: str = Field(description="The query rewritten for retrieval, if different from the original.")
    filters: list[str] = Field(default_factory=list, description="Filter hints implied by the question, e.g. a date range or sender.")
    top_k: int = Field(default=5, ge=1, le=50, description="How many results a real search would request.")
    reasoning: str = Field(description="Why this retrieval shape answers the question.")


class StructuredQueryIntent(BaseModel):
    """What a real typed database query would run. No query is actually executed yet."""

    query_name: Literal["count_records", "list_records", "thread_lookup"] = Field(
        description="The named, parameterised query this maps to."
    )
    parameters: dict[str, str] = Field(default_factory=dict, description="Typed parameters for that named query.")
    reasoning: str = Field(description="Why this query answers the question.")


class ExecutionMetadata(BaseModel):
    guardrail_checked: bool
    guardrail_allowed: bool
    knowledge_base_used: bool
    structured_query_used: bool
    agents_invoked: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    answer: str
    refused: bool = False
    metadata: ExecutionMetadata
