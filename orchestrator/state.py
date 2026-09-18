from __future__ import annotations

from typing import Literal, Optional, TypedDict

from orchestrator.schemas import RetrievalIntent, StructuredQueryIntent


class GraphState(TypedDict):
    query: str
    normalized_query: str
    guardrail_checked: bool
    guardrail_allowed: bool
    guardrail_reason: str
    knowledge_base_done: bool
    structured_query_done: bool
    retrieval_intent: Optional[RetrievalIntent]
    structured_query_intent: Optional[StructuredQueryIntent]
    errors: list[str]
    hop_count: int
    next_agent: Literal["guardrail", "knowledge_base", "structured_query", "finish", ""]
    supervisor_reason: str
