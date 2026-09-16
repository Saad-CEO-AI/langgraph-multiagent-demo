from __future__ import annotations

from typing import Literal, TypedDict


class GraphState(TypedDict):
    query: str
    guardrail_checked: bool
    guardrail_allowed: bool
    guardrail_reason: str
    draft: str
    reviewed: bool
    critique: str
    verdict: Literal["approved", "needs_revision", ""]
    revision_count: int
    next_agent: Literal["guardrail", "researcher", "critic", "finish", ""]
    supervisor_reason: str
