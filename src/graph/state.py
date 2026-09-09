from __future__ import annotations

from typing import Literal, TypedDict


class GraphState(TypedDict):
    query: str
    draft: str
    critique: str
    verdict: Literal["approved", "needs_revision"]
    revision_count: int
    final_answer: str
