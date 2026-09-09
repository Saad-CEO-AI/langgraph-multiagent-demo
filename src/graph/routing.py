from __future__ import annotations

from typing import Literal

from src.graph.nodes import MAX_REVISIONS
from src.graph.state import GraphState


def route_after_critique(state: GraphState) -> Literal["revise", "finalize"]:
    if state["verdict"] == "needs_revision" and state["revision_count"] < MAX_REVISIONS:
        return "revise"
    return "finalize"
