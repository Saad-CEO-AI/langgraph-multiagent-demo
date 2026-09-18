from __future__ import annotations


class OrchestratorError(Exception):
    """Raised when the workflow cannot produce any response at all.

    Individual agent failures degrade gracefully and are recorded in
    GraphState.errors instead of raising -- this is reserved for a total
    failure of the graph itself (e.g. it never reached a terminal state).
    """
