from __future__ import annotations

from typing import Literal

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.logging_config import timed
from orchestrator.llm import get_llm
from orchestrator.prompts.guardrail_prompts import GUARDRAIL_PROMPT
from orchestrator.prompts.knowledge_base_prompts import KNOWLEDGE_BASE_PROMPT
from orchestrator.prompts.structured_query_prompts import STRUCTURED_QUERY_PROMPT
from orchestrator.prompts.supervisor_prompts import SUPERVISOR_ROUTER_PROMPT
from orchestrator.schemas import GuardrailVerdict, RetrievalIntent, RoutingDecision, StructuredQueryIntent
from orchestrator.state import GraphState

MAX_HOPS = 6

logger = structlog.get_logger(__name__)


def supervisor_node(state: GraphState) -> dict:
    """Supervisor: normalises the request, then coordinates every hop after it.

    Guardrail is not a choice here -- the very first visit deterministically
    routes to it, matching "before other agents proceed." Every visit after
    that is a genuine LLM call deciding between Knowledge Base, Structured
    Query, or finishing, validated against what's already been done so it
    can't loop forever or re-invoke something twice.
    """
    logger.info("node_started", node="supervisor")
    hop_count = state.get("hop_count", 0) + 1

    if not state.get("guardrail_checked", False):
        normalized = state["query"].strip()
        logger.info("node_finished", node="supervisor", next_agent="guardrail")
        return {
            "normalized_query": normalized,
            "next_agent": "guardrail",
            "supervisor_reason": "Guardrail runs before any other agent, on every request.",
            "hop_count": hop_count,
        }

    if not state.get("guardrail_allowed", True):
        logger.info("node_finished", node="supervisor", next_agent="finish")
        return {"next_agent": "finish", "supervisor_reason": "Guardrail blocked this request.", "hop_count": hop_count}

    knowledge_base_done = state.get("knowledge_base_done", False)
    structured_query_done = state.get("structured_query_done", False)

    try:
        with timed(logger, "llm_call", node="supervisor"):
            chain = SUPERVISOR_ROUTER_PROMPT | get_llm().with_structured_output(RoutingDecision)
            decision: RoutingDecision = chain.invoke(
                {
                    "query": state["normalized_query"],
                    "knowledge_base_done": knowledge_base_done,
                    "structured_query_done": structured_query_done,
                }
            )
        proposed, reason = decision.next, decision.reason
    except Exception as exc:  # noqa: BLE001 - a failed routing call still has to conclude, not crash the graph
        proposed, reason = "finish", f"Routing decision failed ({exc}); concluding with what is available."

    next_agent = _validated_next(
        guardrail_checked=True,
        guardrail_allowed=True,
        knowledge_base_done=knowledge_base_done,
        structured_query_done=structured_query_done,
        hop_count=hop_count,
        proposed=proposed,
    )
    logger.info("node_finished", node="supervisor", next_agent=next_agent)
    return {"next_agent": next_agent, "supervisor_reason": reason, "hop_count": hop_count}


def _validated_next(
    guardrail_checked: bool,
    guardrail_allowed: bool,
    knowledge_base_done: bool,
    structured_query_done: bool,
    hop_count: int,
    proposed: str,
) -> str:
    """The deterministic backstop on the Supervisor's LLM decision, in priority order."""
    if not guardrail_checked:
        return "guardrail"
    if not guardrail_allowed:
        return "finish"
    if hop_count >= MAX_HOPS:
        return "finish"
    if proposed == "knowledge_base" and not knowledge_base_done:
        return "knowledge_base"
    if proposed == "structured_query" and not structured_query_done:
        return "structured_query"
    if proposed == "finish":
        return "finish"
    # An invalid proposal, or a request to redo something already done -- conclude rather than loop.
    return "finish"


def route_from_supervisor(state: GraphState) -> Literal["guardrail", "knowledge_base", "structured_query", "finish"]:
    return state["next_agent"]


def guardrail_node(state: GraphState) -> dict:
    """Guardrail agent: validates scope, safety, and policy before anything else proceeds."""
    logger.info("node_started", node="guardrail")
    try:
        with timed(logger, "llm_call", node="guardrail"):
            chain = GUARDRAIL_PROMPT | get_llm().with_structured_output(GuardrailVerdict)
            verdict: GuardrailVerdict = chain.invoke({"query": state["normalized_query"]})
    except Exception as exc:  # noqa: BLE001 - a guardrail that fails open on an error stops being a guardrail
        logger.warning("node_finished", node="guardrail", error=str(exc))
        return {
            "guardrail_checked": True,
            "guardrail_allowed": False,
            "guardrail_reason": "Guardrail could not be evaluated; failing closed.",
            "errors": state.get("errors", []) + [f"guardrail: {exc}"],
        }

    logger.info("node_finished", node="guardrail", allowed=verdict.allowed)
    return {"guardrail_checked": True, "guardrail_allowed": verdict.allowed, "guardrail_reason": verdict.reason}


def knowledge_base_node(state: GraphState) -> dict:
    """Knowledge Base agent: prepares retrieval intent. No search is performed."""
    logger.info("node_started", node="knowledge_base")
    try:
        with timed(logger, "llm_call", node="knowledge_base"):
            chain = KNOWLEDGE_BASE_PROMPT | get_llm().with_structured_output(RetrievalIntent)
            intent: RetrievalIntent = chain.invoke({"query": state["normalized_query"]})
        logger.info("node_finished", node="knowledge_base")
        return {"retrieval_intent": intent, "knowledge_base_done": True}
    except Exception as exc:  # noqa: BLE001 - a failed sub-agent still lets the workflow conclude with what it has
        logger.warning("node_finished", node="knowledge_base", error=str(exc))
        return {"knowledge_base_done": True, "errors": state.get("errors", []) + [f"knowledge_base: {exc}"]}


def structured_query_node(state: GraphState) -> dict:
    """Structured Query agent: prepares typed query intent. No database call is made."""
    logger.info("node_started", node="structured_query")
    try:
        with timed(logger, "llm_call", node="structured_query"):
            chain = STRUCTURED_QUERY_PROMPT | get_llm().with_structured_output(StructuredQueryIntent)
            intent: StructuredQueryIntent = chain.invoke({"query": state["normalized_query"]})
        logger.info("node_finished", node="structured_query")
        return {"structured_query_intent": intent, "structured_query_done": True}
    except Exception as exc:  # noqa: BLE001 - a failed sub-agent still lets the workflow conclude with what it has
        logger.warning("node_finished", node="structured_query", error=str(exc))
        return {"structured_query_done": True, "errors": state.get("errors", []) + [f"structured_query: {exc}"]}


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("knowledge_base", knowledge_base_node)
    builder.add_node("structured_query", structured_query_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "guardrail": "guardrail",
            "knowledge_base": "knowledge_base",
            "structured_query": "structured_query",
            "finish": END,
        },
    )
    builder.add_edge("guardrail", "supervisor")
    builder.add_edge("knowledge_base", "supervisor")
    builder.add_edge("structured_query", "supervisor")

    return builder.compile()
