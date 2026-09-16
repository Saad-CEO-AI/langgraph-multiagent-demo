from __future__ import annotations

import json
from typing import Literal

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.llm_client import MODEL, get_client
from app.logging_config import timed
from app.prompts import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_USER_PROMPT,
    GUARDRAIL_SYSTEM_PROMPT,
    GUARDRAIL_USER_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_USER_PROMPT,
    SUPERVISOR_ROUTER_SYSTEM_PROMPT,
    SUPERVISOR_ROUTER_USER_PROMPT,
)
from app.schemas import CritiqueVerdict, GuardrailVerdict, SupervisorRoutingDecision
from app.state import GraphState
from app.tools import WEB_SEARCH_TOOL_SCHEMA, web_search

MAX_REVISIONS = 2
MAX_TOOL_ROUNDS = 3

logger = structlog.get_logger(__name__)


def guardrail_node(state: GraphState) -> dict:
    """Guardrail agent: a secondary, deeper check -- only reached when the Supervisor's
    own triage flags a query as worth it, never run on every query.

    Only reachable as the Supervisor's very first decision (see
    _validated_next): once any draft exists, Guardrail is no longer in the
    Supervisor's option set, so it cannot be invoked mid-flow.
    """
    logger.info("node_started", node="guardrail")
    messages = [
        {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GUARDRAIL_USER_PROMPT.format(
                query=state["query"], flag_reason=state.get("supervisor_reason") or "(no specific reason given)"
            ),
        },
    ]
    with timed(logger, "llm_call", node="guardrail"):
        content = (
            get_client()
            .chat.completions.create(
                model=MODEL, messages=messages, temperature=0.0, response_format={"type": "json_object"}
            )
            .choices[0]
            .message.content
        )

    try:
        verdict = GuardrailVerdict.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError):
        # A guardrail that fails open on a parse error stops being a guardrail.
        verdict = GuardrailVerdict(allowed=False, reason="Guardrail response could not be parsed; failing closed.")

    logger.info("node_finished", node="guardrail", allowed=verdict.allowed)
    return {"guardrail_checked": True, "guardrail_allowed": verdict.allowed, "guardrail_reason": verdict.reason}


def supervisor_node(state: GraphState) -> dict:
    """Supervisor: the first agent to see every query, and the graph's hub -- every
    other agent returns here, and it decides what happens next each time.

    Genuinely consulted on every visit (a real LLM call, not a relabelled
    if/else), including its own triage judgment on whether a query needs
    Guardrail at all. That judgment, like every other routing choice, is
    validated against the actual state before being trusted: the LLM decides,
    the state has the final say.
    """
    logger.info("node_started", node="supervisor")
    guardrail_checked = state.get("guardrail_checked", False)
    guardrail_allowed = state.get("guardrail_allowed", True)
    has_draft = bool(state.get("draft"))
    reviewed = state.get("reviewed", False)
    verdict = state.get("verdict", "")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    messages = [
        {"role": "system", "content": SUPERVISOR_ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SUPERVISOR_ROUTER_USER_PROMPT.format(
                query=state["query"],
                guardrail_checked=guardrail_checked,
                guardrail_verdict=("allowed" if guardrail_allowed else "blocked") if guardrail_checked else "(not checked)",
                has_draft=has_draft,
                reviewed=reviewed,
                verdict=verdict if reviewed else "(not yet reviewed)",
                critique=critique or "(none)",
                revision_count=revision_count,
                max_revisions=MAX_REVISIONS,
            ),
        },
    ]
    with timed(logger, "llm_call", node="supervisor"):
        content = (
            get_client()
            .chat.completions.create(
                model=MODEL, messages=messages, temperature=0.0, response_format={"type": "json_object"}
            )
            .choices[0]
            .message.content
        )

    try:
        decision = SupervisorRoutingDecision.model_validate(json.loads(content))
        proposed, reason = decision.next, decision.reason
    except (json.JSONDecodeError, ValueError):
        # An unreadable routing decision is the one moment we can't trust our own
        # judgment about whether this query is safe -- fail toward more scrutiny,
        # not less. Harmless once a draft exists: _validated_next only honors
        # "guardrail" before any draft exists, and computes the correct step
        # regardless everywhere else.
        proposed, reason = "guardrail", "Routing response could not be parsed; escalating to Guardrail to be safe."

    next_agent = _validated_next(guardrail_checked, guardrail_allowed, has_draft, reviewed, verdict, revision_count, proposed)
    logger.info("node_finished", node="supervisor", next_agent=next_agent)
    return {"next_agent": next_agent, "supervisor_reason": reason}


def _validated_next(
    guardrail_checked: bool,
    guardrail_allowed: bool,
    has_draft: bool,
    reviewed: bool,
    verdict: str,
    revision_count: int,
    proposed: str,
) -> str:
    """The deterministic backstop on the Supervisor's LLM decision, in priority order."""
    if guardrail_checked and not guardrail_allowed:
        return "finish"  # blocked; nothing left to do but let the caller turn this into a refusal
    if not has_draft and not guardrail_checked:
        # The one genuinely open call: does this query need a deeper Guardrail read first?
        return proposed if proposed in ("guardrail", "researcher") else "researcher"
    if not has_draft:
        return "researcher"
    if not reviewed:
        return "critic"
    if revision_count >= MAX_REVISIONS:
        return "finish"
    if verdict == "needs_revision":
        return proposed if proposed in ("researcher", "finish") else "researcher"
    return "finish"


def route_from_supervisor(state: GraphState) -> Literal["guardrail", "researcher", "critic", "finish"]:
    return state["next_agent"]


def researcher_node(state: GraphState) -> dict:
    """Researcher agent: drafts an answer, searching the web when it needs to.

    Passes its own previous draft, not just the critique, on a revision pass
    -- "paragraph 3 is wrong" means nothing without the paragraph it refers to.
    """
    logger.info("node_started", node="researcher")
    draft = _run_researcher(state["query"], state.get("draft", ""), state.get("critique", ""))
    logger.info("node_finished", node="researcher", draft_length=len(draft))
    return {"draft": draft, "reviewed": False}


def _run_researcher(query: str, previous_draft: str, critique: str) -> str:
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RESEARCHER_USER_PROMPT.format(
                query=query, previous_draft=previous_draft or "(none)", critique=critique or "(none)"
            ),
        },
    ]

    for round_number in range(MAX_TOOL_ROUNDS):
        with timed(logger, "llm_call", node="researcher", round=round_number):
            message = (
                get_client()
                .chat.completions.create(model=MODEL, messages=messages, tools=[WEB_SEARCH_TOOL_SCHEMA], temperature=0.0)
                .choices[0]
                .message
            )

        if not message.tool_calls:
            return message.content or ""

        messages.append(
            {"role": "assistant", "content": message.content, "tool_calls": [tc.model_dump() for tc in message.tool_calls]}
        )
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = web_search(args.get("query", query))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return _forced_text_answer(query, previous_draft, critique)


def _forced_text_answer(query: str, previous_draft: str, critique: str) -> str:
    """Tool-call budget exhausted; ask fresh, from a clean prompt, for a plain-text answer.

    Continuing the tool-call-laden conversation here can still prime this
    model into emitting another tool call even with none declared, which the
    API rejects outright -- so this asks fresh instead of reusing that
    history, and never returns an empty draft even if this also fails.
    """
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RESEARCHER_USER_PROMPT.format(
                query=query, previous_draft=previous_draft or "(none)", critique=critique or "(none)"
            ),
        },
        {"role": "user", "content": "Answer now in plain text using what you already know. Do not call any tools."},
    ]
    try:
        with timed(logger, "llm_call", node="researcher", round="forced_text"):
            message = get_client().chat.completions.create(model=MODEL, messages=messages, temperature=0.0).choices[0].message
        return message.content or ""
    except Exception:  # noqa: BLE001 - last-resort boundary: guarantee non-empty text, never propagate here
        return "Unable to produce a complete answer after repeated search attempts."


def critic_node(state: GraphState) -> dict:
    """Critic agent: grades the draft and returns a typed verdict, not free text."""
    logger.info("node_started", node="critic")
    verdict = _run_critic(state["query"], state["draft"])
    logger.info("node_finished", node="critic", verdict=verdict.verdict)
    return {
        "verdict": verdict.verdict,
        "critique": verdict.critique,
        "revision_count": state.get("revision_count", 0) + 1,
        "reviewed": True,
    }


def _run_critic(query: str, draft: str) -> CritiqueVerdict:
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": CRITIC_USER_PROMPT.format(query=query, draft=draft)},
    ]
    with timed(logger, "llm_call", node="critic"):
        content = (
            get_client()
            .chat.completions.create(
                model=MODEL, messages=messages, temperature=0.0, response_format={"type": "json_object"}
            )
            .choices[0]
            .message.content
        )

    try:
        return CritiqueVerdict.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError):
        return CritiqueVerdict(verdict="needs_revision", critique="Critic returned a malformed verdict; revise and retry.")


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("critic", critic_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"guardrail": "guardrail", "researcher": "researcher", "critic": "critic", "finish": END},
    )
    builder.add_edge("guardrail", "supervisor")
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("critic", "supervisor")

    return builder.compile()
