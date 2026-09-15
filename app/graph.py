from __future__ import annotations

import json
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.llm_client import MODEL, get_client
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


def guardrail_node(state: GraphState) -> dict:
    """Guardrail agent: a precondition on every path, checked before any other agent runs.

    Not routable by the Supervisor -- a guardrail the router could decide to
    skip isn't a guardrail. It runs once, deterministically, before the
    Supervisor ever sees the query.
    """
    messages = [
        {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
        {"role": "user", "content": GUARDRAIL_USER_PROMPT.format(query=state["query"])},
    ]
    content = (
        get_client()
        .chat.completions.create(model=MODEL, messages=messages, temperature=0.0, response_format={"type": "json_object"})
        .choices[0]
        .message.content
    )

    try:
        verdict = GuardrailVerdict.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError):
        # A guardrail that fails open on a parse error stops being a guardrail.
        verdict = GuardrailVerdict(allowed=False, reason="Guardrail response could not be parsed; failing closed.")

    return {"guardrail_allowed": verdict.allowed, "guardrail_reason": verdict.reason}


def route_after_guardrail(state: GraphState) -> Literal["allowed", "blocked"]:
    return "allowed" if state["guardrail_allowed"] else "blocked"


def supervisor_node(state: GraphState) -> dict:
    """Supervisor: the graph's hub. Every path passes back through here, and it decides
    what happens next -- not a fixed edge function reading a single field.

    Genuinely consulted on every visit (a real LLM call, not a relabelled
    if/else), but its answer is validated against the actual state before
    being trusted: a routing decision that skips Critic, or loops past the
    revision cap, is a worse failure than one that's occasionally
    over-cautious. The LLM decides; the state has the final say.
    """
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
                has_draft=has_draft,
                reviewed=reviewed,
                verdict=verdict if reviewed else "(not yet reviewed)",
                critique=critique or "(none)",
                revision_count=revision_count,
                max_revisions=MAX_REVISIONS,
            ),
        },
    ]
    content = (
        get_client()
        .chat.completions.create(model=MODEL, messages=messages, temperature=0.0, response_format={"type": "json_object"})
        .choices[0]
        .message.content
    )

    try:
        decision = SupervisorRoutingDecision.model_validate(json.loads(content))
        proposed, reason = decision.next, decision.reason
    except (json.JSONDecodeError, ValueError):
        proposed, reason = "researcher", "Routing response could not be parsed; defaulting to research."

    next_agent = _validated_next(has_draft, reviewed, verdict, revision_count, proposed)
    return {"next_agent": next_agent, "supervisor_reason": reason}


def _validated_next(has_draft: bool, reviewed: bool, verdict: str, revision_count: int, proposed: str) -> str:
    """The deterministic backstop on the Supervisor's LLM decision, in priority order."""
    if not has_draft:
        return "researcher"
    if not reviewed:
        return "critic"
    if revision_count >= MAX_REVISIONS:
        return "finish"
    if verdict == "needs_revision":
        return proposed if proposed in ("researcher", "finish") else "researcher"
    return "finish"


def route_from_supervisor(state: GraphState) -> Literal["researcher", "critic", "finish"]:
    return state["next_agent"]


def researcher_node(state: GraphState) -> dict:
    """Researcher agent: drafts an answer, searching the web when it needs to."""
    draft = _run_researcher(state["query"], state.get("critique", ""))
    return {"draft": draft, "reviewed": False}


def _run_researcher(query: str, critique: str) -> str:
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": RESEARCHER_USER_PROMPT.format(query=query, critique=critique or "(none)")},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
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

    return _forced_text_answer(query, critique)


def _forced_text_answer(query: str, critique: str) -> str:
    """Tool-call budget exhausted; ask fresh, from a clean prompt, for a plain-text answer.

    Continuing the tool-call-laden conversation here can still prime this
    model into emitting another tool call even with none declared, which the
    API rejects outright -- so this asks fresh instead of reusing that
    history, and never returns an empty draft even if this also fails.
    """
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": RESEARCHER_USER_PROMPT.format(query=query, critique=critique or "(none)")},
        {"role": "user", "content": "Answer now in plain text using what you already know. Do not call any tools."},
    ]
    try:
        message = get_client().chat.completions.create(model=MODEL, messages=messages, temperature=0.0).choices[0].message
        return message.content or ""
    except Exception:  # noqa: BLE001 - last-resort boundary: guarantee non-empty text, never propagate here
        return "Unable to produce a complete answer after repeated search attempts."


def critic_node(state: GraphState) -> dict:
    """Critic agent: grades the draft and returns a typed verdict, not free text."""
    verdict = _run_critic(state["query"], state["draft"])
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
    content = (
        get_client()
        .chat.completions.create(model=MODEL, messages=messages, temperature=0.0, response_format={"type": "json_object"})
        .choices[0]
        .message.content
    )

    try:
        return CritiqueVerdict.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError):
        return CritiqueVerdict(verdict="needs_revision", critique="Critic returned a malformed verdict; revise and retry.")


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("guardrail", guardrail_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("critic", critic_node)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges("guardrail", route_after_guardrail, {"allowed": "supervisor", "blocked": END})
    builder.add_conditional_edges(
        "supervisor", route_from_supervisor, {"researcher": "researcher", "critic": "critic", "finish": END}
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("critic", "supervisor")

    return builder.compile()
