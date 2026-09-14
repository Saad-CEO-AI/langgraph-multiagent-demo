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
)
from app.schemas import CritiqueVerdict, GuardrailVerdict
from app.state import GraphState
from app.tools import WEB_SEARCH_TOOL_SCHEMA, web_search

MAX_REVISIONS = 2
MAX_TOOL_ROUNDS = 3


def guardrail_node(state: GraphState) -> dict:
    """Guardrail agent: a precondition on every path, checked before any other agent runs."""
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


def researcher_node(state: GraphState) -> dict:
    """Researcher agent: drafts an answer, searching the web when it needs to."""
    return {"draft": _run_researcher(state["query"], state.get("critique", ""))}


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


def route_after_critique(state: GraphState) -> Literal["revise", "finalize"]:
    if state["verdict"] == "needs_revision" and state["revision_count"] < MAX_REVISIONS:
        return "revise"
    return "finalize"


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("guardrail", guardrail_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("critic", critic_node)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges("guardrail", route_after_guardrail, {"allowed": "researcher", "blocked": END})
    builder.add_edge("researcher", "critic")
    builder.add_conditional_edges("critic", route_after_critique, {"revise": "researcher", "finalize": END})

    return builder.compile()
