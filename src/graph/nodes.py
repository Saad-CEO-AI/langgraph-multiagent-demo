from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.llm import get_llm
from src.agents.prompts import CRITIC_PROMPT, RESEARCHER_PROMPT
from src.agents.schemas import CritiqueVerdict
from src.agents.tools import search_tool
from src.graph.state import GraphState
from src.utils.retry import retry_transient

MAX_REVISIONS = 2
MAX_TOOL_ROUNDS = 3
TOOLS = [search_tool]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def researcher_node(state: GraphState) -> dict:
    messages = RESEARCHER_PROMPT.format_messages(
        query=state["query"],
        critique=state.get("critique", ""),
    )

    llm_with_tools = get_llm().bind_tools(TOOLS)
    response = retry_transient(lambda: llm_with_tools.invoke(messages))

    rounds = 0
    while isinstance(response, AIMessage) and response.tool_calls and rounds < MAX_TOOL_ROUNDS:
        messages.append(response)
        for call in response.tool_calls:
            tool = TOOLS_BY_NAME[call["name"]]
            result = tool.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        response = retry_transient(lambda: llm_with_tools.invoke(messages))
        rounds += 1

    if isinstance(response, AIMessage) and response.tool_calls:
        draft = _forced_text_answer(state, messages)
    else:
        draft = response.content

    return {"draft": draft}


def _forced_text_answer(state: GraphState, messages: list) -> str:
    """Tool-call budget exhausted; get a plain-text answer no matter what.

    Continuing the tool-call-laden conversation and asking again (even with
    no tools bound) sometimes still primes this class of model into emitting
    another tool-call-shaped response, which the API rejects outright rather
    than returning as text -- observed directly against Groq's
    openai/gpt-oss-20b, non-deterministically. So this asks fresh, from a
    clean prompt plus a plain-text summary of what was already gathered,
    and falls back to the raw research notes if even that fails, so the
    researcher never returns an empty or crashing draft.
    """
    research_notes = "\n\n".join(m.content for m in messages if isinstance(m, ToolMessage) and m.content)
    fallback_messages = RESEARCHER_PROMPT.format_messages(
        query=state["query"], critique=state.get("critique", "")
    ) + [
        HumanMessage(
            content=(
                f"Research notes gathered so far:\n{research_notes}\n\n"
                "Answer the question now in plain text using these notes. Do not call any tools."
            )
        )
    ]
    try:
        response = retry_transient(lambda: get_llm().invoke(fallback_messages))
        return response.content
    except Exception:  # noqa: BLE001 - last-resort boundary: guarantee non-empty text, never propagate here
        return (
            "Unable to produce a complete answer after repeated tool-call attempts. "
            f"Research notes gathered:\n{research_notes}"
        )


def critic_node(state: GraphState) -> dict:
    messages = CRITIC_PROMPT.format_messages(query=state["query"], draft=state["draft"])

    structured_llm = get_llm().with_structured_output(CritiqueVerdict)
    result: CritiqueVerdict = retry_transient(lambda: structured_llm.invoke(messages))

    return {
        "verdict": result.verdict,
        "critique": result.critique,
        "revision_count": state.get("revision_count", 0) + 1,
    }


def finalizer_node(state: GraphState) -> dict:
    return {"final_answer": state["draft"]}
