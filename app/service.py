from __future__ import annotations

from functools import lru_cache
from typing import Iterator

import structlog
from langgraph.graph.state import CompiledStateGraph

from app.exceptions import AgentError, GuardrailBlockedError, UpstreamProviderError
from app.graph import build_graph
from app.llm_client import MODEL, get_client
from app.logging_config import timed
from app.prompts import SUPERVISOR_SYNTHESIS_SYSTEM_PROMPT, SUPERVISOR_SYNTHESIS_USER_PROMPT
from app.schemas import AgentEvent
from app.state import GraphState

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _compiled_graph() -> CompiledStateGraph:
    return build_graph()


def _initial_state(query: str) -> GraphState:
    return GraphState(
        query=query,
        guardrail_checked=False,
        guardrail_allowed=True,
        guardrail_reason="",
        draft="",
        reviewed=False,
        critique="",
        verdict="",
        revision_count=0,
        next_agent="",
        supervisor_reason="",
    )


def _translate(node_name: str, node_output: dict) -> Iterator[AgentEvent]:
    """Turns one LangGraph node's output into the client-facing event(s) for it."""
    if node_name == "guardrail":
        if node_output["guardrail_allowed"]:
            yield AgentEvent(agent="guardrail", type="status", content="Query allowed.")
        else:
            yield AgentEvent(
                agent="guardrail", type="status", content=f"Query blocked: {node_output['guardrail_reason']}"
            )
    elif node_name == "supervisor":
        yield AgentEvent(
            agent="supervisor",
            type="status",
            content=f"Routing to {node_output['next_agent']}: {node_output['supervisor_reason']}",
        )
    elif node_name == "researcher":
        yield AgentEvent(agent="researcher", type="draft", content=node_output["draft"])
    elif node_name == "critic":
        if node_output["verdict"] == "approved":
            yield AgentEvent(agent="critic", type="verdict", content="Approved.")
        else:
            yield AgentEvent(agent="critic", type="verdict", content=f"Needs revision: {node_output['critique']}")


def _stream_final_answer(query: str, draft: str) -> Iterator[AgentEvent]:
    """The Supervisor's own call, made once it has routed to "finish": synthesises the
    approved draft into the streamed reply."""
    messages = [
        {"role": "system", "content": SUPERVISOR_SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": SUPERVISOR_SYNTHESIS_USER_PROMPT.format(query=query, draft=draft)},
    ]
    logger.info("node_started", node="supervisor_synthesis")
    with timed(logger, "llm_call", node="supervisor_synthesis"):
        stream = get_client().chat.completions.create(model=MODEL, messages=messages, temperature=0.0, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield AgentEvent(agent="supervisor", type="token", content=delta)
    yield AgentEvent(agent="supervisor", type="final", content="")


def stream_chat_response(query: str) -> Iterator[AgentEvent]:
    """Runs the graph -- Guardrail, then the Supervisor-routed Researcher/Critic loop --
    then streams the Supervisor's final synthesis.

    The graph's own "values" stream is the authoritative final state; nothing
    here re-merges node outputs by hand. "updates" chunks (paired via
    stream_mode=["updates", "values"]) are used only to translate each node's
    output into a client-facing event as it happens.

    Every failure mode ends the stream with one clean AgentEvent rather than
    a truncated response: a blocked query yields a refusal, and any upstream
    failure yields a typed error -- neither ever raises past this function.
    """
    logger.info("request_started", query_length=len(query))
    final_state = _initial_state(query)

    try:
        for mode, chunk in _compiled_graph().stream(final_state, stream_mode=["updates", "values"]):
            if mode == "values":
                final_state = chunk
                continue
            for node_name, node_output in chunk.items():
                yield from _translate(node_name, node_output)

        if not final_state["guardrail_allowed"]:
            raise GuardrailBlockedError(final_state["guardrail_reason"] or "This request cannot be fulfilled.")

        yield from _stream_final_answer(query, final_state["draft"])
        logger.info("request_finished", outcome="answered")

    except GuardrailBlockedError as exc:
        logger.info("request_finished", outcome="refused")
        yield AgentEvent(agent="supervisor", type="refusal", content=exc.reason)
    except AgentError as exc:
        logger.warning("request_finished", outcome="agent_error", error=str(exc))
        yield AgentEvent(agent="supervisor", type="error", content=str(exc))
    except Exception as exc:  # noqa: BLE001 - stream boundary: any unexpected failure becomes one clean event, never a truncated response
        logger.error("request_finished", outcome="upstream_error", error=str(exc))
        yield AgentEvent(agent="supervisor", type="error", content=str(UpstreamProviderError(str(exc))))
