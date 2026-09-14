from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from langgraph.graph.state import CompiledStateGraph

from app.exceptions import AgentError, GuardrailBlockedError, UpstreamProviderError
from app.graph import build_graph
from app.llm_client import MODEL, get_client
from app.prompts import SUPERVISOR_SYSTEM_PROMPT, SUPERVISOR_USER_PROMPT
from app.schemas import AgentEvent
from app.state import GraphState


@lru_cache(maxsize=1)
def _compiled_graph() -> CompiledStateGraph:
    return build_graph()


def _initial_state(query: str) -> GraphState:
    return GraphState(
        query=query,
        guardrail_allowed=True,
        guardrail_reason="",
        draft="",
        critique="",
        verdict="needs_revision",
        revision_count=0,
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
    elif node_name == "researcher":
        yield AgentEvent(agent="researcher", type="draft", content=node_output["draft"])
    elif node_name == "critic":
        if node_output["verdict"] == "approved":
            yield AgentEvent(agent="critic", type="verdict", content="Approved.")
        else:
            yield AgentEvent(agent="critic", type="verdict", content=f"Needs revision: {node_output['critique']}")


def _stream_final_answer(query: str, draft: str) -> Iterator[AgentEvent]:
    """Supervisor's own call: synthesises the approved draft into the streamed reply."""
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": SUPERVISOR_USER_PROMPT.format(query=query, draft=draft)},
    ]
    stream = get_client().chat.completions.create(model=MODEL, messages=messages, temperature=0.0, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield AgentEvent(agent="supervisor", type="token", content=delta)
    yield AgentEvent(agent="supervisor", type="final", content="")


def run_supervisor(query: str) -> Iterator[AgentEvent]:
    """Runs the Guardrail/Researcher/Critic graph, then streams the Supervisor's synthesis.

    Every failure mode ends the stream with one clean AgentEvent rather than
    a truncated response: a blocked query yields a refusal, and any upstream
    failure yields a typed error -- neither ever raises past this function.
    """
    accumulated = _initial_state(query)

    try:
        for step in _compiled_graph().stream(accumulated, stream_mode="updates"):
            for node_name, node_output in step.items():
                accumulated.update(node_output)
                yield from _translate(node_name, node_output)

        if not accumulated["guardrail_allowed"]:
            raise GuardrailBlockedError(accumulated["guardrail_reason"] or "This request cannot be fulfilled.")

        yield from _stream_final_answer(query, accumulated["draft"])

    except GuardrailBlockedError as exc:
        yield AgentEvent(agent="supervisor", type="refusal", content=exc.reason)
    except AgentError as exc:
        yield AgentEvent(agent="supervisor", type="error", content=str(exc))
    except Exception as exc:  # noqa: BLE001 - stream boundary: any unexpected failure becomes one clean event, never a truncated response
        yield AgentEvent(agent="supervisor", type="error", content=str(UpstreamProviderError(str(exc))))
