from __future__ import annotations

from functools import lru_cache

import structlog
from langgraph.graph.state import CompiledStateGraph

from app.logging_config import timed
from orchestrator.exceptions import OrchestratorError
from orchestrator.graph import build_graph
from orchestrator.llm import get_llm
from orchestrator.prompts.supervisor_prompts import SUPERVISOR_SYNTHESIS_PROMPT
from orchestrator.schemas import ConversationRequest, ConversationResponse, ExecutionMetadata
from orchestrator.state import GraphState

logger = structlog.get_logger(__name__)


class ConversationOrchestrator:
    """Main entry point: accepts a conversation request, runs the supervised
    agent workflow, and returns the final response.

    No API, no server, no real tool connections -- this is the core workflow
    only. Building one instance compiles the graph once; call handle() per
    request.
    """

    def __init__(self) -> None:
        self._graph: CompiledStateGraph = build_graph()

    def handle(self, request: ConversationRequest) -> ConversationResponse:
        logger.info("conversation_started", conversation_id=request.conversation_id, query_length=len(request.query))

        try:
            final_state: GraphState = self._graph.invoke(self._initial_state(request))
        except Exception as exc:
            logger.error("conversation_failed", error=str(exc))
            error = OrchestratorError(str(exc))
            return ConversationResponse(
                answer="Something went wrong while processing this request.",
                refused=False,
                metadata=ExecutionMetadata(
                    guardrail_checked=False,
                    guardrail_allowed=True,
                    knowledge_base_used=False,
                    structured_query_used=False,
                    agents_invoked=[],
                    errors=[str(error)],
                ),
            )

        response = self._build_response(request, final_state)
        logger.info("conversation_finished", refused=response.refused, agents_invoked=response.metadata.agents_invoked)
        return response

    @staticmethod
    def _initial_state(request: ConversationRequest) -> GraphState:
        return GraphState(
            query=request.query,
            normalized_query="",
            guardrail_checked=False,
            guardrail_allowed=True,
            guardrail_reason="",
            knowledge_base_done=False,
            structured_query_done=False,
            retrieval_intent=None,
            structured_query_intent=None,
            errors=[],
            hop_count=0,
            next_agent="",
            supervisor_reason="",
        )

    def _build_response(self, request: ConversationRequest, final_state: GraphState) -> ConversationResponse:
        metadata = ExecutionMetadata(
            guardrail_checked=final_state["guardrail_checked"],
            guardrail_allowed=final_state["guardrail_allowed"],
            knowledge_base_used=final_state["knowledge_base_done"],
            structured_query_used=final_state["structured_query_done"],
            agents_invoked=self._agents_invoked(final_state),
            errors=final_state.get("errors", []),
        )

        if final_state["guardrail_checked"] and not final_state["guardrail_allowed"]:
            return ConversationResponse(
                answer=final_state["guardrail_reason"] or "This request cannot be fulfilled.",
                refused=True,
                metadata=metadata,
            )

        return ConversationResponse(answer=self._synthesize(request, final_state), refused=False, metadata=metadata)

    @staticmethod
    def _synthesize(request: ConversationRequest, final_state: GraphState) -> str:
        try:
            with timed(logger, "llm_call", node="supervisor_synthesis"):
                chain = SUPERVISOR_SYNTHESIS_PROMPT | get_llm()
                result = chain.invoke(
                    {
                        "query": request.query,
                        "retrieval_intent": final_state.get("retrieval_intent") or "(none prepared)",
                        "structured_query_intent": final_state.get("structured_query_intent") or "(none prepared)",
                        "errors": final_state.get("errors") or "(none)",
                    }
                )
            return result.content
        except Exception as exc:  # noqa: BLE001 - synthesis failing must not hide that the workflow itself succeeded
            return f"The workflow completed, but the final synthesis failed: {exc}"

    @staticmethod
    def _agents_invoked(final_state: GraphState) -> list[str]:
        invoked = ["supervisor"]
        if final_state["guardrail_checked"]:
            invoked.append("guardrail")
        if final_state["knowledge_base_done"]:
            invoked.append("knowledge_base")
        if final_state["structured_query_done"]:
            invoked.append("structured_query")
        return invoked


@lru_cache(maxsize=1)
def get_orchestrator() -> ConversationOrchestrator:
    """Convenience accessor for callers that just want a shared, lazily-built instance."""
    return ConversationOrchestrator()
