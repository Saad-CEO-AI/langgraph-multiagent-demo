from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.nodes import critic_node, finalizer_node, researcher_node
from src.graph.routing import route_after_critique
from src.graph.state import GraphState


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("researcher", researcher_node)
    builder.add_node("critic", critic_node)
    builder.add_node("finalizer", finalizer_node)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critique,
        {"revise": "researcher", "finalize": "finalizer"},
    )
    builder.add_edge("finalizer", END)

    return builder.compile()
