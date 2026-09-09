from __future__ import annotations

from src.graph.graph import build_graph


def test_graph_has_the_three_expected_nodes():
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())

    assert {"researcher", "critic", "finalizer"}.issubset(node_names)


def test_critic_has_a_conditional_edge_to_both_researcher_and_finalizer():
    graph = build_graph()
    edges = graph.get_graph().edges

    targets = {edge.target for edge in edges if edge.source == "critic"}

    assert targets == {"researcher", "finalizer"}


def test_finalizer_is_terminal():
    graph = build_graph()
    edges = graph.get_graph().edges

    sources_from_finalizer = {edge.source for edge in edges if edge.source == "finalizer"}
    targets_from_finalizer = {edge.target for edge in edges if edge.source == "finalizer"}

    assert sources_from_finalizer == {"finalizer"}
    assert targets_from_finalizer == {"__end__"}
