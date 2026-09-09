from __future__ import annotations

from langchain_core.messages import AIMessage

import src.graph.nodes as nodes_module
from src.agents.schemas import CritiqueVerdict
from src.graph.graph import build_graph
from src.graph.nodes import MAX_REVISIONS


class _FakeToolBoundModel:
    def invoke(self, messages):
        return AIMessage(content="Paris is the capital of France.", tool_calls=[])


class _FakeStructuredModel:
    def __init__(self, shared_state: dict):
        self._shared_state = shared_state

    def invoke(self, messages):
        self._shared_state["critic_calls"] += 1
        if self._shared_state["critic_calls"] == 1:
            return CritiqueVerdict(verdict="needs_revision", critique="Explain why it is the capital.")
        return CritiqueVerdict(verdict="approved", critique="")


class _FakeLLM:
    def __init__(self, shared_state: dict):
        self._shared_state = shared_state

    def bind_tools(self, tools):
        return _FakeToolBoundModel()

    def with_structured_output(self, schema):
        return _FakeStructuredModel(self._shared_state)


def test_graph_runs_end_to_end_and_exercises_the_revision_loop(monkeypatch):
    shared_state = {"critic_calls": 0}
    monkeypatch.setattr(nodes_module, "get_llm", lambda *a, **kw: _FakeLLM(shared_state))

    graph = build_graph()
    initial_state = {
        "query": "What is the capital of France?",
        "draft": "",
        "critique": "",
        "verdict": "needs_revision",
        "revision_count": 0,
        "final_answer": "",
    }

    final_state = graph.invoke(initial_state)

    assert final_state["final_answer"] == "Paris is the capital of France."
    assert final_state["revision_count"] <= MAX_REVISIONS
    assert shared_state["critic_calls"] == 2
