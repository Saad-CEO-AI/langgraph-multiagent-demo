from __future__ import annotations

from langchain_core.messages import AIMessage

from src.agents.schemas import CritiqueVerdict

_CALL_STATE = {"critic_calls": 0}


def _field_after(messages: list, prefix: str) -> str:
    for message in messages:
        content = str(message.content)
        if content.startswith(prefix):
            return content[len(prefix) :].strip()
    return ""


class _MockToolBoundModel:
    def invoke(self, messages: list) -> AIMessage:
        query = _field_after(messages, "Question:")
        critique = _field_after(messages, "Previous critique to address (empty if this is the first draft):")

        if critique:
            content = (
                f"(revised draft) Answer to '{query}': incorporating the critique -- {critique} "
                "Here is a more complete answer that addresses that feedback directly."
            )
        else:
            content = f"(draft) Answer to '{query}': here is an initial answer covering the key point."

        return AIMessage(content=content, tool_calls=[])


class _MockStructuredModel:
    def invoke(self, messages: list) -> CritiqueVerdict:
        _CALL_STATE["critic_calls"] += 1
        if _CALL_STATE["critic_calls"] % 2 == 1:
            return CritiqueVerdict(
                verdict="needs_revision",
                critique="Add more supporting detail and be more specific.",
            )
        return CritiqueVerdict(verdict="approved", critique="")


class MockChatModel:
    """Offline stand-in for a real chat model -- no API key, no network call.

    Selected via LLM_PROVIDER=mock (or the CLI's --mock flag). Deterministic:
    always needs one revision before approving, so the loop is visibly
    exercised on every run.
    """

    def bind_tools(self, tools: list) -> _MockToolBoundModel:
        return _MockToolBoundModel()

    def with_structured_output(self, schema: type) -> _MockStructuredModel:
        return _MockStructuredModel()
