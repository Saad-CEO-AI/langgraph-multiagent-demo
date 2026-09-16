from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, Union


class FakeMessage:
    """A scripted assistant message, for the tool-calling branch of a completion."""

    def __init__(self, content: str | None = None, tool_calls: list[dict] | None = None):
        self.content = content
        self.tool_calls = [SimpleNamespace(**tc) for tc in (tool_calls or [])]


ScriptItem = Union[str, FakeMessage, list[str]]


class FakeGroqClient:
    """Scripted stand-in for groq.Groq.

    Each call to chat.completions.create() consumes the next scripted item,
    in order: a plain string is a JSON/text response; a FakeMessage carries
    tool_calls; a list of strings is a streamed response, one chunk each.
    Raises if the script runs out, so a test can't silently pass on fewer
    calls than it actually expected.
    """

    def __init__(self, script: Iterable[ScriptItem]):
        self._script = list(script)
        self._index = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any):
        if self._index >= len(self._script):
            raise AssertionError(f"FakeGroqClient script exhausted after {self._index} call(s)")
        item = self._script[self._index]
        self._index += 1

        if kwargs.get("stream"):
            tokens = item if isinstance(item, list) else [item]
            return iter(SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=tok))]) for tok in tokens)

        message = item if isinstance(item, FakeMessage) else FakeMessage(content=item)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    @property
    def calls_made(self) -> int:
        return self._index
