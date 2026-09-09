from __future__ import annotations

import os
from typing import Any


def get_llm(temperature: float = 0.0) -> Any:
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "mock":
        from src.agents.mock_llm import MockChatModel

        return MockChatModel()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "groq":
        from langchain_openai import ChatOpenAI

        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider!r} (expected 'openai', 'anthropic', 'groq', or 'mock')"
    )
