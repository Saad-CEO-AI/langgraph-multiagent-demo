from __future__ import annotations

import os
from functools import lru_cache

from langchain_groq import ChatGroq

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
REQUEST_TIMEOUT_SECONDS = 30.0


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return ChatGroq(
        model_name=MODEL,
        temperature=0.0,
        request_timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=3,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )
