from __future__ import annotations

import os
from functools import lru_cache

from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


REQUEST_TIMEOUT_SECONDS = 30.0


@lru_cache(maxsize=1)
def get_client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=3, timeout=REQUEST_TIMEOUT_SECONDS)
