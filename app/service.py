from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Iterator

from groq import Groq

from app.prompts import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_USER_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCHER_USER_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    SUPERVISOR_USER_PROMPT,
)
from app.schemas import CritiqueVerdict
from app.tools import WEB_SEARCH_TOOL_SCHEMA, web_search

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_REVISIONS = 2
MAX_TOOL_ROUNDS = 3


@lru_cache(maxsize=1)
def _client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=3)


def _run_researcher(query: str, critique: str) -> str:
    """Researcher agent: drafts an answer, searching the web when it needs to."""
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": RESEARCHER_USER_PROMPT.format(query=query, critique=critique or "(none)")},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        message = _client().chat.completions.create(
            model=MODEL, messages=messages, tools=[WEB_SEARCH_TOOL_SCHEMA], temperature=0.0
        ).choices[0].message

        if not message.tool_calls:
            return message.content or ""

        messages.append(
            {"role": "assistant", "content": message.content, "tool_calls": [tc.model_dump() for tc in message.tool_calls]}
        )
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = web_search(args.get("query", query))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return _forced_text_answer(query, critique)


def _forced_text_answer(query: str, critique: str) -> str:
    """Tool-call budget exhausted; ask fresh, from a clean prompt, for a plain-text answer.

    Continuing the tool-call-laden conversation here can still prime this
    model into emitting another tool call even with none declared, which the
    API rejects outright -- so this asks fresh instead of reusing that
    history, and never returns an empty draft even if this also fails.
    """
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": RESEARCHER_USER_PROMPT.format(query=query, critique=critique or "(none)")},
        {"role": "user", "content": "Answer now in plain text using what you already know. Do not call any tools."},
    ]
    try:
        message = _client().chat.completions.create(model=MODEL, messages=messages, temperature=0.0).choices[0].message
        return message.content or ""
    except Exception:  # noqa: BLE001 - last-resort boundary: guarantee non-empty text, never propagate here
        return "Unable to produce a complete answer after repeated search attempts."


def _run_critic(query: str, draft: str) -> CritiqueVerdict:
    """Critic agent: grades the draft and returns a typed verdict, not free text."""
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": CRITIC_USER_PROMPT.format(query=query, draft=draft)},
    ]
    content = _client().chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0, response_format={"type": "json_object"}
    ).choices[0].message.content

    try:
        return CritiqueVerdict.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValueError):
        return CritiqueVerdict(verdict="needs_revision", critique="Critic returned a malformed verdict; revise and retry.")


def _stream_final_answer(query: str, draft: str) -> Iterator[str]:
    """Supervisor's own call: synthesises the approved draft into the streamed reply."""
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": SUPERVISOR_USER_PROMPT.format(query=query, draft=draft)},
    ]
    stream = _client().chat.completions.create(model=MODEL, messages=messages, temperature=0.0, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def run_supervisor(query: str) -> Iterator[str]:
    """Orchestrates Researcher and Critic, then streams the Supervisor's final synthesis."""
    draft, critique = "", ""

    for revision_count in range(1, MAX_REVISIONS + 1):
        draft = _run_researcher(query, critique)
        verdict = _run_critic(query, draft)
        if verdict.verdict == "approved" or revision_count == MAX_REVISIONS:
            break
        critique = verdict.critique

    yield from _stream_final_answer(query, draft)
