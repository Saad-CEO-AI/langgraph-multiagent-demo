from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SUPERVISOR_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Supervisor, coordinating this workflow. Guardrail has
already cleared this request -- you are deciding what happens next.

- If the question needs factual, documentary, or knowledge-base-style
  reasoning, and that has not been prepared yet, choose "knowledge_base".
- If the question is data-oriented -- counts, filters, specific records,
  threads -- and that has not been prepared yet, choose "structured_query".
- If both kinds of reasoning are relevant, choose whichever hasn't run yet;
  you will be asked again once it completes.
- If nothing further is needed (including if everything relevant has
  already run), choose "finish".

Respond with a routing decision matching the required schema.""",
        ),
        (
            "human",
            "Request: {query}\n\n"
            "Knowledge base already prepared: {knowledge_base_done}\n"
            "Structured query already prepared: {structured_query_done}",
        ),
    ]
)

SUPERVISOR_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Supervisor, presenting the outcome of this workflow to the
user. No real retrieval or database call has been made yet -- Knowledge Base
and Structured Query only prepare typed intent for a future integration.

Describe honestly what the workflow determined and what it would do next,
using the prepared intent below. Do not invent facts or present this as a
real, data-backed answer -- there is no real data yet. If something failed,
say so plainly rather than glossing over it.""",
        ),
        (
            "human",
            "Request: {query}\n\n"
            "Retrieval intent prepared (if any): {retrieval_intent}\n\n"
            "Structured query intent prepared (if any): {structured_query_intent}\n\n"
            "Errors encountered (if any): {errors}\n\n"
            "Present the outcome now.",
        ),
    ]
)
