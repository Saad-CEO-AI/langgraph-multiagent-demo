from __future__ import annotations

from ddgs import DDGS

WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current or factual information needed to answer the question.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}


def web_search(query: str, max_results: int = 5) -> str:
    results = DDGS().text(query, max_results=max_results)
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)
