from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the web for current or factual information needed to answer the question.",
)
