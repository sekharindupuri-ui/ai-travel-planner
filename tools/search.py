"""Tavily web search tool for the itinerary agent."""

import json
import os

from config import TAVILY_API_KEY

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

try:
    from langchain_tavily import TavilySearch
    tavily_search = TavilySearch(max_results=3)
except Exception:
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily_search = TavilySearchResults(max_results=3)


def get_tavily_tool_name() -> str:
    return tavily_search.name


def run_tavily_search(query: str, usage_tracker=None) -> str:
    try:
        results = tavily_search.invoke(query)
        if usage_tracker:
            usage_tracker.log_tavily("Itinerary", detail=query[:50])
        if isinstance(results, str):
            return results
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Search error: {e}"
