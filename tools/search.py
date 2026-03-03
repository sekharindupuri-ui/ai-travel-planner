"""Tavily web search tool for the itinerary agent."""

import json
import os

from config import TAVILY_API_KEY

# Ensure env var is set for the SDK
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

try:
    from langchain_tavily import TavilySearch
    tavily_search = TavilySearch(max_results=3)
except Exception:
    from langchain_community.tools.tavily_search import TavilySearchResults
    tavily_search = TavilySearchResults(max_results=3)


def get_tavily_tool_name() -> str:
    """Get the actual registered tool name, which varies by library version."""
    return tavily_search.name


def run_tavily_search(query: str) -> str:
    """Run a Tavily web search and return formatted results."""
    try:
        results = tavily_search.invoke(query)
        if isinstance(results, str):
            return results
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Search error: {e}"
