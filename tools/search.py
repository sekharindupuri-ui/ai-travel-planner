"""Tavily web search tool for the itinerary agent."""

import json
import os

from langchain_tavily import TavilySearch

from config import TAVILY_API_KEY

# Ensure env var is set for the SDK
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

tavily_search = TavilySearch(max_results=3)


def run_tavily_search(query: str) -> str:
    """Run a Tavily web search and return formatted results."""
    try:
        results = tavily_search.invoke(query)
        if isinstance(results, str):
            return results
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Search error: {e}"
