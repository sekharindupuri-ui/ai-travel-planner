"""Itinerary agent — plans trips using Tavily search + LLM reasoning."""

import json

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from tools.search import run_tavily_search, tavily_search

ITINERARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert travel itinerary planner.

RULES:
- Only respond to travel-related questions.
- If asked about non-travel topics, politely redirect.
- Use the search tool to find current info about destinations.
- Provide detailed, well-structured itineraries with timings, transport, costs, and tips.

Use the ReAct approach:
1. THOUGHT: What travel info do I need?
2. ACTION: Search for current details
3. OBSERVATION: Process results
4. Provide a comprehensive response""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def build_itinerary_agent(llm):
    """Create the itinerary agent chain with Tavily tool bound."""
    tools = [tavily_search]
    llm_with_tools = llm.bind_tools(tools)
    return ITINERARY_PROMPT | llm_with_tools, tools


def run_itinerary_agent(llm, messages: list) -> AIMessage:
    """Run the itinerary agent, handling tool calls if needed."""
    agent_chain, tools = build_itinerary_agent(llm)

    response = agent_chain.invoke({"messages": messages})

    # Handle tool calls (ReAct loop — one round)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_messages = []
        for tc in response.tool_calls:
            if tc["name"] == "tavily_search_results_json":
                try:
                    result = run_tavily_search(tc["args"].get("query", ""))
                except Exception as e:
                    result = f"Search failed: {e}"
                tool_messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

        if tool_messages:
            full_messages = messages + [response] + tool_messages
            response = agent_chain.invoke({"messages": full_messages})

    return response
