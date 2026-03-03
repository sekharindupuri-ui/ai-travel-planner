"""Itinerary agent — plans trips using Tavily search + LLM reasoning."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools.search import run_tavily_search, tavily_search, get_tavily_tool_name

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
    tools = [tavily_search]
    llm_with_tools = llm.bind_tools(tools)
    return ITINERARY_PROMPT | llm_with_tools, tools


def run_itinerary_agent(llm, messages: list, usage_tracker=None) -> AIMessage:
    agent_chain, tools = build_itinerary_agent(llm)

    try:
        response = agent_chain.invoke({"messages": messages})
        if usage_tracker:
            usage_tracker.log_gemini("Itinerary", detail="initial response")
    except Exception as e:
        return AIMessage(content=f"I encountered an error while planning: {e}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_messages = []
        tavily_name = get_tavily_tool_name()

        for tc in response.tool_calls:
            if tc["name"] == tavily_name:
                try:
                    query = tc["args"].get("query", tc["args"].get("input", ""))
                    result = run_tavily_search(query, usage_tracker=usage_tracker)
                except Exception as e:
                    result = f"Search failed: {e}"
                tool_messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

        if tool_messages:
            full_messages = messages + [response] + tool_messages
            try:
                response = agent_chain.invoke({"messages": full_messages})
                if usage_tracker:
                    usage_tracker.log_gemini("Itinerary", detail="with search results")
            except Exception as e:
                return AIMessage(content=f"Error processing search results: {e}")

    if isinstance(response, AIMessage):
        return response
    return AIMessage(content=str(response.content) if hasattr(response, "content") else str(response))
