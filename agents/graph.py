"""LangGraph orchestration — wires together the router and all agents."""

import os
from typing import Annotated, List, Literal, Optional, TypedDict

import operator
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from config import GEMINI_MODEL, GEMINI_TEMPERATURE, GOOGLE_API_KEY, SERPAPI_API_KEY, TAVILY_API_KEY
from agents.router import build_router_chain, resolve_route
from agents.flight import run_flight_agent
from agents.hotel import run_hotel_agent
from agents.itinerary import run_itinerary_agent

# Ensure all env vars are set for SDKs that read them directly
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# ---- State schema ----
class TravelState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: Optional[str]


# ---- Shared LLM instance ----
def get_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
    )


# ---- Node functions ----
def router_node(state: TravelState) -> dict:
    """Classify the query and decide which agent handles it."""
    llm = get_llm()
    user_msg = state["messages"][-1].content
    chain = build_router_chain(llm)
    try:
        decision = chain.invoke({"query": user_msg})
        next_agent = resolve_route(decision)
    except Exception as e:
        print(f"Router error: {e}")
        next_agent = "itinerary_agent"
    return {"next_agent": next_agent}


def flight_node(state: TravelState) -> dict:
    llm = get_llm()
    user_query = state["messages"][-1].content
    response = run_flight_agent(llm, user_query)
    return {"messages": [response]}


def hotel_node(state: TravelState) -> dict:
    llm = get_llm()
    user_query = state["messages"][-1].content
    response = run_hotel_agent(llm, user_query)
    return {"messages": [response]}


def itinerary_node(state: TravelState) -> dict:
    llm = get_llm()
    response = run_itinerary_agent(llm, state["messages"])
    return {"messages": [response]}


# ---- Conditional edge ----
def pick_agent(state: TravelState) -> Literal["flight_agent", "hotel_agent", "itinerary_agent"]:
    return state.get("next_agent", "itinerary_agent")


# ---- Build graph ----
def build_graph():
    """Construct and compile the LangGraph travel planner."""
    wf = StateGraph(TravelState)

    wf.add_node("router", router_node)
    wf.add_node("flight_agent", flight_node)
    wf.add_node("hotel_agent", hotel_node)
    wf.add_node("itinerary_agent", itinerary_node)

    wf.set_entry_point("router")
    wf.add_conditional_edges(
        "router",
        pick_agent,
        {
            "flight_agent": "flight_agent",
            "hotel_agent": "hotel_agent",
            "itinerary_agent": "itinerary_agent",
        },
    )
    wf.add_edge("flight_agent", END)
    wf.add_edge("hotel_agent", END)
    wf.add_edge("itinerary_agent", END)

    checkpointer = InMemorySaver()
    return wf.compile(checkpointer=checkpointer)
