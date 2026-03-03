"""Router agent — classifies user queries and picks the right specialist."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Classify the user's travel query into exactly ONE category.

FLIGHT — flight bookings, airlines, airports, tickets, air travel, flight prices
HOTEL — hotels, accommodations, stays, rooms, lodging, resorts, hotel prices
ITINERARY — trip planning, destinations, activities, attractions, sightseeing, general travel advice

Respond with a single word: FLIGHT, HOTEL, or ITINERARY""",
        ),
        ("human", "{query}"),
    ]
)

AGENT_MAP = {
    "FLIGHT": "flight_agent",
    "HOTEL": "hotel_agent",
    "ITINERARY": "itinerary_agent",
}


def build_router_chain(llm):
    """Return a chain that emits a routing decision string."""
    return ROUTER_PROMPT | llm | StrOutputParser()


def resolve_route(decision: str) -> str:
    """Map the LLM's one-word answer to a graph node name."""
    cleaned = decision.strip().upper().split()[0] if decision else ""
    return AGENT_MAP.get(cleaned, "itinerary_agent")
