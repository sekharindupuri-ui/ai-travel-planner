"""Flight agent — extracts structured params from user query, then searches."""

from datetime import date
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from tools.flights import search_flights


class FlightParams(BaseModel):
    departure_airport: str
    arrival_airport: str
    outbound_date: str
    return_date: Optional[str] = None
    adults: int = 1
    children: int = 0


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract flight search parameters from the conversation.
Today's date is {today}. Use this to resolve relative dates and assume the correct year.
If the user says "March 8" without a year, assume the nearest future March 8 from today.

IMPORTANT: The user may be referring to details from earlier in the conversation.
Use the full conversation context to fill in any missing parameters.
For example, if the user previously searched for "Chicago to Denver" and now says
"how about returning on April 8th instead", use Chicago, Denver, and the original
outbound date from the earlier message.

Return ONLY a JSON object with these fields:
{{
  "departure_airport": "city name or IATA code",
  "arrival_airport": "city name or IATA code",
  "outbound_date": "YYYY-MM-DD",
  "return_date": "YYYY-MM-DD or null if one-way",
  "adults": 1,
  "children": 0
}}
All dates MUST be in YYYY-MM-DD format. No extra text — just the JSON.""",
        ),
        ("human", "{query}"),
    ]
)


def build_flight_chain(llm):
    parser = PydanticOutputParser(pydantic_object=FlightParams)
    return EXTRACTION_PROMPT | llm | parser


def _build_context_query(messages: list, current_query: str) -> str:
    """Build a context-rich query from conversation history."""
    history_lines = []
    for msg in messages[:-1]:  # exclude the current message
        if isinstance(msg, HumanMessage):
            history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Keep only the first 200 chars of AI responses to save tokens
            history_lines.append(f"Assistant: {content[:200]}")

    if history_lines:
        history = "\n".join(history_lines[-6:])  # last 3 exchanges max
        return f"Previous conversation:\n{history}\n\nCurrent request: {current_query}"
    return current_query


def run_flight_agent(llm, user_query: str, usage_tracker=None, messages: list = None) -> AIMessage:
    try:
        chain = build_flight_chain(llm)
        query = _build_context_query(messages, user_query) if messages else user_query
        params: FlightParams = chain.invoke({
            "query": query,
            "today": date.today().isoformat(),
        })
        if usage_tracker:
            usage_tracker.log_gemini("Flight", detail="param extraction")

        result = search_flights(
            departure_airport=params.departure_airport,
            arrival_airport=params.arrival_airport,
            outbound_date=params.outbound_date,
            return_date=params.return_date,
            adults=params.adults,
            children=params.children,
            usage_tracker=usage_tracker,
        )
    except Exception as e:
        result = f"Sorry, I couldn't process that flight request: {e}"

    return AIMessage(content=result)
