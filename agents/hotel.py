"""Hotel agent — extracts structured params from user query, then searches."""

from datetime import date
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from tools.hotels import search_hotels


class HotelParams(BaseModel):
    location: str
    check_in_date: str
    check_out_date: str
    adults: int = 1
    children: int = 0
    rooms: int = 1
    hotel_class: Optional[str] = None


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract hotel search parameters from the conversation.

CRITICAL DATE RULE: Today is {today} (year {year}). The current year is {year}.
- If a user mentions a month/day WITHOUT a year, use {year} as the year.
- "April 8th" means {year}-04-08. NOT 2027, NOT 2025. Always {year}.
- Only use a future year if the date has already passed in {year}.

Use the full conversation context to fill in missing parameters.
For example, if the user previously searched for flights to Denver and now says
"find me a hotel there", use Denver as the location and infer dates from context.

Return ONLY a JSON object:
{{
  "location": "city or area name",
  "check_in_date": "YYYY-MM-DD",
  "check_out_date": "YYYY-MM-DD",
  "adults": 1,
  "children": 0,
  "rooms": 1,
  "hotel_class": null
}}
All dates MUST use year {year} unless the date has already passed.
For hotel_class, use comma-separated star ratings if mentioned (e.g., "3,4,5").
No extra text — just the JSON.""",
        ),
        ("human", "{query}"),
    ]
)


def build_hotel_chain(llm):
    parser = PydanticOutputParser(pydantic_object=HotelParams)
    return EXTRACTION_PROMPT | llm | parser


def _build_context_query(messages: list, current_query: str) -> str:
    """Build a context-rich query from conversation history."""
    history_lines = []
    for msg in messages[:-1]:
        if isinstance(msg, HumanMessage):
            history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            history_lines.append(f"Assistant: {content[:200]}")

    if history_lines:
        history = "\n".join(history_lines[-6:])
        return f"Previous conversation:\n{history}\n\nCurrent request: {current_query}"
    return current_query


def run_hotel_agent(llm, user_query: str, usage_tracker=None, messages: list = None) -> AIMessage:
    try:
        chain = build_hotel_chain(llm)
        query = _build_context_query(messages, user_query) if messages else user_query
        today = date.today()
        params: HotelParams = chain.invoke({
            "query": query,
            "today": today.isoformat(),
            "year": str(today.year),
        })
        if usage_tracker:
            usage_tracker.log_gemini("Hotel", detail="param extraction")

        result = search_hotels(
            location=params.location,
            check_in_date=params.check_in_date,
            check_out_date=params.check_out_date,
            adults=params.adults,
            children=params.children,
            rooms=params.rooms,
            hotel_class=params.hotel_class,
            usage_tracker=usage_tracker,
        )
    except Exception as e:
        result = f"Sorry, I couldn't process that hotel request: {e}"

    return AIMessage(content=result)
