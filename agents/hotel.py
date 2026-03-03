"""Hotel agent — extracts structured params from user query, then searches."""

from datetime import date
from typing import Optional

from langchain_core.messages import AIMessage
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
            """Extract hotel search parameters from the user message.
Today's date is {today}. Use this to resolve relative dates and assume the correct year.
If the user says "March 8" without a year, assume the nearest future March 8 from today.

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
All dates MUST be in YYYY-MM-DD format.
For hotel_class, use comma-separated star ratings if mentioned (e.g., "3,4,5").
No extra text — just the JSON.""",
        ),
        ("human", "{query}"),
    ]
)


def build_hotel_chain(llm):
    parser = PydanticOutputParser(pydantic_object=HotelParams)
    return EXTRACTION_PROMPT | llm | parser


def run_hotel_agent(llm, user_query: str, usage_tracker=None) -> AIMessage:
    try:
        chain = build_hotel_chain(llm)
        params: HotelParams = chain.invoke({
            "query": user_query,
            "today": date.today().isoformat(),
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
