"""Flight search tool using Google Flights via SerpAPI."""

import json
import os
from typing import Optional

from serpapi.google_search import GoogleSearch

from config import SERPAPI_API_KEY

# ---- Airport code mapping ----
AIRPORT_CODES: dict[str, str] = {
    "boston": "BOS",
    "san francisco": "SFO",
    "sf": "SFO",
    "new york": "JFK",
    "nyc": "JFK",
    "newark": "EWR",
    "los angeles": "LAX",
    "la": "LAX",
    "chicago": "ORD",
    "seattle": "SEA",
    "paris": "CDG",
    "london": "LHR",
    "delhi": "DEL",
    "new delhi": "DEL",
    "tokyo": "NRT",
    "singapore": "SIN",
    "munich": "MUC",
    "nagpur": "NAG",
    "dubai": "DXB",
    "indianapolis": "IND",
    "miami": "MIA",
    "atlanta": "ATL",
    "denver": "DEN",
    "dallas": "DFW",
    "houston": "IAH",
    "washington": "IAD",
    "dc": "IAD",
    "toronto": "YYZ",
    "bangkok": "BKK",
    "sydney": "SYD",
    "rome": "FCO",
    "barcelona": "BCN",
    "amsterdam": "AMS",
    "frankfurt": "FRA",
    "hong kong": "HKG",
    "mumbai": "BOM",
    "bangalore": "BLR",
}


def normalize_airport(name: str) -> str:
    if not name:
        return name
    cleaned = name.strip().lower()
    return AIRPORT_CODES.get(cleaned, name.strip().upper())


def normalize_date(date_str: str) -> str:
    if not date_str:
        return date_str
    try:
        from dateutil import parser as date_parser
        return date_parser.parse(date_str).strftime("%Y-%m-%d")
    except Exception:
        return date_str


def search_flights(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    usage_tracker=None,
) -> str:
    dep = normalize_airport(departure_airport)
    arr = normalize_airport(arrival_airport)
    out_date = normalize_date(outbound_date)
    ret_date = normalize_date(return_date) if return_date else None

    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google_flights",
        "departure_id": dep,
        "arrival_id": arr,
        "outbound_date": out_date,
        "currency": "USD",
        "adults": adults,
        "children": children,
    }
    if ret_date:
        params["return_date"] = ret_date

    try:
        results = GoogleSearch(params).get_dict()
        if usage_tracker:
            usage_tracker.log_serpapi("Flight", detail=f"{dep}→{arr} {out_date}")

        flights = results.get("best_flights") or results.get("other_flights", [])
        if not flights:
            return f"No flights found from {dep} to {arr} on {out_date}."

        lines = []
        for f in flights[:10]:
            leg = f["flights"][0]
            airline = leg.get("airline", "Unknown")
            dep_name = leg["departure_airport"]["name"]
            arr_name = leg["arrival_airport"]["name"]
            dep_time = leg["departure_airport"]["time"]
            arr_time = leg["arrival_airport"]["time"]
            duration = f.get("total_duration", leg.get("duration", "?"))
            price = f.get("price", "N/A")
            travel_class = leg.get("travel_class", "Economy")
            stops = len(f.get("flights", [])) - 1
            stop_label = "Direct" if stops == 0 else f"{stops} stop(s)"
            lines.append(
                f"**{airline}** ({travel_class}) — **${price}** | {duration} min | {stop_label}\n"
                f"  {dep_name} ({dep_time}) → {arr_name} ({arr_time})"
            )

        header = f"✈️ Flights from **{dep}** to **{arr}** on {out_date}"
        if ret_date:
            header += f" (return {ret_date})"
        return header + "\n\n" + "\n\n".join(lines)

    except Exception as e:
        return f"Flight search error: {e}"
