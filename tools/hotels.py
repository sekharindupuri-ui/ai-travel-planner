"""Hotel search tool using Google Hotels via SerpAPI."""

import json
from typing import Optional

from serpapi.google_search import GoogleSearch

from config import SERPAPI_API_KEY


def search_hotels(
    location: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    children: int = 0,
    rooms: int = 1,
    hotel_class: Optional[str] = None,
    sort_by: int = 8,
    usage_tracker=None,
) -> str:
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google_hotels",
        "hl": "en",
        "gl": "us",
        "q": location,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "currency": "USD",
        "adults": int(adults),
        "children": int(children),
        "rooms": int(rooms),
        "sort_by": int(sort_by),
    }
    if hotel_class:
        params["hotel_class"] = hotel_class

    try:
        results = GoogleSearch(params).get_dict()
        if usage_tracker:
            usage_tracker.log_serpapi("Hotel", detail=f"{location} {check_in_date}")

        properties = results.get("properties", [])
        if not properties:
            return f"No hotels found in {location} for those dates."

        lines = []
        for h in properties[:8]:
            name = h.get("name", "Unknown Hotel")
            rate_per_night = h.get("rate_per_night", {})
            price = rate_per_night.get("lowest", "N/A")
            total = h.get("total_rate", {}).get("lowest", "")
            rating = h.get("overall_rating", "N/A")
            reviews = h.get("reviews", "")
            stars = h.get("hotel_class", "")
            star_str = f" {'⭐' * stars}" if isinstance(stars, int) else ""
            amenities = ", ".join(h.get("amenities", [])[:5])

            line = f"**{name}**{star_str} — {price}/night"
            if total:
                line += f" ({total} total)"
            line += f" | Rating: {rating}"
            if reviews:
                line += f" ({reviews} reviews)"
            if amenities:
                line += f"\n  Amenities: {amenities}"
            lines.append(line)

        header = f"🏨 Hotels in **{location}** ({check_in_date} → {check_out_date})"
        return header + "\n\n" + "\n\n".join(lines)

    except Exception as e:
        return f"Hotel search error: {e}"
