"""Usage tracker — monitors API calls, tokens, and estimated costs per session."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


@dataclass
class APICall:
    timestamp: str
    agent: str
    service: str  # "gemini", "serpapi", "tavily"
    detail: str = ""


@dataclass
class UsageTracker:
    """Tracks API usage across a session."""

    api_calls: List[APICall] = field(default_factory=list)
    gemini_calls: int = 0
    serpapi_calls: int = 0
    tavily_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def log_gemini(self, agent: str, input_tokens: int = 0, output_tokens: int = 0, detail: str = ""):
        self.gemini_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.api_calls.append(APICall(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            agent=agent,
            service="Gemini",
            detail=detail or f"in={input_tokens}, out={output_tokens}",
        ))

    def log_serpapi(self, agent: str, detail: str = ""):
        self.serpapi_calls += 1
        self.api_calls.append(APICall(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            agent=agent,
            service="SerpAPI",
            detail=detail,
        ))

    def log_tavily(self, agent: str, detail: str = ""):
        self.tavily_calls += 1
        self.api_calls.append(APICall(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            agent=agent,
            service="Tavily",
            detail=detail,
        ))

    @property
    def total_calls(self) -> int:
        return self.gemini_calls + self.serpapi_calls + self.tavily_calls

    @property
    def estimated_cost(self) -> float:
        """Rough cost estimate in USD. Gemini Flash is very cheap; SerpAPI is the main cost."""
        # Gemini 2.5 Flash: ~$0.15/1M input, ~$0.60/1M output (approximate)
        gemini_cost = (self.total_input_tokens * 0.00000015) + (self.total_output_tokens * 0.0000006)
        # SerpAPI: ~$0.01 per search on paid plan (free tier = 100/month)
        serpapi_cost = self.serpapi_calls * 0.01
        # Tavily: free tier 1000/month, then ~$0.01 per search
        tavily_cost = self.tavily_calls * 0.01
        return gemini_cost + serpapi_cost + tavily_cost

    def summary(self) -> Dict:
        return {
            "total_calls": self.total_calls,
            "gemini_calls": self.gemini_calls,
            "serpapi_calls": self.serpapi_calls,
            "tavily_calls": self.tavily_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost, 4),
        }
