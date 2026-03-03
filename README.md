# ✈️ AI Travel Planner — Multi-Agent System

A multi-agent travel assistant built with **LangGraph**, **Streamlit**, and **Google Gemini**.
Three specialist agents handle flights, hotels, and itinerary planning, coordinated by an LLM-powered router.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.38+-red)
![LangGraph](https://img.shields.io/badge/langgraph-0.2+-green)

## Architecture

```
User ──▶ Streamlit Chat UI ──▶ LangGraph State Machine
                                      │
                                  ┌────┴────┐
                                  │  Router  │  (Gemini classifies query)
                                  └────┬────┘
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                    ┌──────────┐ ┌──────────┐ ┌───────────┐
                    │ Flight   │ │  Hotel   │ │ Itinerary │
                    │  Agent   │ │  Agent   │ │   Agent   │
                    └────┬─────┘ └────┬─────┘ └─────┬─────┘
                         │            │             │
                    Google Flights  Google Hotels  Tavily
                     (SerpAPI)      (SerpAPI)     Search
```

## Features

- **Intelligent routing** — LLM classifies each query and dispatches to the right agent
- **Flight search** — Real-time results via Google Flights / SerpAPI
- **Hotel search** — Prices, ratings, and amenities via Google Hotels / SerpAPI
- **Itinerary planning** — Web-researched trip plans via Tavily + Gemini
- **Password protection** — Simple auth gate for public deployment
- **Conversation memory** — Multi-turn chat within a session

## Quick Start (Local)

```bash
git clone https://github.com/YOUR_USERNAME/ai-travel-planner.git
cd ai-travel-planner
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-google-ai-studio-key"
SERPAPI_API_KEY = "your-serpapi-key"
TAVILY_API_KEY = "your-tavily-key"
APP_PASSWORD = "your-password"
```

Run:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

1. **Push this repo to GitHub** (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app.py`
4. Open **Advanced settings** → **Secrets** and paste:

```toml
GOOGLE_API_KEY = "..."
SERPAPI_API_KEY = "..."
TAVILY_API_KEY = "..."
APP_PASSWORD = "a-strong-password"
```

5. Click **Deploy** — your app will be live at `https://your-app.streamlit.app`

## API Keys (All Have Free Tiers)

| Service | Free Tier | Sign Up |
|---------|-----------|---------|
| Google AI Studio (Gemini) | Generous free tier | [aistudio.google.com](https://aistudio.google.com/apikey) |
| SerpAPI | 100 searches/month | [serpapi.com](https://serpapi.com) |
| Tavily | 1,000 searches/month | [tavily.com](https://tavily.com) |

## Project Structure

```
ai-travel-planner/
├── app.py                 # Streamlit UI + chat loop
├── config.py              # Secrets + settings
├── requirements.txt
├── agents/
│   ├── graph.py           # LangGraph state machine
│   ├── router.py          # Query classifier
│   ├── flight.py          # Flight agent
│   ├── hotel.py           # Hotel agent
│   └── itinerary.py       # Itinerary agent
├── tools/
│   ├── flights.py         # SerpAPI flight search
│   ├── hotels.py          # SerpAPI hotel search
│   └── search.py          # Tavily web search
├── auth/
│   └── login.py           # Password gate
└── .streamlit/
    └── config.toml        # Theme + server settings
```

## Example Queries

- "Find flights from Chicago to Indianapolis on March 8, returning March 12"
- "Search for 4-star hotels in Tokyo for March 15-20 for 2 adults"
- "Plan a 5-day itinerary for Rome with family-friendly activities"
- "What's the best time to visit Bali and what should I see?"

## How It Differs from the Colab Version

| Aspect | Colab Notebook | This Project |
|--------|---------------|--------------|
| Structure | Single file, flat | Modular packages |
| Auth | None | Password-gated |
| Deployment | Can't deploy | Streamlit Cloud ready |
| State | InMemorySaver with reused thread IDs | Per-session thread IDs |
| Error handling | Minimal try/except | Structured error boundaries |
| Message types | Mixed HumanMessage/AIMessage | Correct AIMessage for agents |
| Tool shadowing | `tool` variable overwritten | Clean naming |

## License

MIT
