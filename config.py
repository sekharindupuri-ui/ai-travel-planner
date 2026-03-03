"""
Centralized configuration — reads from environment variables or Streamlit secrets.
On Streamlit Community Cloud, set these in the app's Secrets panel.
Locally, use a .env file or export them in your shell.
"""

import os
import streamlit as st


def get_secret(key: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets first, then fall back to env vars."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)


# --- API Keys ---
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
SERPAPI_API_KEY = get_secret("SERPAPI_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

# --- Auth ---
# A simple shared password that gates access to the app.
# For production, swap this out for a real user database.
APP_PASSWORD = get_secret("APP_PASSWORD", "travel2026")

# --- Model ---
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = float(get_secret("GEMINI_TEMPERATURE", "0.2"))
