"""
AI Travel Planner — Streamlit Chat Interface

A multi-agent travel assistant powered by LangGraph + Gemini.
Agents: Flight booking, Hotel search, Itinerary planning.
Built by Sekhar Indupuri.
"""

import uuid
import traceback

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from auth.login import check_password
from agents.graph import build_graph, set_tracker
from usage import UsageTracker


def extract_text(message) -> str:
    """Extract plain text from a message, handling various content formats."""
    if not hasattr(message, "content"):
        return str(message)
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content)
    return str(content)


# ---- Page config ----
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="centered",
)

# ---- Auth gate ----
if not check_password():
    st.stop()

# ---- Session state init ----
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "usage_tracker" not in st.session_state:
    st.session_state.usage_tracker = UsageTracker()
if "graph" not in st.session_state:
    try:
        st.session_state.graph = build_graph()
    except Exception as e:
        st.error(f"Failed to initialize the travel planner: {e}")
        st.code(traceback.format_exc())
        st.stop()

graph = st.session_state.graph
tracker = st.session_state.usage_tracker

# ---- Header ----
st.title("✈️ AI Travel Planner")
st.caption(
    "Ask me to **search flights**, **find hotels**, or **plan an itinerary**. "
    "I'll route your request to the right specialist agent."
)

# ---- Sidebar ----
with st.sidebar:
    st.header("About")
    st.markdown(
        """
This app uses **3 specialized AI agents** coordinated by a router:

| Agent | Capability |
|-------|-----------|
| ✈️ Flight | Real-time flight search |
| 🏨 Hotel | Hotel search & comparison |
| 🗺️ Itinerary | Trip planning & research |

**Powered by:** LangGraph, Gemini, SerpAPI, Tavily
"""
    )

    st.divider()

    # ---- Usage Dashboard ----
    st.header("📊 Usage (this session)")

    col1, col2 = st.columns(2)
    col1.metric("Total API Calls", tracker.total_calls)
    col2.metric("Est. Cost", f"${tracker.estimated_cost:.4f}")

    col3, col4, col5 = st.columns(3)
    col3.metric("Gemini", tracker.gemini_calls)
    col4.metric("SerpAPI", tracker.serpapi_calls)
    col5.metric("Tavily", tracker.tavily_calls)

    if tracker.api_calls:
        with st.expander("📋 Call Log", expanded=False):
            for call in reversed(tracker.api_calls[-20:]):
                st.text(f"{call.timestamp} | {call.service:8s} | {call.agent:10s} | {call.detail}")

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.usage_tracker = UsageTracker()
        st.rerun()

    st.divider()
    st.caption(f"Session: `{st.session_state.thread_id[:8]}…`")

    # ---- Footer with author name ----
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.85em;">
            Built by <strong>Sekhar Indupuri</strong><br>
            Multi-Agent System • 2026
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Chat history display ----
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(extract_text(msg))

# ---- Chat input ----
if prompt := st.chat_input("Where would you like to go?"):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                # Set the tracker via module-level ref (avoids serialization)
                set_tracker(tracker)

                result = graph.invoke(
                    {
                        "messages": list(st.session_state.messages),
                        "next_agent": None,
                    },
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id,
                        }
                    },
                )

                response_messages = result.get("messages", [])
                ai_response = None
                for m in reversed(response_messages):
                    if isinstance(m, AIMessage):
                        ai_response = m
                        break

                if ai_response:
                    text = extract_text(ai_response)
                    if text:
                        st.markdown(text)
                        st.session_state.messages.append(AIMessage(content=text))
                    else:
                        st.warning("The agent returned an empty response. Please try again.")
                else:
                    for m in reversed(response_messages):
                        text = extract_text(m)
                        if text and not isinstance(m, HumanMessage):
                            st.markdown(text)
                            st.session_state.messages.append(AIMessage(content=text))
                            break
                    else:
                        st.warning("No response generated. Please try rephrasing your question.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.code(traceback.format_exc())
