"""
AI Travel Planner — Streamlit Chat Interface

A multi-agent travel assistant powered by LangGraph + Gemini.
Agents: Flight booking, Hotel search, Itinerary planning.
"""

import uuid
import traceback

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from auth.login import check_password
from agents.graph import build_graph

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
if "graph" not in st.session_state:
    try:
        st.session_state.graph = build_graph()
    except Exception as e:
        st.error(f"Failed to initialize the travel planner: {e}")
        st.code(traceback.format_exc())
        st.stop()

graph = st.session_state.graph

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

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Session: `{st.session_state.thread_id[:8]}…`")

# ---- Chat history display ----
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ---- Chat input ----
if prompt := st.chat_input("Where would you like to go?"):
    # Show user message
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = graph.invoke(
                    {
                        "messages": [user_msg],
                        "next_agent": None,
                    },
                    config={
                        "configurable": {
                            "thread_id": st.session_state.thread_id,
                        }
                    },
                )

                # Get the last AI message from the result
                response_messages = result.get("messages", [])
                ai_response = None
                for m in reversed(response_messages):
                    if isinstance(m, AIMessage):
                        ai_response = m
                        break

                if ai_response and ai_response.content:
                    st.markdown(ai_response.content)
                    st.session_state.messages.append(ai_response)
                else:
                    # Debug: show what we actually got back
                    msg_types = [type(m).__name__ for m in response_messages]
                    st.warning(
                        f"No response text generated. "
                        f"Messages returned: {len(response_messages)} ({', '.join(msg_types)})"
                    )
                    # Try to show any content we got
                    for m in response_messages:
                        if hasattr(m, "content") and m.content:
                            st.markdown(m.content)
                            st.session_state.messages.append(
                                AIMessage(content=m.content)
                            )
                            break

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.code(traceback.format_exc())
