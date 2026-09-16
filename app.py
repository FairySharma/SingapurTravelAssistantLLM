"""
Singapore Travel Planning Assistant — Streamlit UI
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.travel_assistant import TravelAssistant


# Page configuration
st.set_page_config(
    page_title="Singapore Travel Assistant",
    page_icon="🦁",
    layout="centered",
)

# Initialise session state keys if they don't exist yet
for key, default in [
    ("assistant", None),
    ("ready", False),
    ("messages", []),
    ("meta", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# App header
st.title("🦁 Singapore Travel Assistant")
st.caption("Ask about attractions, itineraries, weather, or currency — all in one place.")
st.divider()


# Setup form — shown before the assistant is initialised
if not st.session_state.ready:
    with st.form("setup"):
        st.subheader("Setup")

        provider = st.radio(
            "AI Provider",
            ["Google Gemini (Free)", "OpenAI"],
            horizontal=True,
            help="Gemini free tier: 1,500 requests/day. Get key at aistudio.google.com/apikey",
        )
        is_google = provider.startswith("Google")

        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="AIza...  (free at aistudio.google.com/apikey)" if is_google else "sk-...",
        )

        submitted = st.form_submit_button("▶ Start", type="primary", use_container_width=True)

    if submitted:
        if not api_key:
            st.error("Please enter an API key.")
        else:
            model = "gemini-3.6-flash" if is_google else "gpt-4o-mini"
            with st.spinner("Starting up — building knowledge index and connecting tools…"):
                try:
                    a = TravelAssistant(
                        api_key=api_key,
                        model=model,
                        provider="google" if is_google else "openai",
                    )
                    status = a.initialize()
                    st.session_state.assistant = a
                    st.session_state.ready = True

                    # Show which components connected successfully
                    parts = []
                    if status.get("rag"):
                        parts.append("✅ Knowledge base")
                    if status.get("weather_mcp"):
                        parts.append("✅ Weather tool")
                    if status.get("currency_mcp"):
                        parts.append("✅ Currency tool")
                    st.success("Ready!  " + "  ·  ".join(parts))
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")

    st.info(
        "**Google Gemini is free** — get a key in 30 seconds at "
        "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)",
        icon="💡",
    )


# Chat interface — shown after the assistant is initialised
else:
    # Reset button in the top-right corner
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Reset", help="Clear conversation and start over"):
            st.session_state.assistant.reset_conversation()
            st.session_state.messages = []
            st.session_state.meta = []
            st.session_state.ready = False
            st.rerun()

    # Render previous messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🦁"):
            st.markdown(msg["content"])

            # Show source references under each assistant message
            if msg["role"] == "assistant":
                idx = i // 2
                if idx < len(st.session_state.meta):
                    m = st.session_state.meta[idx]
                    parts = []
                    for s in m.get("sources", []):
                        label = s["title"]
                        parts.append(f"[{label}]({s['url']})" if s.get("url") else label)
                    for t in m.get("mcp_tools_used", []):
                        parts.append(f"`{t}`")
                    if parts:
                        st.caption("📎 Sources: " + "  ·  ".join(parts))

    # Chat input box
    user_input = st.chat_input("Ask anything about Singapore travel…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🦁"):
            with st.spinner("Thinking…"):
                result = st.session_state.assistant.chat(user_input)
            st.markdown(result["response"])

            # Show sources for this response
            parts = []
            for s in result.get("sources", []):
                parts.append(f"[{s['title']}]({s['url']})" if s.get("url") else s["title"])
            for t in result.get("mcp_tools_used", []):
                parts.append(f"`{t}`")
            if parts:
                st.caption("📎 Sources: " + "  ·  ".join(parts))

        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
        st.session_state.meta.append(result)
        st.rerun()

    # Collapsible panel with example questions
    with st.expander("💡 Example questions"):
        examples = [
            "What are the must-visit attractions in Singapore?",
            "Create a 3-day sightseeing itinerary.",
            "What indoor attractions can I visit on a rainy day?",
            "Suggest activities for a family with young children.",
            "What is the current weather in Singapore?",
            "What is the forecast for the next 3 days?",
            "Convert INR 50,000 to SGD.",
            "How much is 200 SGD in USD?",
            "Create a 3-day itinerary adjusted for the weather forecast.",
            "I have a budget of INR 60,000 — convert to SGD and suggest a 3-day trip.",
        ]
        cols = st.columns(2)
        for i, q in enumerate(examples):
            if cols[i % 2].button(q, key=f"eg{i}", use_container_width=True):
                st.session_state["_eq"] = q
                st.rerun()

    # Process a question selected from the examples panel
    if "_eq" in st.session_state:
        q = st.session_state.pop("_eq")
        st.session_state.messages.append({"role": "user", "content": q})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(q)

        with st.chat_message("assistant", avatar="🦁"):
            with st.spinner("Thinking…"):
                result = st.session_state.assistant.chat(q)
            st.markdown(result["response"])

            parts = []
            for s in result.get("sources", []):
                parts.append(f"[{s['title']}]({s['url']})" if s.get("url") else s["title"])
            for t in result.get("mcp_tools_used", []):
                parts.append(f"`{t}`")
            if parts:
                st.caption("📎 Sources: " + "  ·  ".join(parts))

        st.session_state.messages.append({"role": "assistant", "content": result["response"]})
        st.session_state.meta.append(result)
        st.rerun()
