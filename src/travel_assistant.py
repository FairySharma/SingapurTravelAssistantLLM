"""
Travel assistant core — orchestrates RAG retrieval, MCP tool calls, and LLM generation.
"""

import re
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.rag_engine import (
    get_or_build_vector_store,
    retrieve_relevant_chunks,
    format_retrieved_sources,
    get_source_references,
)
from src.mcp_client import MCPToolManager


# System prompt sent to the LLM on every request
SYSTEM_PROMPT = """You are a knowledgeable, helpful AI Travel Planning Assistant specialising in Singapore.

You have access to two types of information:
1. **Knowledge Base (RAG)** — curated travel documents about Singapore covering attractions, neighbourhoods, transportation, food, cultural tips, and sample itineraries.
2. **MCP Tools** — real-time data including current weather forecasts and live currency exchange rates.

## Core Behaviour Rules

### Accuracy and Attribution
- Use the provided KNOWLEDGE BASE EXCERPTS as your primary source for destination facts (attractions, transport, food, culture, tips, itineraries).
- Use MCP TOOL RESULTS for all current information: weather and currency conversion.
- Always attribute facts to their source. Use phrases like "According to the knowledge base..." or "Based on [Source Name]..." for RAG content, and "According to current weather data from Open-Meteo..." or "Based on live exchange rate data..." for MCP results.
- NEVER invent attractions, prices, transport details, or facts not present in the provided context.
- If the knowledge base does not contain sufficient information for a question, say so clearly: "I don't have detailed information about that in my knowledge base."

### Response Structure
- Structure responses clearly with headers (##) and bullet points where appropriate.
- For itineraries, use a day-by-day format with morning/afternoon/evening sections.
- Always end responses with a "📚 Sources" section listing which knowledge base documents and/or MCP tools were used.
- Distinguish between: (a) Knowledge Base Facts, (b) Live MCP Data, (c) AI-generated suggestions/recommendations.
- Mark AI-generated suggestions explicitly: "✨ Suggested by AI:" when going beyond the provided sources.

### Weather-Aware Planning
- When weather data is provided, always adjust activity recommendations:
  - Rain or high precipitation probability → prefer indoor attractions
  - Clear/sunny → recommend outdoor activities but note Singapore's heat
  - Always mention bringing an umbrella in Singapore regardless of forecast (tropical climate)

### Currency and Budget
- When currency data is provided, use exact figures from the MCP tool.
- Relate converted amounts to Singapore costs (hawker meals, MRT fares, etc.) to help users understand their budget.

### Conversation Context
- Remember user preferences stated earlier in the conversation (e.g., travelling with family, budget constraints, interests).
- Reference earlier context when relevant: "Since you mentioned travelling with children earlier..."

### What NOT to Do
- Do not provide flight or hotel booking guidance beyond general tips.
- Do not fabricate weather data, exchange rates, or attraction details.
- Do not use MCP tools to answer questions already well-covered by the knowledge base.
- Do not present AI suggestions as established facts.
"""


# Keywords used to decide which sources to fetch for a given question
WEATHER_KEYWORDS = [
    "weather", "forecast", "rain", "sunny", "temperature", "humidity",
    "climate", "umbrella", "monsoon", "hot", "wet", "dry", "storm",
    "activities tomorrow", "activities next",
]

CURRENCY_KEYWORDS = [
    "convert", "currency", "exchange rate", "sgd", "dollars", "rupee",
    "inr", "usd", "eur", "gbp", "budget in", "how much is", "rate",
    "how many sgd", "cost in singapore dollars",
]

# Word-boundary regex patterns to avoid false matches (e.g. "eat" inside "weather")
DESTINATION_KEYWORDS = [
    r"\battraction\b", r"\bvisit\b", r"\bneighbourhood\b", r"\bdistrict\b",
    r"\bfood\b", r"\beat\b", r"\brestaurant\b", r"\bhawker\b",
    r"\bitinerary\b", r"\bplan\b", r"\btransport\b", r"\bmrt\b",
    r"\bmuseum\b", r"\bpark\b", r"\bbeach\b", r"\bculture\b",
    r"\btip\b", r"\brecommendation\b", r"\bhotel\b", r"\bstay\b",
    r"\bthing to do\b", r"\bactivities\b", r"\bfamily\b", r"\bchildren\b",
    r"\bindoor\b", r"\boutdoor\b", r"\bguide\b", r"\bsafety\b", r"\blanguage\b",
]

# Phrases that indicate a pure weather question (no RAG needed)
PURE_WEATHER_PHRASES = [
    "what is the weather",
    "what's the weather",
    "weather forecast",
    "weather today",
    "weather tomorrow",
    "forecast for the next",
    "is rain expected",
    "will it rain",
]

# Phrases that indicate a pure currency question (no RAG needed)
PURE_CURRENCY_PHRASES = [
    "convert ", "how much is ", "how many sgd", "exchange rate",
]

# Destination keywords without "visit" and "eat" for the pure-weather override check
DESTINATION_ONLY_PATTERNS = [
    r"\battraction\b", r"\bneighbourhood\b", r"\bdistrict\b", r"\bfood\b",
    r"\beat\b", r"\brestaurant\b", r"\bhawker\b", r"\bitinerary\b",
    r"\bplan\b", r"\btransport\b", r"\bmrt\b", r"\bmuseum\b", r"\bpark\b",
    r"\bbeach\b", r"\bculture\b", r"\btip\b", r"\brecommendation\b",
    r"\bhotel\b", r"\bstay\b", r"\bthing to do\b", r"\bactivities\b",
    r"\bfamily\b", r"\bchildren\b", r"\bindoor\b", r"\boutdoor\b",
    r"\bguide\b", r"\bsafety\b", r"\blanguage\b",
]


def classify_intent(message: str) -> dict:
    """
    Decide which sources are needed for the user's question.
    Returns flags: needs_rag, needs_weather, needs_currency, is_combined.
    """
    msg_lower = message.lower()

    needs_weather = any(kw in msg_lower for kw in WEATHER_KEYWORDS)
    needs_currency = any(kw in msg_lower for kw in CURRENCY_KEYWORDS)
    needs_rag = any(re.search(kw, msg_lower) for kw in DESTINATION_KEYWORDS)

    # Always search the knowledge base for itinerary or trip planning requests
    if "itinerary" in msg_lower or "plan" in msg_lower or "trip" in msg_lower:
        needs_rag = True

    # Pure weather question — skip RAG unless destination info is also needed
    is_pure_weather = any(phrase in msg_lower for phrase in PURE_WEATHER_PHRASES)
    if is_pure_weather and not any(re.search(kw, msg_lower) for kw in DESTINATION_ONLY_PATTERNS):
        needs_rag = False

    # Pure currency question — skip RAG unless destination info is also needed
    is_pure_currency = any(phrase in msg_lower for phrase in PURE_CURRENCY_PHRASES)
    if is_pure_currency and not needs_rag:
        needs_rag = False

    # Default to RAG for anything unrecognised
    if not needs_weather and not needs_currency and not needs_rag:
        needs_rag = True

    return {
        "needs_rag": needs_rag,
        "needs_weather": needs_weather,
        "needs_currency": needs_currency,
        "is_combined": (needs_rag and (needs_weather or needs_currency)),
    }


def extract_currency_params(message: str) -> dict:
    """
    Parse currency conversion details from a natural language message.
    Supports patterns like:
      - "convert 50000 INR to SGD"
      - "how much is 200 SGD in USD"
      - "budget of INR 60,000"
    Returns a dict with amount, from_currency, to_currency,
    or {"show_rates": True} if no specific conversion is found.
    """
    msg = message.upper()

    # number + currency code + TO/IN + currency code
    pattern_num_first = r"(\d[\d,\.]*)\s*([A-Z]{3})\s+(?:TO|IN)\s+([A-Z]{3})"
    # currency code + number + TO/IN + currency code
    pattern_code_first = r"([A-Z]{3})\s+(\d[\d,\.]*)\s+(?:TO|IN)\s+([A-Z]{3})"
    # number + currency code + TO/INTO + currency code
    pattern_into = r"(\d[\d,\.]*)\s+([A-Z]{3})\s+(?:TO|INTO)\s+([A-Z]{3})"

    for pattern in [pattern_num_first, pattern_into]:
        match = re.search(pattern, msg)
        if match:
            try:
                return {
                    "amount": float(match.group(1).replace(",", "")),
                    "from_currency": match.group(2),
                    "to_currency": match.group(3),
                }
            except ValueError:
                pass

    match = re.search(pattern_code_first, msg)
    if match:
        try:
            return {
                "amount": float(match.group(2).replace(",", "")),
                "from_currency": match.group(1),
                "to_currency": match.group(3),
            }
        except ValueError:
            pass

    # Fallback: look for a currency code followed by a number (e.g. "INR 60,000")
    match = re.search(r"([A-Z]{3})\s+(\d[\d,\.]*)", msg)
    if match:
        try:
            from_curr = match.group(1)
            return {
                "amount": float(match.group(2).replace(",", "")),
                "from_currency": from_curr,
                "to_currency": "SGD" if from_curr != "SGD" else "USD",
            }
        except ValueError:
            pass

    return {"show_rates": True}


class TravelAssistant:
    """
    Main orchestrator — handles intent classification, RAG retrieval,
    MCP tool calls, and LLM response generation.
    Supports Google Gemini and OpenAI as providers.
    """

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash", provider: str = "google"):
        self.api_key = api_key
        self.provider = provider
        self.model = model

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            os.environ["GOOGLE_API_KEY"] = api_key
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=0.3,
                google_api_key=api_key,
                convert_system_message_to_human=True,
            )
        else:
            from langchain_openai import ChatOpenAI
            os.environ["OPENAI_API_KEY"] = api_key
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.3,
                openai_api_key=api_key,
            )

        self.vector_store = None
        self.mcp_manager = MCPToolManager()
        self.conversation_history = []
        self._initialized = False

    def initialize(self, force_rebuild_index: bool = False) -> dict:
        """
        Build or load the vector store and start MCP servers.
        Returns a status dict with rag, weather_mcp, currency_mcp flags.
        """
        status = {"rag": False, "weather_mcp": False, "currency_mcp": False}

        try:
            self.vector_store = get_or_build_vector_store(
                force_rebuild=force_rebuild_index,
                provider=self.provider,
                api_key=self.api_key,
            )
            status["rag"] = True
        except Exception as e:
            print(f"RAG initialization failed: {e}")
            status["rag_error"] = str(e)

        mcp_status = self.mcp_manager.initialize()
        status["weather_mcp"] = mcp_status.get("weather", False)
        status["currency_mcp"] = mcp_status.get("currency", False)

        self._initialized = True
        return status

    def chat(self, user_message: str) -> dict:
        """
        Process one user message and return the assistant's response.

        Steps:
          1. Classify intent (RAG / weather / currency / combined)
          2. Fetch relevant knowledge base chunks via RAG
          3. Call MCP tools if weather or currency data is needed
          4. Build the LLM prompt with all retrieved context
          5. Generate and return the response

        Returns a dict with: response, sources, mcp_tools_used, intent, rag_context.
        """
        if not self._initialized:
            return {
                "response": "Assistant not initialized. Please call initialize() first.",
                "sources": [],
                "mcp_tools_used": [],
                "intent": {},
                "rag_context": "",
            }

        intent = classify_intent(user_message)

        rag_context = ""
        sources = []
        mcp_results = {}
        mcp_tools_used = []

        # Step 1: retrieve relevant knowledge base chunks
        if intent["needs_rag"] and self.vector_store is not None:
            retrieved_docs = retrieve_relevant_chunks(self.vector_store, user_message, k=5)
            if retrieved_docs:
                rag_context = format_retrieved_sources(retrieved_docs)
                sources = get_source_references(retrieved_docs)

        # Step 2: call weather MCP tool if needed
        if intent["needs_weather"] and self.mcp_manager.is_connected():
            weather_result = self.mcp_manager.call_tool("get_full_weather", {})
            if weather_result["success"]:
                mcp_results["weather"] = weather_result["content"]
                mcp_tools_used.append("get_full_weather (Open-Meteo Weather API)")
            else:
                mcp_results["weather"] = f"Weather data unavailable: {weather_result.get('error', 'Unknown error')}"
                mcp_tools_used.append("get_full_weather (FAILED)")

        # Step 3: call currency MCP tool if needed
        if intent["needs_currency"] and self.mcp_manager.is_connected():
            currency_params = extract_currency_params(user_message)

            if currency_params.get("show_rates"):
                result = self.mcp_manager.call_tool("get_sgd_rates", {})
                if result["success"]:
                    mcp_results["currency"] = result["content"]
                    mcp_tools_used.append("get_sgd_rates (Exchange Rate API)")
                else:
                    mcp_results["currency"] = f"Currency data unavailable: {result.get('error', 'Unknown error')}"
                    mcp_tools_used.append("get_sgd_rates (FAILED)")
            else:
                result = self.mcp_manager.call_tool("convert_currency", currency_params)
                if result["success"]:
                    mcp_results["currency"] = result["content"]
                    mcp_tools_used.append(
                        f"convert_currency: {currency_params.get('amount', '')} "
                        f"{currency_params.get('from_currency', '')} → "
                        f"{currency_params.get('to_currency', '')} (Exchange Rate API)"
                    )
                else:
                    mcp_results["currency"] = f"Currency conversion unavailable: {result.get('error', 'Unknown error')}"
                    mcp_tools_used.append("convert_currency (FAILED)")

        # Step 4: assemble context and build the prompt
        context_sections = []

        if rag_context:
            context_sections.append(f"## KNOWLEDGE BASE EXCERPTS\n\n{rag_context}")
        elif intent["needs_rag"]:
            context_sections.append(
                "## KNOWLEDGE BASE EXCERPTS\n\nNo relevant information found in the knowledge base for this question."
            )

        if mcp_results.get("weather"):
            context_sections.append(
                f"## LIVE WEATHER DATA (from MCP Weather Tool)\n\n{mcp_results['weather']}"
            )

        if mcp_results.get("currency"):
            context_sections.append(
                f"## LIVE CURRENCY DATA (from MCP Currency Tool)\n\n{mcp_results['currency']}"
            )

        context_block = "\n\n---\n\n".join(context_sections)

        user_prompt_with_context = (
            f"User Question: {user_message}\n\n"
            f"---\n\n{context_block}\n\n---\n\n"
            "Please provide a helpful, accurate, and well-structured response. "
            "Use knowledge base excerpts for destination facts, live MCP data for weather and currency, "
            "clearly attribute information to its source, end with a '📚 Sources Used' section, "
            "and mark any AI suggestions explicitly."
        )

        # Step 5: send to LLM and parse the response
        messages = []

        if self.provider == "google":
            # Gemini works best with the system prompt merged into the first human message
            first_human = f"{SYSTEM_PROMPT}\n\n---\n\n{user_prompt_with_context}"
            for turn in self.conversation_history[-8:]:
                messages.append(
                    HumanMessage(content=turn["content"])
                    if turn["role"] == "user"
                    else AIMessage(content=turn["content"])
                )
            messages.append(HumanMessage(content=first_human))
        else:
            # OpenAI supports a dedicated system message
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            for turn in self.conversation_history[-8:]:
                messages.append(
                    HumanMessage(content=turn["content"])
                    if turn["role"] == "user"
                    else AIMessage(content=turn["content"])
                )
            messages.append(HumanMessage(content=user_prompt_with_context))

        try:
            response = self.llm.invoke(messages)
            raw = response.content

            if isinstance(raw, str):
                response_text = raw
            elif isinstance(raw, list):
                # Gemini 3.x returns a list of content blocks; extract text fields only
                parts = []
                for block in raw:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("text"):
                        parts.append(block["text"])
                response_text = "\n".join(parts)
            else:
                response_text = str(raw)

            # Remove any stray control characters that some model versions emit
            response_text = "".join(
                ch for ch in response_text
                if ch in ("\n", "\t") or (32 <= ord(ch) <= 126) or ord(ch) > 127
            )

        except Exception as e:
            response_text = f"I encountered an error generating a response: {str(e)}"

        # Save this turn to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "sources": sources,
            "mcp_tools_used": mcp_tools_used,
            "intent": intent,
            "rag_context": rag_context,
        }

    def reset_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def shutdown(self) -> None:
        """Stop MCP servers and release resources."""
        self.mcp_manager.shutdown()
