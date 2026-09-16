# Singapore AI Travel Planning Assistant

A context-aware travel planning assistant built with Python and LangChain that combines a **document-based RAG knowledge base** with **live data retrieved through MCP tools** to help users plan trips to Singapore.

The application answers destination questions from a curated knowledge base, retrieves live weather forecasts and currency exchange rates through MCP-connected external services, and combines both sources when a query requires it — for example, generating a weather-aware itinerary.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Features](#features)
3. [Knowledge Base](#knowledge-base)
4. [RAG Workflow](#rag-workflow)
5. [MCP Tools](#mcp-tools)
6. [Prompt and Context Strategy](#prompt-and-context-strategy)
7. [Technology Stack](#technology-stack)
8. [Project Structure](#project-structure)
9. [Setup and Installation](#setup-and-installation)
10. [Running the Application](#running-the-application)
11. [Sample Questions and Responses](#sample-questions-and-responses)
12. [Acceptance Criteria Checklist](#acceptance-criteria-checklist)

---

## Architecture

The application has three main parts that work together to answer a user's question.

**1. The UI** (`app.py`)
A simple Streamlit chat interface. The user types a question, the app sends it to the assistant, and displays the response with source references.

**2. The Assistant** (`src/travel_assistant.py`)
The brain of the application. Every time a message comes in, it does three things in order:

- **Figures out what kind of question it is** — Does it need destination facts? Live weather? Currency conversion? Or a mix? This runs before any API call is made.
- **Fetches the right information** — If destination facts are needed, it searches the knowledge base. If weather or currency is needed, it calls the relevant MCP tool.
- **Generates the response** — It hands the retrieved information to the LLM along with the conversation history, and the LLM writes the final answer.

**3. The Data Sources**

There are two types of data sources:

- **Knowledge Base (RAG)** — Four Markdown files about Singapore travel, broken into chunks, embedded, and stored in a FAISS vector store on disk. When a destination question comes in, the top 5 most relevant chunks are retrieved and passed to the LLM as context.

- **MCP Tools** — Two small Python servers that run as background processes and communicate over JSON-RPC. One calls the Open-Meteo weather API, the other calls the Frankfurter/Open Exchange Rates currency API. The assistant sends a request, gets back live data, and includes it in the LLM prompt.

**How a typical request flows:**

```
User types a question
        │
        ▼
Intent classifier checks the message
        │
        ├── destination question? ──► search FAISS vector store ──► get top 5 chunks
        │
        ├── weather question?     ──► call weather MCP server   ──► get live forecast
        │
        └── currency question?    ──► call currency MCP server  ──► get live rate
                                                │
                                                ▼
                        Combine retrieved context + conversation history
                                                │
                                                ▼
                                    Send to LLM (Gemini / OpenAI)
                                                │
                                                ▼
                            Response with sources shown in the UI
```

A combined question (e.g. "plan a trip based on the weather") triggers both the knowledge base search and the weather MCP call before a single LLM call generates the merged response.

---

## Features

| Feature | Details |
|---|---|
| Destination knowledge (RAG) | FAISS vector store with embedding-based semantic search over Singapore travel documents |
| Weather information | MCP weather server calls Open-Meteo API — current conditions and 7-day forecast |
| Currency conversion | MCP currency server calls Frankfurter API (ECB data) with fallback to Open Exchange Rates |
| Combined RAG + MCP | Intent classifier routes queries to both sources; LLM merges them in one response |
| Multi-turn conversation | Last 8 conversation turns passed to LLM to preserve context between messages |
| Source attribution | Every response cites the knowledge base document and/or MCP tool used |
| Graceful degradation | Failed tool calls and missing knowledge are stated explicitly — nothing is fabricated |
| Weather-aware planning | When forecast data is present, itineraries suggest indoor alternatives on rainy days |
| Two LLM providers | Supports Google Gemini (free tier) and OpenAI — selectable from the UI |

---

## Knowledge Base

The knowledge base is four Markdown files in the `knowledge_base/` directory, each sourced from publicly available Singapore travel resources. The source title and original URL are stored as document metadata so the application can produce meaningful citations.

| File | Source | Coverage |
|---|---|---|
| `wikivoyage_singapore.md` | [Wikivoyage — Singapore](https://en.wikivoyage.org/wiki/Singapore) | Districts, attractions, transport, food, cultural tips, itineraries |
| `visitsingapore_essential.md` | [Visit Singapore — Travel Guide](https://www.visitsingapore.com/travel-guide-tips/) | Entry requirements, climate, language, currency, connectivity, safety |
| `visitsingapore_itineraries.md` | [Visit Singapore — Itinerary Planner](https://www.visitsingapore.com/itinerary-planner/) | 1-day, 2-day, 3-day, family, budget, and rainy-day itineraries |
| `visitsingapore_things_to_do.md` | [Visit Singapore — Things To Do](https://www.visitsingapore.com/see-do-singapore/) | Attractions, theme parks, beaches, food experiences, shopping |

Content was manually reviewed and adapted from these public sources. The original URLs are retained in each document header and carried through as chunk metadata for citation purposes.

---

## RAG Workflow

```
knowledge_base/*.md  (4 documents)
         │
         ▼
MarkdownHeaderTextSplitter
  splits on # / ## / ### headers      ← preserves section context per chunk
         │
         ▼
RecursiveCharacterTextSplitter
  chunk_size=800, overlap=150          ← further splits sections that are too long
         │
         ▼
Embedding model
  Google: text-embedding-004
  OpenAI: text-embedding-3-small       ← 115 chunks embedded on first run
         │
         ▼
FAISS vector store
  saved to vector_store/ on disk       ← reloaded on subsequent runs, no re-embedding
         │
         ▼
Similarity search  k=5
  retrieves the 5 most relevant chunks for each user question
         │
         ▼
Context assembly
  each chunk includes source name + URL as a citation header
         │
         ▼
LLM  (Gemini / OpenAI)
  generates a grounded response from the retrieved excerpts
```

**Design decisions:**

- **Header-first splitting** — splitting by Markdown headers before characters means each chunk stays within a single topic section. An attraction like "Gardens by the Bay" is never split mid-description.
- **Metadata on every chunk** — source name, URL, and section hierarchy are attached to every chunk at load time, so citations are always available regardless of which chunks are retrieved.
- **k=5** — retrieving five chunks gives the LLM enough context for most questions without exceeding prompt token limits or introducing irrelevant content.
- **Disk caching** — the FAISS index is saved after the first build and reloaded on subsequent runs, so embeddings are generated only once per knowledge base.

---

## MCP Tools

Both MCP servers are standalone Python scripts that communicate over **stdin/stdout using the JSON-RPC 2.0 protocol** (MCP spec version 2024-11-05). The `MCPToolManager` in `src/mcp_client.py` spawns each server as a subprocess, performs the `initialize` / `notifications/initialized` handshake, discovers available tools via `tools/list`, and routes `tools/call` requests to the correct server.

### Tool 1 — Weather (`mcp_servers/weather_server.py`)

**API:** [Open-Meteo](https://open-meteo.com/) — completely free, no API key required.

| Tool name | Description |
|---|---|
| `get_current_weather` | Temperature, humidity, precipitation, wind speed, UV index for Singapore right now |
| `get_weather_forecast` | Daily forecast for 1–7 days: condition, min/max temp, precipitation, rain probability |
| `get_full_weather` | Current conditions + full 7-day forecast in one call |

Weather codes are mapped to WMO standard descriptions. The server also generates a simple activity recommendation (indoor vs outdoor) based on the weather code returned.

### Tool 2 — Currency (`mcp_servers/currency_server.py`)

**Primary API:** [Frankfurter](https://www.frankfurter.app/) — European Central Bank data, free, no key required.  
**Fallback API:** [Open Exchange Rates](https://open.er-api.com/) — broader currency coverage for INR, IDR, THB, etc.

| Tool name | Description |
|---|---|
| `convert_currency` | Convert any amount between two currencies using live rates |
| `get_sgd_rates` | Current SGD exchange rates against USD, EUR, GBP, INR, AUD, JPY, MYR, IDR, THB |

Conversion results include Singapore cost context (hawker meal prices, MRT fares, hotel ranges in the source currency) to help users interpret their budget.

### Error handling

If an MCP server is unavailable or an API call fails, the tool returns a clear error message that is passed to the LLM with an instruction not to fabricate data. The assistant will state that the information is currently unavailable rather than invent figures.

---

## Prompt and Context Strategy

The system prompt is defined in `src/travel_assistant.py` as `SYSTEM_PROMPT`. It establishes rules that govern every response the LLM generates:

**Source discipline**
The prompt instructs the model to use knowledge base excerpts exclusively for destination facts (attractions, transport, food, culture) and MCP tool results exclusively for current information (weather, exchange rates). It explicitly forbids inventing prices, attraction details, or travel facts not present in the provided context.

**Attribution**
The model is instructed to attribute every factual claim to its source using phrases like "According to the Wikivoyage Singapore guide..." for RAG content and "Based on live data from Open-Meteo..." for MCP results.

**Transparency between source types**
Responses distinguish three categories of content: (a) knowledge base facts, (b) live MCP data, and (c) AI-generated suggestions. Suggestions that go beyond the provided sources are marked with "✨ Suggested by AI:" so the reader can identify them.

**Weather-aware planning**
When weather data is present in the context, the prompt instructs the model to adjust activity recommendations — preferring indoor attractions when rain probability is high, and recommending outdoor activities when conditions are clear.

**Conversation memory**
The last 8 turns of the conversation are included in the message list sent to the LLM. The prompt instructs the model to reference prior context where relevant — for example, referring back to a family travel preference or budget constraint mentioned earlier in the chat.

**Graceful degradation**
If the knowledge base does not contain relevant information for a question, the model is instructed to say so explicitly rather than filling the gap with invented content.

**Intent classification**
Before any LLM call, `classify_intent()` in `travel_assistant.py` analyses the user message using keyword and regex matching to determine whether the query needs RAG, weather, currency, or a combination. This avoids unnecessary API calls — a pure weather question does not trigger a knowledge base lookup, and a pure destination question does not call any MCP tool.

---

## Technology Stack

| Component | Technology | Notes |
|---|---|---|
| Application orchestration | LangChain | Chains, prompts, retrieval, message history |
| LLM (default) | Google Gemini 3.6 Flash | Free tier: 1,500 requests/day |
| LLM (alternative) | OpenAI GPT-4o mini | Paid API |
| Embeddings (Google) | `text-embedding-004` | Via `langchain-google-genai` |
| Embeddings (OpenAI) | `text-embedding-3-small` | Via `langchain-openai` |
| Vector store | FAISS (`faiss-cpu`) | Persisted to disk |
| MCP protocol | Custom Python stdio servers | JSON-RPC 2.0, MCP spec 2024-11-05 |
| Weather data | Open-Meteo API | Free, no API key |
| Currency data | Frankfurter API + Open Exchange Rates | Free, no API key |
| User interface | Streamlit | Single-page chat UI |
| Python | 3.10+ | Tested on 3.14 |

---

## Project Structure

```
singapore-travel-assistant/
│
├── app.py                          # Streamlit UI — main entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore
├── README.md
│
├── knowledge_base/                 # Singapore travel content (Markdown)
│   ├── wikivoyage_singapore.md
│   ├── visitsingapore_essential.md
│   ├── visitsingapore_itineraries.md
│   └── visitsingapore_things_to_do.md
│
├── mcp_servers/                    # Standalone MCP tool servers (JSON-RPC 2.0 stdio)
│   ├── weather_server.py           # Open-Meteo weather tools
│   └── currency_server.py          # Frankfurter + Open Exchange Rates currency tools
│
├── src/                            # Core application modules
│   ├── __init__.py
│   ├── rag_engine.py               # Document loading, chunking, embedding, FAISS
│   ├── mcp_client.py               # MCP subprocess client and tool manager
│   └── travel_assistant.py         # Orchestrator: intent classification, RAG, MCP, LLM
│
└── vector_store/                   # FAISS index — generated on first run, git-ignored
    ├── index.faiss
    └── index.pkl
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- A Google AI API key (free) **or** an OpenAI API key (paid)
  - Google: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
  - OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 1. Navigate to the project directory

```bash
cd singapore-travel-assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure API key in environment

```bash
cp .env.example .env
# Open .env and set GOOGLE_API_KEY or OPENAI_API_KEY
```

The API key can also be entered directly in the UI — no `.env` file is required.

---

## Running the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

**First-time setup in the browser:**

1. Select **Google Gemini (Free)** or **OpenAI** as the provider
2. Paste your API key into the key field
3. Click **▶ Start**
4. Wait approximately 30 seconds while the knowledge base index is built
5. The FAISS index is saved to `vector_store/` and reused on all subsequent runs

To stop the application, press `Ctrl+C` in the terminal.

---

## Sample Questions and Responses

### RAG only — destination knowledge

**Question:** What are the must-visit attractions in Singapore?

**Response summary:**
The assistant retrieves relevant chunks from the knowledge base and lists attractions including Gardens by the Bay (Supertree Grove, Cloud Forest, Flower Dome), Marina Bay Sands (SkyPark, ArtScience Museum, Spectra show), Singapore Zoo, Night Safari, Universal Studios Singapore, Merlion Park, and the Singapore Botanic Gardens. Each recommendation is attributed to the source document.

---

### MCP only — currency conversion

**Question:** Convert INR 50,000 to SGD.

**Response summary:**
The assistant calls the `convert_currency` MCP tool with `amount=50000`, `from_currency=INR`, `to_currency=SGD`. The live rate is retrieved from Open Exchange Rates and the result is returned with the exchange rate, date, and a breakdown of what the converted amount covers in Singapore (hawker meals, MRT trips, budget accommodation).

---

### MCP only — weather

**Question:** What is the weather forecast for the next 3 days in Singapore?

**Response summary:**
The assistant calls `get_weather_forecast` with `days=3`. The response shows daily high/low temperatures, weather condition, precipitation amount, rain probability, and UV index for each day, sourced from Open-Meteo.

---

### Combined — weather-aware itinerary

**Question:** Create a 3-day Singapore itinerary for next week adjusted for the weather forecast.

**Response summary:**
The assistant retrieves attraction and itinerary information from the knowledge base (RAG) and simultaneously calls `get_full_weather` (MCP) to get the 7-day forecast. It then generates a day-by-day itinerary where activities are chosen based on the forecast — outdoor activities on clear days, indoor alternatives (Cloud Forest dome, ArtScience Museum, Chinatown Heritage Centre) on days with high rain probability. The response clearly separates knowledge base facts from live weather data and AI-generated suggestions.

---

### Combined — budget conversion and itinerary

**Question:** I have a budget of INR 60,000. Convert it to SGD and suggest a 3-day itinerary.

**Response summary:**
The assistant calls `convert_currency` (MCP) to convert 60,000 INR to SGD, then retrieves itinerary and budget guidance from the knowledge base (RAG). The response shows the converted amount, daily cost estimates for food and transport in SGD, and a 3-day itinerary with hawker centre meals and free or low-cost attractions to stay within budget.

---

## Acceptance Criteria Checklist

- ✅ Knowledge base built from 4 public travel resources (Wikivoyage + 3 Visit Singapore pages)
- ✅ Embedding-based semantic retrieval (FAISS + Google/OpenAI embeddings)
- ✅ Grounded answers with source references — title and URL shown per response
- ✅ Weather information through an MCP tool (Open-Meteo via `weather_server.py`)
- ✅ Currency conversion through an MCP tool (Frankfurter + ER-API via `currency_server.py`)
- ✅ At least one response combining RAG and MCP — weather-aware itinerary and budget+itinerary scenarios
- ✅ Multi-turn conversation with retained context — last 8 turns in LangChain message history
- ✅ Appropriate tool selection based on user intent — keyword + regex intent classifier
- ✅ Clear handling of missing knowledge and tool failures — explicit messages, no fabrication
- ✅ Simple, usable interface — Streamlit chat UI with example questions and reset control
