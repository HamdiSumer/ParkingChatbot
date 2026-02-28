# Parking Chatbot - Intelligent RAG System

An intelligent parking chatbot with **Retrieval-Augmented Generation (RAG)**, **Weaviate vector database**, **LangChain**, **LangGraph**, and flexible LLM providers.

## Requirements

- **Python 3.10+**
- **Docker & Docker Compose** (for Weaviate)
- **One LLM provider** (choose one):
  - Ollama (free, local) - default
  - OpenAI API | Google Gemini API | Anthropic Claude API

## Setup (4 Steps)

### 1. Start Weaviate
```bash
docker-compose up -d
sleep 10
docker-compose ps  # Should show: weaviate ... Up
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env if needed (default is Ollama)
```

### 3. Install Dependencies
```bash
uv init
uv install
```

### 4. LLM Configuration
Choose one and set in `.env`:

### Ollama (Free, Local) - Default
```env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2:7b
```

In another terminal:
```bash
ollama serve
ollama pull llama2:7b  # First time only
```

### OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
```

### Google Gemini
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-pro
```

### Anthropic Claude
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```


### 4. Run the Chatbot
```bash
uv run python main.py
```

### 5. Admin Dashboard
```bash
uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8001
```

Access the dashboard from: 0.0.0.0:8001/dashboard

---

## Features

- 🤖 **Intelligent ReAct Agent** - LLM decides what tools to use (no retrieval for greetings!)
- 🚗 **Parking Information** - RAG-based document retrieval (vector search)
- 📊 **Hybrid Retrieval** - Vector DB (static) + SQL Agent (real-time data)
- 📋 **Reservations** - Interactive multi-step booking workflow
- 👤 **Human-in-the-Loop** - Admin approval system
- 🖥️ **Admin Dashboard** - Web UI for viewing/approving/rejecting reservations
- 🔐 **API Security** - API key authentication & rate limiting for dashboard
- 📁 **Reservation Export** - Confirmed reservations written to file (MCP-style)
- 🔒 **Security** - PII detection and sensitive data filtering
- 📈 **Evaluation** - Comprehensive RAG metrics (RAGAS framework) + auto-reports
- 🔄 **Flexible LLMs** - Ollama, OpenAI, Gemini, or Claude

---

## Architecture
![Architecture Diagram](docs/architecture_diagram.svg)

---

## Intelligent Agent Routing

The chatbot uses a **ReAct (Reasoning and Acting) agent** that intelligently decides what tools to use:

| User Input | Agent Decision | What Happens |
|------------|----------------|--------------|
| "Hey" / "Thanks" | `direct_response` | No retrieval, just responds |
| "Where is downtown parking?" | `vector_search` | Searches static knowledge base |
| "How many spaces available?" | `sql_query` | Queries real-time database |
| "I want to book parking" | `start_reservation` | Starts booking workflow |

This means greetings don't trigger expensive RAG retrieval, and real-time queries use the SQL agent instead of outdated vector data.

---

## Testing

Run comprehensive tests including RAGAS metrics, retrieval quality, and end-to-end workflows:

```bash
uv run python test_rag.py              # Run tests + generate markdown report
uv run python test_rag.py --no-report  # Run tests without report
```

Tests included:
- Guardrails (security & sensitive data detection)
- RAG Metrics (faithfulness, relevance, context precision)
- Recall@K (retrieval ranking quality)
- Hybrid Retrieval (SQL Agent + Vector DB)
- Component initialization (embeddings, vector DB, LLM)
- End-to-end workflow
- **Agent Routing** (ReAct agent tool selection - greetings vs retrieval)
- Data architecture verification

**Reports** are auto-saved to `reports/` folder as:
`{provider}_{model}_test_results_{timestamp}.md`

For detailed testing methodology, metrics explanation, and how tests map to system components, see **[docs/TESTING.md](docs/TESTING.md)**.

---


## Project Structure

```
src/
├── rag/              # RAG pipeline, embeddings, LLM providers, SQL agent
├── database/         # Weaviate & SQLite storage
├── agents/           # LangGraph ReAct agent workflow
│   ├── workflow.py   # Agent-based routing (vector/sql/direct)
│   ├── state.py      # Conversation state + agent tracking
│   ├── tools.py      # Tool definitions (VectorSearch, SQLQuery)
│   └── prompts.py    # Agent decision prompts
├── api/              # REST API & Admin Dashboard
│   ├── dashboard.py  # Admin web UI + API endpoints
│   └── security.py   # API key auth, rate limiting
├── services/         # Background services
│   └── reservation_writer.py  # File export for confirmed reservations
├── guardrails/       # Security, PII detection, data filtering
├── evaluation/       # Metrics and evaluation components
├── app.py           # Main application logic
└── cli.py           # Interactive command-line interface

test_rag.py          # Comprehensive test suite + report generation
reports/             # Auto-generated test reports (markdown)
main.py              # Entry point
```

---

## Data Storage

- **Weaviate Vector DB** - Static parking information (RAG knowledge base)
- **SQLite** - Dynamic data (availability, prices, reservations, user info)
- **File Export** - Confirmed reservations saved to `data/confirmed_reservations/`

---

## Admin Dashboard

Access the admin dashboard at `http://localhost:8000/dashboard/` when the server is running.

**Features:**
- View all pending reservation requests
- One-click approve/reject with real-time updates
- Rejection reason modal
- Recent activity history
- Auto-refresh every 30 seconds

**API Security (optional):**
```env
# In .env
ADMIN_API_KEY=your-secret-key
REQUIRE_API_KEY=true
```

When enabled, approve/reject endpoints require the API key via:
- Header: `X-API-Key: your-secret-key`
- Query: `?api_key=your-secret-key`

**Rate Limiting:** 100 requests per minute per IP (prevents abuse).

---

## Quick Commands in Chat

```
help              - Show available commands
evaluate          - Run evaluation metrics
list spaces       - Show parking spaces
quit              - Exit

Or just ask:
"Where is downtown parking?"
"What are the prices?"
"I want to book a space"
```
