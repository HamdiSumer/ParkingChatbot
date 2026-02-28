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
- 📊 **Hybrid Retrieval** - Vector DB (static) + SQL Agent (real-time data with reservation awareness)
- 📋 **Reservations** - Interactive multi-step booking with LLM-based intent classification
- 👤 **Human-in-the-Loop (HITL)** - LangGraph workflow with interrupt support for admin approval
- 🖥️ **Admin Dashboard** - Web UI for viewing/approving/rejecting reservations
- 🔐 **API Security** - API key authentication & rate limiting for dashboard
- 🔧 **MCP Server** - Model Context Protocol server for reservation file operations (Claude Desktop/Ollama compatible)
- 🔒 **Security** - PII detection, input sanitization, access logging, and sensitive data filtering
- 📈 **Evaluation** - Comprehensive RAG metrics (RAGAS framework) + auto-reports
- 🔄 **Flexible LLMs** - Ollama, OpenAI, Gemini, or Claude
- ⚡ **Load Testing** - Concurrent user simulation, admin operations stress testing, MCP server performance
- 🔗 **Pipeline Integration** - Full end-to-end testing from user query to data recording

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

Run comprehensive tests including RAGAS metrics, load tests, retrieval quality, and end-to-end workflows:

```bash
uv run python test_rag.py              # Run tests + generate markdown report
uv run python test_rag.py --no-report  # Run tests without report
```

### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| **Security** | Guardrails | PII detection, SQL injection prevention, input sanitization |
| **RAG Quality** | RAGAS Metrics | Faithfulness, answer relevance, context precision |
| **Retrieval** | Recall@K, Hybrid | Vector DB + SQL Agent ranking quality |
| **Agent** | Routing | ReAct agent tool selection (greetings vs retrieval) |
| **Admin** | HITL Flow | Human-in-the-loop approval/rejection workflow |
| **Load Tests** | Concurrent Users | 5 concurrent users, 3 queries each |
| **Load Tests** | Admin Operations | 5 concurrent admins, 4 operations each |
| **Load Tests** | MCP Server | 5 concurrent writers, 4 operations each |
| **MCP Server** | Functional | Write, read, file info, input validation |
| **Integration** | Full Pipeline | End-to-end: Query → RAG → Admin → MCP |

### Load Testing

The test suite includes comprehensive load tests to evaluate system performance under concurrent access:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           LOAD TEST METRICS                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│  Chatbot Load Test                                                           │
│  ├─ Concurrent Users: 5                                                      │
│  ├─ Queries per User: 3                                                      │
│  ├─ Metrics: Success Rate, Avg/Min/Max/P95 Response Time                     │
│  └─ Pass Criteria: >80% success, <30s avg response                           │
│                                                                              │
│  Admin Load Test                                                             │
│  ├─ Concurrent Admins: 5                                                     │
│  ├─ Operations: Create, List, Approve, Reject                                │
│  └─ Pass Criteria: >80% success rate                                         │
│                                                                              │
│  MCP Server Load Test                                                        │
│  ├─ Concurrent Writers: 5                                                    │
│  ├─ Operations: Write, Read, File Info                                       │
│  └─ Pass Criteria: >80% success rate                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Full Pipeline Integration Test

Tests the complete system workflow:

```
User Query → Safety Check → RAG Retrieval → Response
     ↓
Reservation Request → Data Collection → Admin Queue
     ↓
Admin Decision → Status Update → MCP Recording
```

**Reports** are auto-saved to `reports/` folder as:
`{provider}_{model}_test_results_{timestamp}.md`

For detailed testing methodology, metrics explanation, and how tests map to system components, see **[docs/TESTING.md](docs/TESTING.md)**.

---


## Project Structure

```
src/
├── rag/              # RAG pipeline, embeddings, LLM providers, SQL agent
│   └── sql_agent.py  # SQL Agent with reservation-aware availability queries
├── database/         # Weaviate & SQLite storage
├── agents/           # LangGraph agent workflows
│   ├── workflow.py   # ReAct agent routing (vector/sql/direct)
│   ├── hitl_workflow.py  # Human-in-the-Loop workflow with LLM intent classification
│   ├── state.py      # Conversation state + agent tracking
│   ├── tools.py      # Tool definitions (VectorSearch, SQLQuery)
│   └── prompts.py    # Agent decision prompts
├── admin/            # Admin service for reservation management
├── api/              # REST API & Admin Dashboard
│   ├── dashboard.py  # Admin web UI + API endpoints
│   └── security.py   # API key auth, rate limiting
├── mcp/              # MCP (Model Context Protocol) server
│   └── reservation_server.py  # MCP tools with security (rate limit, auth, logging)
├── guardrails/       # Security, PII detection, data filtering
├── evaluation/       # Metrics and evaluation components
├── app.py           # Main application logic
└── cli.py           # Interactive command-line interface

test_rag.py          # Comprehensive test suite (14 tests) + report generation
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

## MCP Server (Model Context Protocol)

The MCP server enables external AI assistants (Claude Desktop, Ollama) to interact with reservation files.

**Tools Available:**
- `write_reservation` - Write confirmed reservations to file
- `read_reservations` - Read reservation history
- `get_reservation_file_info` - Get file metadata

**Security Features:**
- Rate limiting (60 requests/minute, configurable)
- API key authentication (optional)
- Input sanitization
- File locking for concurrent access
- Access logging/audit trail

**Configuration:**
```env
MCP_REQUIRE_AUTH=true     # Enable API key auth
MCP_API_KEY=your-key      # Set API key
MCP_RATE_LIMIT=60         # Max requests per minute
```

**For Claude Desktop/Ollama:** See `mcp_config.json` for connection settings.

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
