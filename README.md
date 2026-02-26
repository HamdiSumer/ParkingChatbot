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

---

## Features

- 🚗 **Parking Information** - RAG-based document retrieval
- 📋 **Reservations** - Interactive multi-step booking workflow
- 👤 **Human-in-the-Loop** - Admin approval system
- 🔒 **Security** - PII detection and sensitive data filtering
- 📊 **Evaluation** - Comprehensive RAG metrics (RAGAS framework)
- 🔄 **Flexible LLMs** - Ollama, OpenAI, Gemini, or Claude

---

## Architecture
![Architecture Diagram](docs/architecture-diagram.svg)

---

## Testing

Run comprehensive tests including RAGAS metrics, retrieval quality, and end-to-end workflows:

```bash
uv run python test_rag.py
```

Tests included:
- Guardrails (security & sensitive data detection)
- RAG Metrics (faithfulness, relevance, context precision)
- Recall@K (retrieval ranking quality)
- Component initialization (embeddings, vector DB, LLM)
- End-to-end workflow
- Data architecture verification

---


## Project Structure

```
src/
├── rag/              # RAG pipeline, embeddings, LLM providers
├── database/         # Weaviate & SQLite storage
├── agents/           # LangGraph workflow automation
├── guardrails/       # Security, PII detection, data filtering
├── evaluation/       # Metrics and evaluation components
├── app.py           # Main application logic
└── cli.py           # Interactive command-line interface

test_rag.py          # Comprehensive test suite
main.py              # Entry point
```

---

## Data Storage

- **Weaviate Vector DB** - Static parking information (RAG knowledge base)
- **SQLite** - Dynamic data (availability, prices, reservations, user info)

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
