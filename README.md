# Parking Chatbot - Intelligent RAG System

An intelligent parking chatbot with **Retrieval-Augmented Generation (RAG)**, **Milvus vector database**, **LangChain**, **LangGraph**, and flexible LLM providers.

## 🚀 Quick Start

**See [INSTALLATION.md](INSTALLATION.md) for setup instructions.**

## Requirements

- **Python 3.10+**
- **Docker & Docker Compose** (for Weaviate)
- **One LLM provider** (choose one):
  - Ollama (free, local) - default
  - OpenAI API
  - Google Gemini API
  - Anthropic Claude API

## Setup (4 Steps)

```bash
# 1. Start Weaviate
docker-compose up -d

# 2. Configure
cp .env.example .env
# Edit .env if needed

# 3. Install
uv init
uv install

# 4. Run
uv run python main.py
```

## Features

- 🚗 **Parking Information**: RAG-based document retrieval
- 📋 **Reservations**: Interactive multi-step booking
- 👤 **Human-in-the-Loop**: Admin approval workflow
- 🔒 **Security**: PII detection and response filtering
- 📊 **Evaluation**: Comprehensive performance metrics
- 🔄 **Flexible LLMs**: Ollama, OpenAI, Gemini, or Claude

## Architecture

```
User Input
    ↓
[Safety Filter] → [Intent Detection]
    ↓
[RAG Pipeline] → [Milvus Vector DB]
    ↓
[LLM] (Ollama/OpenAI/Gemini/Claude)
    ↓
[Response Filter]
    ↓
[Human Review] (Optional)
    ↓
User Output
```

## Documentation

- **[WEAVIATE_SETUP.md](WEAVIATE_SETUP.md)** ⭐ Start here - Easiest setup
- **[INSTALLATION.md](INSTALLATION.md)** - Full setup instructions
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - How to verify everything works
- **[QUICK_START.md](QUICK_START.md)** - Copy-paste commands
- **[project_overview/](project_overview/)** - Detailed docs (not pushed to git)

## LLM Providers

### Ollama (Free, Local)
```env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
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

## Project Structure

```
src/
├── rag/              # RAG pipeline, embeddings, LLM providers
├── database/         # Milvus & SQLite
├── agents/           # LangGraph workflow
├── guardrails/       # Security & PII detection
├── evaluation/       # Metrics & testing
├── app.py           # Main application
└── cli.py           # Interactive interface
```

## Running the Chatbot

```bash
uv run python main.py
```

