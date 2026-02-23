# Installation

## Prerequisites
- Python 3.10+
- Docker & Docker Compose
- One LLM provider: Ollama, OpenAI, Gemini, or Anthropic

## Setup (4 Steps)

### 1. Start Weaviate
```bash
docker-compose up -d
```

Wait 10 seconds for startup:
```bash
sleep 10
docker-compose ps
# Should show: weaviate ... Up
```

Verify:
```bash
curl http://localhost:8080/v1/.well-known/ready
# Should return: ok
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env if needed - default is Ollama
```

### 3. Install Dependencies
```bash
uv init
uv install
```

### 4. Run
```bash
uv run python main.py
```

---

## Configuration Options

### LLM Provider

**Ollama (Free, Local)**
```env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama2
```

Then run in another terminal:
```bash
ollama serve
ollama pull llama2  # First time only
```

**OpenAI**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
```

**Google Gemini**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-pro
```

**Anthropic Claude**
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-sonnet-20240229
```

---

## Running the Chatbot

```bash
uv run python main.py
```

Type `help` for available commands.

---

## Docker Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f weaviate

# Clean restart
docker-compose down -v
docker-compose up -d
```

---

## Troubleshooting

**Container won't start:**
```bash
docker-compose logs weaviate
```

**Port already in use:**
```bash
docker-compose down
docker-compose up -d
```

**Connection refused:**
```bash
# Make sure Docker containers are running
docker-compose ps

# Wait a bit longer for startup
sleep 20
curl http://localhost:8080/v1/.well-known/ready
```

**Python import errors:**
```bash
uv sync
```
