# Quick Start (Copy & Paste)

## Get Running in 5 Minutes

### 1. Start Weaviate
```bash
docker-compose up -d
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env if needed (default is Ollama)
```

### 3. Install & Run
```bash
uv init
uv install
uv run python main.py
```

That's it!

---

## In Another Terminal: Start Ollama (if using Ollama)

```bash
ollama serve
# In yet another terminal:
ollama pull llama2
```

---

## Try These Commands

```
You: Where is downtown parking?
You: What are the prices?
You: I want to book a space
You: evaluate
You: quit
```

---

## Docker Commands

```bash
# View status
docker-compose ps

# View logs
docker-compose logs -f weaviate

# Stop
docker-compose down

# Restart
docker-compose down
docker-compose up -d
```

---

See [INSTALLATION.md](INSTALLATION.md) for full guide.
