# Quick Start Guide - Parking Chatbot

Get the parking chatbot up and running in 5 minutes.

## 🚀 Fast Setup

### Step 1: Install Ollama (2 minutes)
```bash
# Download from https://ollama.ai
# Or via package manager:
# macOS: brew install ollama
# Linux: curl https://ollama.ai/install.sh | sh
# Windows: Download installer from ollama.ai

# Start Ollama server (keeps running)
ollama serve
```

### Step 2: Pull a Model (2 minutes)
In a **new terminal**, pull a language model:
```bash
# Recommended for fast performance
ollama pull llama2

# Or try these alternatives:
ollama pull mistral       # Faster, less accurate
ollama pull neural-chat   # Optimized for chat
```

### Step 3: Install Python Dependencies (1 minute)
```bash
cd /home/hamdi/Desktop/ai_task

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Step 4: Run the Chatbot! (0 minutes)
```bash
python main.py
```

## 💬 First Interaction

```
You: What parking spaces are available?
Bot: [RAG-retrieved parking information]

You: I want to book a space
Bot: Great! What is your first name?

You: John
Bot: What is your last name?

...and so on
```

## 🧪 Run Demo

See all features in action:
```bash
python demo.py
```

This demonstrates:
- Information retrieval
- Reservation collection
- Safety filtering
- Data management
- System evaluation

## ⚙️ Configuration

Edit `.env` if needed:
```bash
cp .env.example .env
# Edit .env to change:
# - OLLAMA_MODEL (llama2, mistral, etc.)
# - MILVUS_HOST/PORT (for vector database)
```

## 🎯 Features to Try

### 1. Information Queries
```
You: Where is downtown parking?
You: What are the parking prices?
You: How do I make a reservation?
```

### 2. Reservation Process
```
You: I want to book a parking space
```
(Follow the prompts to provide your details)

### 3. Check Parking Status
```
You: parking list
You: parking info downtown_1
```

### 4. Safety Testing
```
You: My credit card is 1234-5678-9012-3456
```
(Should be blocked)

### 5. Run Evaluation
```
You: evaluate
```
(Generates detailed performance report)

## 🔧 Troubleshooting

### "Connection refused" error
**Problem**: Ollama not running
```bash
# In a new terminal:
ollama serve
```

### "Model not found" error
**Problem**: Model not downloaded
```bash
# Download the model
ollama pull llama2
# Then try again
```

### Slow responses
**Problem**: Model is too large for your hardware
```bash
# Use faster model
ollama pull mistral
# Edit .env: OLLAMA_MODEL=mistral
```

### Out of memory
**Problem**: GPU/RAM issues
```bash
# Use CPU instead and reduce batch size
# Or use smaller model (mistral instead of llama2)
```

## 📚 Next Steps

1. **Explore the Code**:
   - `src/rag/` - RAG pipeline
   - `src/agents/` - Workflow logic
   - `src/guardrails/` - Safety features
   - `src/database/` - Data management

2. **Customize**:
   - Add your own parking data to `src/evaluation/test_data.py`
   - Modify prompts in `src/agents/workflow.py`
   - Adjust safety rules in `src/guardrails/filter.py`

3. **Add Milvus** (Optional):
   ```bash
   # Docker:
   docker run -d --name milvus \
     -p 19530:19530 \
     -p 9091:9091 \
     milvusdb/milvus:latest

   # Then: python main.py (will use Milvus automatically)
   ```

4. **Integrate with External Services**:
   - Connect real parking database
   - Add payment integration
   - Connect to admin approval system
   - Add SMS/email notifications

## 📊 Understanding Output

### Information Response
```
Bot: Downtown Parking is located at 123 Main Street...

Sources:
  [1] Downtown Parking location information...
```

### Blocked Message
```
Bot: ⚠️  BLOCKED: Message contains sensitive data: credit_card
```

### Reservation Confirmation
```
Bot: Your reservation RES_ABC123 is pending admin approval.
     You will be notified once it's reviewed.
```

## 🎓 Learning More

- See `README.md` for full documentation
- Check `src/evaluation/test_data.py` for test queries
- Review `src/agents/workflow.py` for conversation flow
- Look at `src/guardrails/filter.py` for security rules

## 📈 Performance Tips

- **Faster responses**: Use `mistral` or `neural-chat` model
- **Better accuracy**: Use `llama2-uncensored` or `dolphin-mixtral`
- **Lower memory**: Use `mistral` or `orca-mini`
- **Best balance**: Default `llama2`

## 🆘 Getting Help

1. Check existing terminal output for error messages
2. Ensure Ollama is running: `ollama serve`
3. Verify model is downloaded: `ollama list`
4. Check configuration in `.env`
5. Review logs in application output

## 🎉 You're Ready!

```bash
python main.py
```

Type `help` for available commands. Enjoy!

---

**Version**: 0.1.0
**Last Updated**: 2024
**Status**: Ready for Development
