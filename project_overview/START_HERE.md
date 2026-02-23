# 🚀 START HERE

You just cloned the project? Follow these steps to get it running.

## ⚡ 30-Second Quick Start

### macOS/Linux
```bash
bash install-and-run.sh
```

### Windows
```bash
setup.bat
run.bat
```

That's it! The script does everything.

---

## ⚠️ Important: Ollama Must Be Running

**Before** running the chatbot, start Ollama in **another terminal**:

```bash
# Terminal 1: Start Ollama (keep it running)
ollama serve

# Terminal 2: Pull a model (first time only)
ollama pull llama2

# Terminal 3: Run the setup script
bash install-and-run.sh  # or setup.bat on Windows
```

---

## ✅ Step-by-Step (Most Clear)

### macOS/Linux

**Terminal 1: Start Ollama**
```bash
ollama serve
```

**Terminal 2: Set up the project**
```bash
cd ai_task
bash setup.sh
bash run.sh
```

### Windows

**Terminal 1: Start Ollama**
```bash
ollama serve
```

**Terminal 2: Set up the project**
```bash
cd ai_task
setup.bat
run.bat
```

---

## 🎯 What Happens When You Run the Script

```
✓ Creates virtual environment
✓ Installs dependencies (langchain, ollama, etc.)
✓ Sets up configuration file
✓ Creates data directories
✓ Checks Ollama is running
✓ Launches the chatbot
```

---

## 🎮 Once It's Running

You'll see:

```
================================================================================
                PARKING CHATBOT - Powered by Ollama & LangChain
================================================================================
Type 'help' for available commands, 'quit' to exit

You:
```

Try these:

```
You: What are parking prices?
Bot: [RAG-retrieved answer]

You: I want to book a parking space
Bot: [Starts collection process]

You: parking list
Bot: [Shows all parking spaces]

You: evaluate
Bot: [Runs evaluation tests]

You: help
Bot: [Shows available commands]

You: quit
```

---

## 🐛 Something Not Working?

### "Ollama is not running"
```bash
# Start Ollama in a SEPARATE terminal
ollama serve
```

### "Permission denied" (macOS/Linux)
```bash
chmod +x setup.sh run.sh install-and-run.sh
bash install-and-run.sh
```

### "ModuleNotFoundError"
```bash
# Activate the virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Then install
pip install -r requirements.txt
```

### "Python 3.10+ required"
Install Python from https://www.python.org (get 3.13)

---

## 📚 Documentation

After it's working, read these:

1. **GETTING_STARTED.md** - Choose your setup method
2. **SETUP_GUIDE.md** - Detailed setup & troubleshooting
3. **README.md** - Full project documentation
4. **QUICKSTART.md** - 5-minute overview

---

## 🆘 Need More Help?

1. Check `SETUP_GUIDE.md` for troubleshooting
2. Verify Ollama is running: `ollama serve` in another terminal
3. Make sure Python 3.10+ is installed
4. Check the scripts are executable: `ls -la setup.sh`

---

## 📋 Scripts You Have

| What | macOS/Linux | Windows |
|------|------------|---------|
| Setup only | `setup.sh` | `setup.bat` |
| Setup + Run | `install-and-run.sh` | setup.bat + run.bat |
| Run only | `run.sh` | `run.bat` |
| Setup with UV | `setup-with-uv.sh` | - |

---

## ✨ Key Files

- `main.py` - Chatbot entry point
- `demo.py` - Demo script showing all features
- `.env.example` - Configuration template
- `requirements.txt` - Python dependencies

---

## 🎓 Next Steps

1. ✅ Run setup script
2. ✅ Try some queries
3. 📖 Read `README.md`
4. 🧪 Run `python demo.py`
5. 💻 Explore `src/` directory

---

## 🎉 You're Ready!

```bash
# One command to get started
bash install-and-run.sh
```

Enjoy the chatbot! 🚀

