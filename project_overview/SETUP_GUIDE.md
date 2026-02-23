# Setup & Installation Guide

Complete setup instructions for the Parking Chatbot project.

## 📋 Prerequisites

- **Python 3.10+** (Check with `python3 --version`)
- **Ollama** (Download from https://ollama.ai)
- **Git** (for cloning the repository)

## 🚀 Quick Start (Recommended)

### **For macOS/Linux Users**

```bash
# Clone the repository
git clone <repository-url>
cd ai_task

# Run the complete setup script (does everything)
bash install-and-run.sh
```

That's it! The script will:
1. ✅ Create virtual environment
2. ✅ Install all dependencies
3. ✅ Set up configuration files
4. ✅ Check for Ollama
5. ✅ Launch the chatbot

### **For Windows Users**

```bash
# Clone the repository
git clone <repository-url>
cd ai_task

# Run the setup script
setup.bat
```

Then run:
```bash
run.bat
```

---

## 📦 Step-by-Step Installation

### **Option 1: Automated Setup (Easiest)**

#### macOS/Linux
```bash
# Everything in one command
bash install-and-run.sh
```

#### Windows
```bash
# Run setup
setup.bat

# Then run the app
run.bat
```

---

### **Option 2: Manual Setup (More Control)**

#### macOS/Linux

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional)
cp .env.example .env

# 5. Start Ollama (in another terminal)
ollama serve

# 6. Pull a model (in yet another terminal)
ollama pull llama2

# 7. Run the app
python main.py
```

#### Windows

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional)
copy .env.example .env

# 5. Start Ollama (in another terminal)
ollama serve

# 6. Pull a model (in yet another terminal)
ollama pull llama2

# 7. Run the app
python main.py
```

---

## 🔧 Available Scripts

### macOS/Linux

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup.sh` | Initial setup only | `bash setup.sh` |
| `run.sh` | Run the chatbot | `bash run.sh` |
| `install-and-run.sh` | Setup + run | `bash install-and-run.sh` |

### Windows

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup.bat` | Initial setup only | `setup.bat` |
| `run.bat` | Run the chatbot | `run.bat` |

---

## 🎯 First Time Setup Workflow

### What the Scripts Do

```
install-and-run.sh (macOS/Linux) or setup.bat (Windows)
├── [2/6] Create virtual environment
├── [3/6] Activate virtual environment
├── [4/6] Install dependencies (pip install -r requirements.txt)
├── [5/6] Setup .env configuration file
├── [6/6] Create data/ and reports/ directories
└── Ready to launch!

Then:
run.sh or run.bat
├── Activate virtual environment
├── Check Ollama is running
└── Launch Python application
```

---

## ✅ Verification Checklist

After running the setup scripts:

- [ ] Virtual environment created (`.venv` directory exists)
- [ ] Dependencies installed (no pip errors)
- [ ] `.env` file created
- [ ] `data/` directory exists
- [ ] `reports/` directory exists
- [ ] Ollama is running (`ollama serve` in another terminal)
- [ ] Model is available (`ollama pull llama2`)

Verify with:
```bash
# Check Python dependencies
python -c "import langchain; import ollama; print('✓ OK')"

# Check Ollama
curl http://localhost:11434/api/tags
```

---

## 🐛 Troubleshooting

### "Python 3.10+ is required"

**Problem:** Your Python version is too old

**Solution:**
```bash
# Check version
python3 --version

# Install newer Python from https://www.python.org or your package manager
# macOS: brew install python@3.13
# Linux: sudo apt update && sudo apt install python3.13
# Windows: Download from python.org
```

### "Permission denied: ./setup.sh"

**Problem:** Script doesn't have execute permission

**Solution (macOS/Linux):**
```bash
chmod +x setup.sh run.sh install-and-run.sh
bash install-and-run.sh
```

### "Ollama is not running"

**Problem:** The chatbot can't connect to Ollama

**Solution:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull a model
ollama pull llama2

# Terminal 3: Run the chatbot
bash run.sh  # or run.bat on Windows
```

### "ModuleNotFoundError: No module named 'langchain'"

**Problem:** Dependencies not installed

**Solution:**
```bash
# Activate venv first
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### "pip: externally-managed-environment"

**Problem:** System Python vs virtual environment

**Solution:**
```bash
# Make sure venv is activated (should show in prompt)
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate  # Windows

# Then install
pip install -r requirements.txt
```

### Virtual environment not activating

**Problem:** Wrong activation command

**Solution (macOS/Linux):**
```bash
# Correct for .venv directory:
source .venv/bin/activate

# Check it worked (prompt should show (.venv)):
echo $VIRTUAL_ENV

# Deactivate when done:
deactivate
```

**Solution (Windows):**
```bash
# Correct command:
.venv\Scripts\activate

# Deactivate when done:
deactivate
```

### "Ollama not found" or "command not found: ollama"

**Problem:** Ollama not installed

**Solution:**
```bash
# Download and install from https://ollama.ai
# macOS: Download DMG from website
# Linux: curl https://ollama.ai/install.sh | sh
# Windows: Download installer from website

# Then start it:
ollama serve
```

---

## 📁 Directory Structure After Setup

```
ai_task/
├── .venv/                  # Virtual environment (created by setup)
├── data/                   # Data directory (created by setup)
│   └── parking.db         # SQLite database (created on first run)
├── reports/               # Reports directory (created by setup)
│   ├── evaluation_report.md
│   └── evaluation_results.json
├── src/                   # Source code
├── main.py               # Entry point
├── .env                  # Configuration (created from .env.example)
├── requirements.txt      # Dependencies
├── setup.sh/setup.bat    # Setup script
├── run.sh/run.bat        # Run script
└── README.md            # Full documentation
```

---

## 🔄 Rerunning After First Setup

After initial setup, you only need to:

1. **Make sure Ollama is running** (in a separate terminal):
   ```bash
   ollama serve
   ```

2. **Run the chatbot**:
   - **macOS/Linux**: `bash run.sh`
   - **Windows**: `run.bat`

That's it! No need to run setup again.

---

## 🌐 Using Different Ollama Models

Edit `.env` to change the model:

```env
OLLAMA_MODEL=llama2        # Default, balanced
OLLAMA_MODEL=mistral       # Faster
OLLAMA_MODEL=neural-chat   # Conversational
```

Then restart the chatbot.

---

## 📚 Next Steps

1. ✅ Run setup: `bash setup.sh` or `setup.bat`
2. ✅ Start Ollama: `ollama serve`
3. ✅ Run chatbot: `bash run.sh` or `run.bat`
4. 📖 Read documentation:
   - `README.md` - Full documentation
   - `QUICKSTART.md` - Quick reference
   - `PROJECT_OVERVIEW.md` - Architecture details

---

## 💡 Tips & Tricks

### Run demo instead of interactive mode
```bash
source .venv/bin/activate  # macOS/Linux
python demo.py
```

### Test without Milvus
The system works without Milvus installed:
```bash
# Just needs Ollama
bash run.sh
```

### Check what's installed
```bash
source .venv/bin/activate  # macOS/Linux
pip list
```

### Upgrade dependencies
```bash
source .venv/bin/activate  # macOS/Linux
pip install --upgrade -r requirements.txt
```

### Remove virtual environment and start fresh
```bash
rm -rf .venv  # macOS/Linux
# OR
rmdir /s .venv  # Windows

bash setup.sh  # macOS/Linux
# OR
setup.bat  # Windows
```

---

## 🎓 Learning Path

1. **Run the chatbot**: `bash run.sh`
2. **Try some queries**: "What are parking prices?"
3. **Try a reservation**: "I want to book a space"
4. **Run demo**: `python demo.py`
5. **Read the code**: Start with `src/app.py`
6. **Customize**: Modify `.env` and `src/` files

---

## 📞 Support

If you encounter issues:

1. Check this troubleshooting section above
2. Verify prerequisites are installed
3. Check that Ollama is running
4. Review output messages (they usually explain the issue)
5. See `README.md` for more detailed documentation

---

**Version**: 0.1.0
**Last Updated**: 2024
**Status**: Ready for use
