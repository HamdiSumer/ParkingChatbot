# Setup Scripts - Complete Summary

All available setup and run scripts for the Parking Chatbot project.

## 📋 Quick Reference

| OS | Setup | Run |
|-----|--------|-----|
| **macOS/Linux (All-in-One)** | `bash install-and-run.sh` | (included in setup) |
| **macOS/Linux (Separate)** | `bash setup.sh` | `bash run.sh` |
| **macOS/Linux (With UV)** | `bash setup-with-uv.sh` | `bash run.sh` |
| **Windows** | `setup.bat` | `run.bat` |

---

## 🎯 Choose Your Path

### **Path 1: All-in-One (Easiest)**
Best if you want everything done in one script.

**macOS/Linux:**
```bash
bash install-and-run.sh
```

What it does:
1. Creates virtual environment
2. Installs all dependencies
3. Configures environment
4. Checks Ollama connection
5. Launches the chatbot

**Result:** Chatbot running (if Ollama is available)

---

### **Path 2: Separate Setup & Run (Most Flexible)**
Best if you want to setup once, run multiple times.

**macOS/Linux:**
```bash
# First time: setup
bash setup.sh

# Verify it worked
source .venv/bin/activate
python -c "import langchain; print('✓ OK')"

# Later: just run
bash run.sh
```

**Windows:**
```batch
REM First time: setup
setup.bat

REM Later: just run
run.bat
```

What setup.sh/setup.bat does:
- Creates `.venv` directory
- Installs dependencies
- Sets up `.env` file
- Creates `data/` and `reports/` directories

What run.sh/run.bat does:
- Activates virtual environment
- Checks Ollama is running
- Launches Python chatbot

---

### **Path 3: With UV (Fastest)**
Best if you have UV installed (blazing fast).

**macOS/Linux:**
```bash
bash setup-with-uv.sh
bash run.sh
```

What it does:
- Uses UV instead of pip (10x faster)
- Everything else same as setup.sh

**Install UV** if needed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### **Path 4: Manual (Full Control)**
Best if you want to understand each step.

See `SETUP_GUIDE.md` for step-by-step instructions.

---

## 📝 Script Details

### **install-and-run.sh** (macOS/Linux)
**File size:** 2.4K
**Purpose:** Complete setup + immediate launch
**Time:** 3-5 minutes
**Prerequisites:** Python 3.10+, Ollama

**What it does:**
```
1. Calls setup.sh
   - Creates venv
   - Installs dependencies
   - Sets up .env
   - Creates directories

2. Activates venv

3. Checks Ollama connection
   - Fails gracefully if Ollama not running

4. Launches chatbot with python main.py
```

**Usage:**
```bash
bash install-and-run.sh
```

**Exit codes:**
- `0` = Success (chatbot running)
- `1` = Setup failed or Ollama not running

---

### **setup.sh** (macOS/Linux)
**File size:** 3.4K
**Purpose:** Setup only (don't run yet)
**Time:** 2-3 minutes
**Prerequisites:** Python 3.10+

**What it does:**
```
[1/6] Check Python version
[2/6] Create virtual environment (.venv)
[3/6] Activate virtual environment
[4/6] Install dependencies (pip install -r requirements.txt)
[5/6] Create .env file (from .env.example)
[6/6] Create data/ and reports/ directories
```

**Usage:**
```bash
bash setup.sh
```

**After setup, run with:**
```bash
bash run.sh
```

---

### **setup-with-uv.sh** (macOS/Linux)
**File size:** 3.9K
**Purpose:** Setup with modern UV package manager
**Time:** 1-2 minutes (much faster!)
**Prerequisites:** Python 3.10+, UV installed

**What it does:**
- Same as setup.sh but uses `uv pip` instead of `pip`
- 10x faster dependency installation
- Same result: ready to run

**Usage:**
```bash
# Install UV first (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then setup
bash setup-with-uv.sh
bash run.sh
```

---

### **run.sh** (macOS/Linux)
**File size:** 2.1K
**Purpose:** Run the chatbot
**Time:** Instant (if setup already done)
**Prerequisites:** setup.sh already run, Ollama running

**What it does:**
```
1. Activate virtual environment
2. Check Ollama is running (curl to localhost:11434)
3. Launch Python chatbot

Optional arguments:
  bash run.sh demo      # Run demo instead of interactive
  bash run.sh evaluate  # Run with evaluation mode
```

**Usage:**
```bash
# Run interactive chatbot
bash run.sh

# Run demo
bash run.sh demo

# Run evaluation
bash run.sh evaluate
```

**Error handling:**
- Fails if venv doesn't exist
- Fails if Ollama not running
- Shows helpful error messages

---

### **setup.bat** (Windows)
**File size:** 3.4K
**Purpose:** Setup only
**Time:** 2-3 minutes
**Prerequisites:** Python 3.10+ in PATH

**What it does:**
```
[1/6] Check Python version
[2/6] Create virtual environment (.venv)
[3/6] Activate virtual environment
[4/6] Install dependencies (pip install -r requirements.txt)
[5/6] Create .env file (from .env.example)
[6/6] Create data/ and reports/ directories
```

**Usage:**
```batch
setup.bat
```

**After setup, run with:**
```batch
run.bat
```

**Features:**
- Colored output for readability
- Pauses at end to show messages
- Clear error messages
- Uses Python's standard venv module

---

### **run.bat** (Windows)
**File size:** 2.3K
**Purpose:** Run the chatbot
**Time:** Instant (if setup already done)
**Prerequisites:** setup.bat already run, Ollama running

**What it does:**
```
1. Activate virtual environment
2. Check Ollama is running (PowerShell curl)
3. Launch Python chatbot

Optional arguments:
  run.bat demo      # Run demo instead of interactive
  run.bat evaluate  # Run with evaluation mode
```

**Usage:**
```batch
REM Run interactive chatbot
run.bat

REM Run demo
run.bat demo
```

---

## 🔧 Technical Details

### Environment Variables Checked
- `OLLAMA_HOST` (default: http://localhost:11434)
- `OLLAMA_MODEL` (default: llama2)
- Python version (requires 3.10+)

### Connections Verified
- Ollama API at localhost:11434
- Python version compatibility
- Virtual environment creation

### Directories Created
- `.venv/` - Virtual environment
- `data/` - Data storage (parking.db)
- `reports/` - Evaluation reports

### Files Created/Modified
- `.env` - Configuration (from .env.example)
- `requirements-lock.txt` - (if using UV)

---

## 🚨 Common Issues & Fixes

### Issue: "No such file or directory: setup.sh"
**Cause:** Not in correct directory
**Fix:** `cd ai_task && bash setup.sh`

### Issue: "Permission denied: ./setup.sh"
**Cause:** Script not executable
**Fix:** `chmod +x setup.sh && bash setup.sh`

### Issue: "Ollama is not running"
**Cause:** Ollama server not started
**Fix:** Open another terminal and run `ollama serve`

### Issue: "Python 3.10+ required"
**Cause:** Python version too old
**Fix:** Install Python 3.13 from python.org

### Issue: Script not found on Windows
**Cause:** Using wrong command or missing file extension
**Fix:** Make sure you're in correct directory, use `setup.bat` (with .bat)

---

## 📊 Execution Flow Charts

### install-and-run.sh
```
START
  ↓
setup.sh ────────────────────────┐
  ├─ Check Python               │
  ├─ Create venv                │
  ├─ Install deps               │
  ├─ Setup .env                 │
  └─ Create dirs                │
                                ↓
                          Activate venv
                                ↓
                          Check Ollama
                          ├─ Running? → Continue
                          └─ Not running? → EXIT with message
                                ↓
                          python main.py
                                ↓
                              END
```

### setup.sh → run.sh (Separate)
```
SETUP.SH               RUN.SH (later)
  ├─ Create venv          ├─ Activate venv
  ├─ Install deps         ├─ Check Ollama
  ├─ Setup .env           ├─ Check Ollama running
  └─ Create dirs          └─ python main.py
```

---

## 💡 Pro Tips

### Tip 1: Keep Ollama Running
```bash
# Terminal 1: Let this run continuously
ollama serve

# Terminal 2: Run chatbot as many times as you want
bash run.sh
bash run.sh demo
bash run.sh
```

### Tip 2: Activate Venv Manually
```bash
# After setup.sh, you can manually activate
source .venv/bin/activate

# Then use Python directly
python main.py
python demo.py
```

### Tip 3: Check What Got Installed
```bash
source .venv/bin/activate
pip list
```

### Tip 4: Use Different Models
Edit `.env`:
```env
OLLAMA_MODEL=mistral     # Faster
OLLAMA_MODEL=neural-chat # Better chat
```

### Tip 5: UV is Worth It
If you'll setup multiple times:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
bash setup-with-uv.sh  # Will be 10x faster
```

---

## 📚 Related Documentation

- `GETTING_STARTED.md` - Quick start guide
- `SETUP_GUIDE.md` - Detailed setup instructions
- `README.md` - Full project documentation
- `QUICKSTART.md` - 5-minute quick start
- `PROJECT_OVERVIEW.md` - Architecture details

---

## ✅ Verification

After running setup, verify everything works:

```bash
# Activate venv
source .venv/bin/activate

# Check dependencies
python -c "import langchain; import ollama; print('✓ All good!')"

# Check if Ollama is running
curl http://localhost:11434/api/tags

# Run the app
python main.py
```

---

## 🎯 Decision Tree

```
┌─ I just cloned the project
├─ Do I have UV installed?
│  ├─ Yes → bash setup-with-uv.sh
│  └─ No  → bash setup.sh or bash install-and-run.sh
│
├─ Do I want everything in one script?
│  ├─ Yes (macOS/Linux) → bash install-and-run.sh
│  └─ No  → bash setup.sh, then bash run.sh later
│
└─ Am I on Windows?
   ├─ Yes → setup.bat, then run.bat
   └─ No (macOS/Linux) → Use .sh scripts above
```

---

**Version:** 0.1.0
**Last Updated:** 2024
**Status:** All scripts tested and ready
