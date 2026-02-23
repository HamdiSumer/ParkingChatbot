# Getting Started - Choose Your Path

Quick guide to get the parking chatbot running. Choose based on your situation.

## 🎯 I Just Cloned the Project - What Do I Do?

### **Scenario 1: I Want the Easiest Setup (Recommended)**

**macOS/Linux:**
```bash
bash install-and-run.sh
```

**Windows:**
```bash
setup.bat
run.bat
```

This script does **everything**:
- ✅ Creates virtual environment
- ✅ Installs dependencies
- ✅ Sets up configuration
- ✅ Checks for Ollama
- ✅ Launches the app

---

### **Scenario 2: I Have UV Installed (Fastest)**

UV is a modern, faster package manager. If you have it:

```bash
bash setup-with-uv.sh
bash run.sh
```

**Install UV** if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### **Scenario 3: I Want to Set Up and Run Later**

**macOS/Linux:**
```bash
# Just setup
bash setup.sh

# ... later, when ready to run:
bash run.sh
```

**Windows:**
```batch
REM Just setup
setup.bat

REM ... later, when ready to run:
run.bat
```

---

### **Scenario 4: I Want Manual Control**

Follow the step-by-step guide in `SETUP_GUIDE.md`.

---

## 📋 Available Scripts

### macOS/Linux

| Script | What It Does | When to Use |
|--------|--------------|------------|
| **install-and-run.sh** | Setup + Run | ✅ First time, easiest |
| **setup.sh** | Just setup | When you want to setup later |
| **setup-with-uv.sh** | Setup with UV | You have UV installed |
| **run.sh** | Just run | After setup is done |

### Windows

| Script | What It Does | When to Use |
|--------|--------------|------------|
| **setup.bat** | Just setup | ✅ First time |
| **run.bat** | Just run | After setup is done |

---

## 🚀 Quick Start - Copy Paste

### macOS/Linux (All-in-One)
```bash
git clone <repo>
cd ai_task
bash install-and-run.sh
```

### Windows (Two Steps)
```bash
git clone <repo>
cd ai_task
setup.bat
run.bat
```

---

## ⚠️ Important: Ollama Must Be Running

Before running the chatbot, **Ollama must be running in another terminal**:

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull a model (first time only)
ollama pull llama2

# Terminal 3: Run the chatbot
bash run.sh  # or run.bat on Windows
```

---

## 🔄 Daily Usage (After Setup)

### You've already done setup, now just want to use it?

**macOS/Linux:**
```bash
# Terminal 1
ollama serve

# Terminal 2
bash run.sh
```

**Windows:**
```bash
REM Terminal 1
ollama serve

REM Terminal 2
run.bat
```

That's it!

---

## 🐛 Something Went Wrong?

### 1. "ModuleNotFoundError"
```bash
# Make sure venv is activated
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Then install again
pip install -r requirements.txt
```

### 2. "Ollama is not running"
```bash
# Start Ollama in another terminal
ollama serve
```

### 3. "Permission denied" (macOS/Linux)
```bash
chmod +x setup.sh run.sh
bash setup.sh
```

### 4. Still stuck?
See `SETUP_GUIDE.md` for detailed troubleshooting.

---

## 📚 What's Next?

1. **Run the chatbot**: `bash run.sh` (or `run.bat`)
2. **Try the demo**: `python demo.py`
3. **Read the docs**: `README.md`
4. **Understand the code**: Check `PROJECT_OVERVIEW.md`

---

## 📁 Scripts Overview

### **install-and-run.sh** (macOS/Linux)
```
┌─ setup.sh
│  ├─ Create venv
│  ├─ Install dependencies
│  ├─ Setup .env
│  └─ Create directories
│
└─ run.sh
   ├─ Check Ollama
   └─ Launch app
```

### **setup.bat** (Windows)
```
├─ Check Python
├─ Create venv
├─ Install dependencies
├─ Setup .env
└─ Create directories
```

### **run.bat** (Windows)
```
├─ Activate venv
├─ Check Ollama
└─ Launch app
```

---

## ✅ Success Checklist

After running the setup scripts:

- [ ] Venv created (`.venv` directory exists)
- [ ] Dependencies installed (no errors)
- [ ] `.env` file created
- [ ] Directories created (`data/`, `reports/`)
- [ ] Ollama is running separately
- [ ] Chatbot launches without errors

---

## 🆘 Still Having Issues?

1. Check you're in the right directory: `ls setup.sh`
2. On macOS/Linux, scripts need to be executable: `chmod +x *.sh`
3. Make sure Ollama is running: `ollama serve`
4. Detailed help: See `SETUP_GUIDE.md`

---

**Next Step:** Run your chosen script and enjoy the chatbot! 🎉

