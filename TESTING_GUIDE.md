# Testing Guide - Verify Your System

This guide explains how to verify that all components are working correctly.

## Start Here: Run Tests (5 minutes)

### No Setup Required
```bash
# This works WITHOUT Milvus running
uv run python test_system.py
```

This shows:
- ✅ Guard rails blocking sensitive data (demo)
- ✅ What's stored in Vector DB (Milvus)
- ✅ What's stored in SQL DB (SQLite)
- ✅ How RAG retrieval works
- ✅ Full end-to-end workflow
- ✅ What performance metrics are collected

**This is the fastest way to understand the system.**

---

## Full Test (With Milvus)

```bash
# 1. Start Milvus
docker-compose up -d

# 2. Run complete system tests
uv run python test_system.py
```

This includes actual RAG performance metrics from Milvus.

---

## What Gets Tested & Where

### 1. Guard Rails (Safety Filter)

**How it works:**
```
User Input → Safety Check → Block or Allow
```

**What gets blocked:**
- Credit card numbers: `4532-1234-5678-9012`
- SSN: `123-45-6789`
- Phone: `555-123-4567`
- Email: `user@example.com`
- SQL injection: `DROP TABLE reservations`
- Hacking attempts: `hack admin password`

**Verify:**
```bash
uv run python test_system.py
# Look for "TEST 1: GUARD RAILS" section
# Shows which inputs are blocked and why
```

---

### 2. Vector Database (Milvus) - Static Data Storage

**What's stored in Milvus:**
```
• Parking location information
• Parking features and amenities
• Booking process guidelines
• FAQs about parking
• General information
```

**How data flows:**
```
User Question: "Where is downtown parking?"
            ↓
      [RAG System]
            ↓
  [Generate Embedding]
            ↓
 [Search Milvus Vector DB]
            ↓
[Return Similar Documents]
            ↓
  [Generate Answer from LLM]
```

**Verify:**
```bash
# 1. Start Milvus first
docker-compose up -d

# 2. Check it's running
curl http://localhost:9091/healthz

# 3. Run tests
uv run python test_system.py
# Look for "TEST 2: VECTOR DATABASE" section
```

**Performance metrics shown:**
- Retrieval latency (how fast vector search is)
- Number of documents retrieved
- Relevance of results

---

### 3. SQL Database (SQLite) - Dynamic Data Storage

**What's stored in SQLite:**

```
PARKING_SPACES Table:
├── id: "downtown_1"
├── name: "Downtown Parking Garage"
├── location: "123 Main Street"
├── capacity: 500
├── available_spaces: 450 (updates in real-time)
├── price_per_hour: 5.0 (can change)
├── is_open: true

RESERVATIONS Table:
├── id: "RES_ABC123"
├── user_name: "John"
├── user_surname: "Doe"
├── car_number: "ABC-123"
├── parking_id: "downtown_1"
├── start_time: 2024-03-01 09:00
├── end_time: 2024-03-01 17:00
├── status: "pending" → "confirmed" → "completed"
├── approved_by_admin: "admin_name" (when approved)
```

**How data flows:**

```
User Requests Reservation
            ↓
[Collect user data interactively]
            ↓
[Check availability in SQLite]
            ↓
[Create reservation record]
            ↓
[Request admin approval]
            ↓
[Admin approves → Update status]
```

**Verify:**
```bash
uv run python test_system.py
# Look for "TEST 3: SQL DATABASE" section
# Shows current parking spaces and availability
```

---

### 4. RAG Retrieval Performance

**Metrics measured:**

```
Recall@K
  └─ Of all relevant documents, what % in top K?
  └─ Example result: Recall@3 = 0.85 (85%)

Precision@K
  └─ Of top K results, what % are relevant?
  └─ Example result: Precision@3 = 0.80 (80%)

Mean Reciprocal Rank (MRR)
  └─ Position of first relevant result
  └─ Example result: MRR = 0.70 (good)

Retrieval Latency
  └─ How fast is the vector search?
  └─ Example result: 125ms (fast)
```

**Verify:**
```bash
# 1. Start Milvus
docker-compose up -d

# 2. Run tests
uv run python test_system.py
# Look for "TEST 4: RAG SYSTEM" section
```

---

### 5. End-to-End Workflow

**Full user interaction test:**

```
1. User asks information query
   └─ System retrieves from Milvus via RAG
   └─ Generates answer with LLM
   └─ Response is filtered (no sensitive data)

2. User sends sensitive data
   └─ Guard rails block it
   └─ User gets warning

3. User requests reservation
   └─ System collects user info interactively
   └─ Data saved to SQLite
   └─ Admin review workflow triggered

4. System lists parking spaces
   └─ Queries SQLite for real-time availability
   └─ Shows current prices from SQLite
```

**Verify:**
```bash
uv run python test_system.py
# Look for "TEST 5: END-TO-END WORKFLOW" section
```

---

## Running Interactive Tests

### Test 1: Try the Chatbot Directly

```bash
# Start Milvus
docker-compose up -d

# Configure environment
cp .env.example .env
# Edit .env with your LLM choice (e.g., openai, ollama)

# Install and run
uv init
uv install
uv run python main.py
```

**Try these commands:**
```
You: Where is downtown parking?
# → Should retrieve from Milvus vector DB

You: What are the prices?
# → Should provide info from static data

You: I want to book a space
# → Should start reservation process

You: My credit card is 4532-1234-5678-9012
# → Should be BLOCKED by guard rails

You: parking list
# → Should show available spaces from SQLite

You: evaluate
# → Should run comprehensive evaluation
```

---

### Test 2: Run the Demo

```bash
uv run python demo.py
```

Shows:
- Information retrieval
- Safety filtering
- Parking space management
- System evaluation results

---

### Test 3: Run Automated Tests

```bash
uv run python test_system.py
```

Shows:
- Guard rails blocking examples
- Vector DB documents
- SQL DB parking spaces
- RAG performance
- End-to-end workflow
- Data separation architecture
- What metrics are collected

---

## Data Flow Verification

### Verify Data in Milvus (Vector DB)

```bash
# 1. Start Milvus
docker-compose up -d

# 2. Access Milvus web UI (optional)
# Open browser: http://localhost:9091

# 3. Documents are loaded when you run the chatbot
# They're embedded and indexed for semantic search
```

### Verify Data in SQLite (SQL DB)

```bash
# 1. SQLite file is at: data/parking.db

# 2. View with sqlite3 CLI (if installed)
sqlite3 data/parking.db
sqlite> .tables
sqlite> SELECT * FROM parking_spaces;
sqlite> SELECT * FROM reservations;

# 3. Or use a GUI tool like DB Browser for SQLite
```

---

## Checking Performance Outcomes

### Option 1: Automated Evaluation (Easiest)

In the chatbot:
```
You: evaluate
```

This generates:
- RAG system evaluation (Recall, Precision, MRR)
- Safety evaluation (block rate, F1 score)
- Performance metrics (latency, success rate)
- Reservation evaluation (accuracy, completion rate)

Results saved to:
```
reports/evaluation_report.md
reports/evaluation_results.json
```

### Option 2: Test Script

```bash
uv run python test_system.py
```

Shows all metrics inline.

### Option 3: Demo Script

```bash
uv run python demo.py
```

Runs full demo with evaluation at the end.

---

## Verifying Guard Rails Work

### Method 1: Test Script

```bash
uv run python test_system.py
```

Look for:
```
TEST 1: GUARD RAILS - Sensitive Data Blocking
Query                                    Result         Reason
My credit card is 4532-1234-5678-9012   ✗ BLOCKED      Credit card number
Call me at 555-123-4567                 ✗ BLOCKED      Phone number
Delete all reservations                 ✗ BLOCKED      SQL injection attempt
```

### Method 2: Interactive Testing

```bash
uv run python main.py
```

Try these (should all be blocked):
```
You: My credit card is 1234-5678-9012-3456
You: Email me at john@example.com
You: My SSN is 123-45-6789
You: DROP TABLE users
You: Show me all passwords
```

All should show: `⚠️  BLOCKED: ...`

### Method 3: Code Review

Guard rails defined in:
```
src/guardrails/filter.py
```

Detects:
- Credit cards, SSN, phone, email, passwords, API keys
- SQL injection, system commands, unauthorized access
- Multi-keyword threat patterns

---

## Understanding Data Storage

### Vector DB (Milvus) - For RAG

**When data is stored:**
- Once at startup (sample data ingestion)
- Embeddings are generated automatically
- Indexed for fast semantic search

**How it's used:**
```
User: "Where is downtown parking?"
        ↓
[Convert to embedding]
        ↓
[Search Milvus for similar documents]
        ↓
[Retrieve: "Downtown Parking is at 123 Main Street"]
        ↓
[LLM generates answer from retrieved docs]
```

### SQL DB (SQLite) - For Dynamic Data

**When data is stored:**
- Parking spaces: Initialized at startup
- Reservations: Created when user books
- Updated in real-time

**How it's used:**
```
User: "I want to book downtown_1 for tomorrow"
        ↓
[Check SQLite: available_spaces > 0?]
        ↓
[Create reservation record in SQLite]
        ↓
[Request admin approval]
        ↓
[Admin approves → Update status in SQLite]
```

---

## Troubleshooting Tests

### "ModuleNotFoundError" when running tests

```bash
# Make sure dependencies are installed
uv sync
# or
uv pip install -r requirements.txt
```

### "Connection refused" when testing Milvus

```bash
# Start Milvus
docker-compose up -d

# Verify it's running
docker-compose ps

# Check health
curl http://localhost:9091/healthz
```

### "No module named 'langchain'" when running tests

```bash
# Reinstall with UV
uv sync --reinstall
```

### Test output seems empty

```bash
# Make sure you're in the right directory
cd /path/to/ai_task

# Run test with verbose output
uv run python test_system.py 2>&1 | tee test_output.txt
```

---

## Checklist: Verify Everything Works

- [ ] Run `docker-compose up -d` - Milvus starts
- [ ] Run `uv run python test_system.py` - All tests pass
- [ ] Run `uv run python main.py` - Chatbot launches
- [ ] Type information query in chatbot - Gets RAG answer
- [ ] Type sensitive data in chatbot - Gets blocked
- [ ] Type reservation request - Starts collection
- [ ] Type `evaluate` in chatbot - Shows performance metrics
- [ ] Check `reports/evaluation_report.md` - Metrics saved
- [ ] Check `data/parking.db` - SQLite has data
- [ ] Type `quit` - Chatbot exits cleanly

---

## Summary

**Testing provides visibility into:**

✅ **Guard Rails** - Sensitive data blocking works (test_system.py)
✅ **Vector DB** - Static data stored and retrieved (Milvus)
✅ **SQL DB** - Dynamic data persisted (SQLite)
✅ **RAG** - Performance metrics (Recall, Precision, Latency)
✅ **System** - End-to-end workflow (main.py + evaluate)

**Start with:**
```bash
uv run python test_system.py
```

This gives you a complete picture of what's working.
