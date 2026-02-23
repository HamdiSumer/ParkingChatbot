# How to Test the System

Quick reference for testing the parking chatbot.

## 1️⃣ Fastest Way to Understand Everything

```bash
uv run python test_system.py
```

**What you'll see:**
- Guard rails demo (blocks sensitive data)
- Vector DB explanation (what's stored in Milvus)
- SQL DB explanation (what's stored in SQLite)
- RAG workflow explanation
- End-to-end workflow demo
- Data separation architecture
- Performance metrics that are collected

**Time:** 2 minutes | **Prerequisites:** None

---

## 2️⃣ Full Interactive Testing

```bash
# Start infrastructure
docker-compose up -d

# Configure
cp .env.example .env
# Edit .env to choose your LLM

# Install
uv init
uv install

# Run
uv run python main.py
```

**Try these commands in the chatbot:**

```
You: What parking spaces are available?
# → Retrieves from Milvus (Vector DB)

You: Where is downtown parking?
# → RAG retrieval from static data

You: I want to book a parking space
# → Starts reservation with data collection

You: My credit card is 4532-1234-5678-9012
# → Guard rails blocks it

You: parking list
# → Shows from SQLite (Dynamic data)

You: evaluate
# → Runs performance evaluation
```

**Time:** 5 minutes | **Prerequisites:** Docker, Milvus

---

## 3️⃣ Automated Demo with Evaluation

```bash
# Start Milvus (optional for full metrics)
docker-compose up -d

# Run demo
uv run python demo.py
```

**What happens:**
1. Shows information retrieval
2. Demonstrates safety filtering
3. Shows parking management
4. Runs evaluation tests
5. Generates performance report

**Output saved to:**
```
reports/demo_evaluation_report.md
reports/demo_evaluation_results.json
```

**Time:** 3-5 minutes | **Prerequisites:** LLM configured

---

## 4️⃣ Run Evaluation in Chatbot

```bash
uv run python main.py
```

Then in the chatbot:
```
You: evaluate
```

**What gets tested:**
- RAG system (Recall@K, Precision@K, MRR, NDCG)
- Safety system (block rate, precision, recall)
- Performance (latency, success rate)
- Reservations (accuracy, completion)

**Results saved to:**
```
reports/evaluation_report.md
reports/evaluation_results.json
```

---

## Understanding What Gets Tested

### Guard Rails (Safety Filter)

**Test:** `uv run python test_system.py`

Shows inputs that should be **blocked**:
```
❌ "My credit card is 4532-1234-5678-9012" → BLOCKED
❌ "Call me at 555-123-4567" → BLOCKED
❌ "DELETE all users" → BLOCKED
✅ "What are parking prices?" → ALLOWED
```

---

### Vector Database (Milvus) - Static Data

**Test:** `uv run python test_system.py`

Shows what's stored:
```
Document 1: Downtown Parking location info
Document 2: Pricing information
Document 3: Booking process guide
...
```

When user asks: "Where is downtown parking?"
- Query gets converted to embedding
- Searched in Milvus
- Similar documents retrieved
- LLM generates answer from docs

**Performance metrics:**
- Retrieval latency (how fast)
- Number of documents found
- Relevance of results

---

### SQL Database (SQLite) - Dynamic Data

**Test:** `uv run python test_system.py`

Shows what's stored:
```
Parking Spaces:
  downtown_1    | 450/500 available | $5/hour
  airport_1     | 800/1000 available | $3/hour

Reservations:
  RES_ABC123    | John Doe | pending → approved
```

When user makes a reservation:
- Availability checked in SQLite
- Reservation record created
- Status updated as admin approves

---

### RAG Performance Metrics

**Test:** `uv run python test_system.py` (or `evaluate` in chatbot)

Metrics collected:

```
Recall@1/3/5
  └─ Of all relevant docs, what % in top K?
  └─ Higher is better

Precision@1/3/5
  └─ Of top K, what % are relevant?
  └─ Higher is better

Mean Reciprocal Rank
  └─ Position of first relevant doc
  └─ Closer to 1.0 is better

Retrieval Latency
  └─ How fast is the vector search?
  └─ Lower is better (ms)
```

---

## Data Flow Verification

### When User Asks a Question

```
User: "Where is downtown parking?"
       ↓
[Safety Check] → Safe, continue
       ↓
[RAG System]
  ├─ Convert question to embedding
  ├─ Search Milvus vector DB
  ├─ Retrieve similar documents
  └─ Pass to LLM
       ↓
[LLM Generation]
  └─ "Downtown Parking is at 123 Main Street"
       ↓
[Response Filter]
  └─ Check for sensitive data exposure
       ↓
User sees answer
```

**Data sources:**
- Query: From user input
- Documents: From Milvus (Vector DB)
- Answer: From LLM

---

### When User Books a Parking Space

```
User: "I want to book downtown_1 for tomorrow"
       ↓
[Safety Check] → Safe, continue
       ↓
[Intent Detection] → "Reservation"
       ↓
[Data Collection]
  ├─ Name?
  ├─ Car number?
  └─ Time period?
       ↓
[SQLite Check]
  ├─ Is space available?
  ├─ Get current pricing
  └─ Get working hours
       ↓
[Create Reservation] → Insert into SQLite
       ↓
[Admin Review Workflow]
  ├─ Pending admin approval
  └─ Update status when approved
```

**Data sources:**
- User input: Interactive prompts
- Availability: SQLite (parking_spaces table)
- Reservation: SQLite (reservations table)

---

## Verifying Each Component

| Component | How to Test | Expected Result |
|-----------|------------|-----------------|
| **Guard Rails** | `test_system.py` | Sensitive data blocked ✓ |
| **Vector DB** | `test_system.py` or `evaluate` | Documents retrieved fast ✓ |
| **SQL DB** | `test_system.py` or `main.py` | Parking spaces listed ✓ |
| **RAG** | `test_system.py` or `evaluate` | High Recall/Precision ✓ |
| **End-to-End** | `main.py` or `demo.py` | Full workflow works ✓ |

---

## Checklist

- [ ] Run `test_system.py` once to understand the system
- [ ] Start Milvus: `docker-compose up -d`
- [ ] Run chatbot: `uv run python main.py`
- [ ] Try information queries
- [ ] Try sensitive data (should be blocked)
- [ ] Try making a reservation
- [ ] Type `evaluate` to see performance metrics
- [ ] Check reports in `reports/` folder
- [ ] View SQLite data: `sqlite3 data/parking.db`

---

## Troubleshooting Tests

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | `uv sync` |
| `Connection refused` | `docker-compose up -d` |
| Tests show no output | Check terminal encoding, run with `2>&1` |
| RAG tests fail | Start Milvus first: `docker-compose up -d` |

---

## Next Steps

1. **Understand the system**: `uv run python test_system.py`
2. **Set up**: Follow `INSTALLATION.md`
3. **Test interactively**: `uv run python main.py`
4. **Review results**: Check `reports/` folder
5. **Explore code**: Look at `src/` directory

---

For detailed information, see [TESTING_GUIDE.md](TESTING_GUIDE.md)
