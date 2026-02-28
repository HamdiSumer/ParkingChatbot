# Parking Chatbot - Complete Architecture & Data Flow

This document explains how the Parking Chatbot works from start to finish, including all production RAG concepts used.

---

## Quick Overview: What Happens When You Run `python main.py`

```
main.py (entry point)
    ↓
cli.py (interactive CLI)
    ↓
create_app() from app.py (initializes all components)
    ↓
[Embeddings] → [Vector DB] → [RAG Retriever] → [SQL Agent] → [Guardrails]
    ↓
workflow.py (LangGraph ReAct agent with intelligent tool selection)
    ↓
User Input → Safety Check → Agent Decides → Tool Execution → Synthesize → Response
```

---

## Part 1: Initialization Flow (What Happens at Startup)

### Step 1: main.py (Entry Point)
**File**: `main.py` - A simple 6-line entry point:
```python
from src.cli import main

if __name__ == "__main__":
    main()
```

**What it does**: Imports and calls the CLI main function. That's it. This is intentionally minimal.

---

### Step 2: cli.py (Initialize the App)
**File**: `src/cli.py` - The interactive command-line interface

**Key Function**: `main()` (line 50)

**What happens in order**:

1. **Print welcome message** (line 52)
   - Shows a pretty header and help text

2. **Initialize the application** (line 56)
   ```python
   app = create_app(skip_milvus=False)  # Creates ALL components
   app.ingest_sample_data()              # Loads parking docs into vector DB
   ```

3. **Start interactive loop** (line 70)
   ```python
   while True:
       user_input = input("You: ")
       # Handle commands (help, quit, parking list, evaluate, etc.)
       # OR process regular message via:
       result = app.process_user_message(user_input)
   ```

---

### Step 3: app.py (The Orchestrator)
**File**: `src/app.py` - The main application class

**Key Function**: `create_app()` (line 208) → Creates a `ParkingChatbotApp` instance

**Initialization sequence** (lines 18-79):

#### 3.1 Get Configuration
```python
self.config = get_config()  # Reads .env file
```

#### 3.2 Create Embeddings
```python
self.embeddings = create_embeddings()
```

**What it does**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Creates a 384-dimensional vector for every piece of text
- This is the foundation of semantic search

**Example**:
```
Question: "Where is downtown parking?"
Embedding: [0.23, -0.54, 0.12, ..., 0.88]  (384 numbers)

Document: "Downtown Garage located at 123 Main Street"
Embedding: [0.25, -0.52, 0.11, ..., 0.87]  (384 numbers)

These embeddings are SIMILAR because they're about the same topic
```

#### 3.3 Initialize SQL Database
```python
self.db = ParkingDatabase()
```

**What it does**:
- Creates SQLite database
- Stores **transactional/dynamic data**:
  - Parking spaces (capacity, availability)
  - Reservations (who booked what)
  - User information
  - Prices and schedules

**Key concept: Hot vs Cold Data**
- **Hot data** = Changes frequently (SQL DB) → availability, reservations
- **Cold data** = Static reference info (Vector DB) → parking information, policies

#### 3.4 Initialize Vector Database
```python
# Auto-detects from config: weaviate, pinecone, or milvus
if provider == "weaviate":
    self.vector_store = create_weaviate_connection(self.embeddings)
```

**What it does**:
- Connects to Weaviate (running in Docker)
- Creates a schema for storing documents with embeddings
- Stores **static reference data**:
  - Parking information documents
  - Policies and guidelines
  - Location details

**Why vector database?**
- SQL can only search by exact keywords: "parking" won't find "garage"
- Vector DB uses semantic search: "parking" WILL find documents about "garage" because they're semantically similar

#### 3.5 Initialize RAG Retriever
```python
self.rag_retriever = ParkingRAGRetriever(self.vector_store)
```

**What it does**:
- Creates the RAG (Retrieval-Augmented Generation) pipeline
- Connects the vector DB to an LLM
- Can now answer questions using retrieved documents as context

#### 3.6 Initialize Guardrails
```python
self.guard_rails = DataProtectionFilter()
```

**What it does**:
- Detects sensitive data (credit cards, SSN, emails, etc.)
- Blocks SQL injection attempts
- Blocks malicious keywords (delete, drop, hack, exploit, etc.)
- Masks sensitive data in responses

#### 3.7 Initialize Workflow
```python
self.workflow = ParkingChatbotWorkflow(
    self.rag_retriever, self.db, self.guard_rails
)
```

**What it does**:
- Creates a LangGraph-based agent workflow
- Handles multi-step request routing
- Coordinates RAG, database, and guardrails

---

## Part 2: User Message Processing Flow

When you type a message and press Enter, here's what happens:

### High-Level Flow
```
User Input
    ↓
app.process_user_message(user_input)  [app.py line 143]
    ↓
workflow.invoke(user_message)  [workflow.py line 254]
    ↓
LangGraph State Machine (processes through nodes)
```

### Detailed Step-by-Step

#### Step 1: Safety Check (workflow.py line 77)
```python
def _safety_check_node(state):
    is_safe, issue = self.guard_rails.check_safety(message)
    if not is_safe:
        state.safety_issue_detected = True
        state.chatbot_response = f"I cannot process this request. {issue}"
        return state  # Stop here, go to END
    return state  # Continue to next node
```

**What it checks**:
- Sensitive data patterns:
  - Credit card: `1234-5678-9012-3456`
  - Phone: `(555) 123-4567`
  - Email: `user@example.com`
  - SSN: `123-45-6789`
  - Passwords: `password=secret123`
  - API keys: `api_key=sk-...`

- Suspicious keywords (2+ matches required):
  - Database, delete, drop, insert, update, sql
  - Hack, bypass, exploit, injection, shell, execute, system

- Blacklisted operations:
  - "access other users data"
  - "modify reservations"
  - "delete user information"
  - etc.

**Routing**:
```
If unsafe → END (response block sent to user)
If safe   → Continue to process_query
```

---

#### Step 2: Intent Detection & Query Processing (workflow.py line 109)
```python
def _process_query_node(state):
    intent = self._detect_intent(state.current_message)

    if intent == "info":
        # Call RAG to answer
        result = self.rag_retriever.query(state.current_message)
        state.chatbot_response = result["answer"]
        state.response_sources = result["sources"]

    elif intent == "reservation":
        state.chatbot_response = "I'll help you make a reservation..."

    return state
```

**Intent Detection** (workflow.py line 238):
```python
reservation_keywords = ["book", "reserve", "reservation", "want to park"]
if any(keyword in message.lower() for keyword in reservation_keywords):
    return "reservation"
return "info"
```

**Routing based on intent**:
```
If "info" intent       → Send to process_query → complete → END
If "reservation" intent → Send to collect_reservation → admin_review → complete → END
```

---

### 🔑 The RAG Pipeline (Core Production Concept)

When intent = "info", here's what `rag_retriever.query()` does (retriever.py line 89):

#### RAG Step 1: Embed the Question
```python
question = "Where is downtown parking?"
question_embedding = embeddings.embed_query(question)
# Result: [0.23, -0.54, 0.12, ..., 0.88] (384 dimensions)
```

#### RAG Step 2: Search Vector Database
```python
# Find top 3 most similar documents in Weaviate
similar_docs = vector_store.similarity_search(question_embedding, k=3)
```

**How similarity works**:
```
Document 1: "Downtown Garage located at 123 Main Street"
Embedding: [0.25, -0.52, 0.11, ..., 0.87]

Question Embedding: [0.23, -0.54, 0.12, ..., 0.88]

Cosine Similarity = (0.23×0.25 + -0.54×-0.52 + ...) / (magnitude1 × magnitude2)
Result: 0.92 (very similar!)
```

#### RAG Step 3: Build Context
```python
retrieved_docs = [
    "Downtown Garage located at 123 Main Street...",
    "Parking policies and procedures...",
    "Payment methods and pricing..."
]

context = "\n".join([doc.text for doc in retrieved_docs])
```

#### RAG Step 4: Create LLM Prompt
```
You are a helpful parking information assistant.
Use the following context to answer the question.

Context:
Downtown Garage located at 123 Main Street...
Parking policies and procedures...
Payment methods and pricing...

Question: Where is downtown parking?

Answer:
```

#### RAG Step 5: Call LLM
```python
answer = llm.invoke(prompt)
# LLM sees the context and generates answer based on it
# Result: "The downtown parking is located at 123 Main Street..."
```

#### RAG Step 6: Return Result
```python
return {
    "answer": answer,
    "sources": retrieved_docs  # Show user where info came from
}
```

---

## Part 3: Component Breakdown

### 1. Embeddings (src/rag/embeddings.py)

**What it is**: A pre-trained neural network that converts text → vectors

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Pre-trained on 215M sentence pairs
- Outputs 384-dimensional vectors
- Fast (optimized for semantic search)
- Free (open source)

**Key methods**:
```python
embeddings.embed_query(text)        # Embed a single query
embeddings.embed_documents(texts)   # Embed multiple documents
```

---

### 2. Vector Database (src/database/weaviate_db.py)

**What it is**: A database optimized for storing and searching vectors

**Configuration**:
- Class: `ParkingStaticData`
- Properties:
  - `text`: The actual document content
  - `metadata_source`: Where it came from
  - `metadata_page`: Page number
- Vectorizer: "none" (we provide custom embeddings)
- Vector dimension: 384 (matches embedding model)

**Key operations**:
```python
vector_store.add_documents(docs)              # Store docs with embeddings
vector_store.as_retriever(k=3)                # Get top 3 most similar
vector_store.similarity_search(query, k=3)    # Search by similarity
```

---

### 3. SQL Database (src/database/sql_db.py)

**What it is**: Traditional SQL database for transactional data

**Tables**:
- `parking_spaces` (id, name, location, capacity, available_spaces, price_per_hour, is_open)
- `reservations` (res_id, user_name, user_surname, car_number, parking_id, start_time, end_time, status)
- `users` (user_id, name, email, phone, created_at)

**Why SQL and not Vector DB?**
- SQL is for **transactional consistency** (availability updates)
- SQL is for **exact queries** (get parking_id = "downtown_1")
- Vector DB is for **semantic search** (find documents about parking)

---

### 4. LLM Provider (src/rag/llm_provider.py)

**What it is**: Abstraction layer supporting multiple LLM providers

**Supported providers**:
- Ollama (free, local) - Default
- OpenAI (ChatGPT)
- Google Gemini
- Anthropic Claude

**Key concept: Provider pattern**
```python
if provider == "ollama":
    llm = Ollama(model="deepseek-r1:1.5b")
elif provider == "openai":
    llm = ChatOpenAI(model="gpt-4", api_key=...)
elif provider == "gemini":
    llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=...)
```

Configuration via `.env`:
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=deepseek-r1:1.5b
```

---

### 5. RAG Retriever (src/rag/retriever.py)

**What it is**: The complete RAG pipeline orchestrator

**Key class**: `ParkingRAGRetriever`

**Key methods**:
```python
def query(question: str) -> dict:
    # 1. Embed question
    # 2. Search vector DB
    # 3. Build context
    # 4. Call LLM
    # 5. Return answer + sources
```

**Temperature setting**:
```python
llm = create_llm(temperature=0.3)
```
- Temperature 0.3 = More consistent/factual (good for facts)
- Temperature 0.9 = More creative (good for brainstorming)

---

### 6. Guardrails (src/guardrails/filter.py)

**What it is**: Security filter for input/output

**Main class**: `DataProtectionFilter`

**Two check methods**:

1. **check_safety(message)** - Check input before processing
   - Detects sensitive data
   - Detects suspicious keywords
   - Detects blacklisted operations
   - Returns: (is_safe: bool, reason: str)

2. **filter_response(response)** - Clean output before returning
   - Masks emails: `user@example.com` → `[EMAIL_MASKED]`
   - Masks phones: `(555) 123-4567` → `[PHONE_MASKED]`
   - Masks IPs: `192.168.1.1` → `[IP_MASKED]`
   - Masks credit cards: `1234-5678-9012-3456` → `[CC_MASKED]`

**Production concept**: Security by default
- Block first, ask questions later
- Better to block a legitimate request than expose sensitive data

---

### 7. Workflow (src/agents/workflow.py)

**What it is**: LangGraph-based ReAct agent with intelligent tool selection

**Framework**: LangGraph (built on LangChain)

**Key concept**: ReAct (Reasoning and Acting) agent that decides WHAT tools to use

```python
workflow = StateGraph(ConversationState)
workflow.add_node("safety_check", ...)
workflow.add_node("agent_decide", ...)      # LLM decides what action to take
workflow.add_node("vector_search", ...)     # Search static knowledge base
workflow.add_node("sql_query", ...)         # Query real-time data
workflow.add_node("direct_response", ...)   # Respond without retrieval
workflow.add_node("synthesize", ...)        # Combine tool results into answer
workflow.add_node("output_filter", ...)     # Apply guardrails to output
```

**ReAct Agent Flow**:
```
User Query
    │
    ▼
┌─────────────────┐
│  safety_check   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
[unsafe]   [safe]
    │         │
    ▼         ▼
   END   ┌─────────────────┐
         │  agent_decide   │ ◄─────────────────────┐
         │  (LLM decides)  │                       │
         └────────┬────────┘                       │
                  │                                │
    ┌─────────────┼─────────────┬─────────────┐   │
    ▼             ▼             ▼             ▼   │
[vector]     [sql_query]   [direct]    [reservation]
    │             │             │             │
    ▼             ▼             ▼             ▼
┌────────┐  ┌────────┐    ┌────────┐   (existing
│ search │  │ query  │    │respond │    flow)
└───┬────┘  └───┬────┘    └───┬────┘
    │           │             │
    └─────┬─────┘             │
          ▼                   │
    ┌──────────┐              │
    │synthesize│◄─────────────┘
    └────┬─────┘
         ▼
    ┌──────────┐
    │ output   │
    │ filter   │
    └────┬─────┘
         ▼
        END
```

**Available Actions** (Agent chooses based on query):

| Action | Use Case | Retrieval | Sources Shown |
|--------|----------|-----------|---------------|
| `direct_response` | Greetings, thanks, chitchat | None | No |
| `vector_search` | Parking locations, policies, rules | Vector DB | Yes |
| `sql_query` | Current availability, prices, status | SQL Agent | Yes |
| `start_reservation` | Booking requests | None | No |
| `synthesize` | Combine results from multiple tools | N/A | Yes |

**Key Innovation**: The agent intelligently decides whether retrieval is needed:
```python
# Example behaviors:
"Hey" → direct_response (no retrieval, no sources)
"Where is downtown parking?" → vector_search (static info)
"How many spaces available?" → sql_query (real-time data)
"I want to book parking" → start_reservation (workflow)
```

---

## Part 4: Complete Request Example

### Example: User asks "Where is downtown parking?"

#### Phase 1: Initialization (First time only)
```
python main.py
  → cli.py main()
  → create_app()
  → Embeddings created: "What is downtown parking?" → [0.23, -0.54, ...]
  → Vector DB connected to Weaviate
  → Documents ingested with embeddings:
     "Downtown Garage located at 123 Main Street" → [0.25, -0.52, ...]
     "Airport Parking 2 miles from terminal" → [0.18, -0.45, ...]
     "Riverside Parking near the river" → [0.20, -0.48, ...]
```

#### Phase 2: User Input
```
User: "Where is downtown parking?"
```

#### Phase 3: Safety Check
```
_safety_check_node():
  check_safety("Where is downtown parking?")
  → No credit cards ✓
  → No suspicious keywords ✓
  → Not blacklisted ✓
  → Result: Safe to process
  → Route to: process_query
```

#### Phase 4: Intent Detection
```
_process_query_node():
  _detect_intent("Where is downtown parking?")
  → No reservation keywords found
  → Intent: "info"
  → Route to: Call RAG retriever
```

#### Phase 5: RAG Pipeline
```
rag_retriever.query("Where is downtown parking?"):

  Step 1 - Embed question:
    embeddings.embed_query("Where is downtown parking?")
    → [0.23, -0.54, 0.12, -0.34, 0.18, ...]  (384 dims)

  Step 2 - Search vector DB:
    vector_store.similarity_search(question_embedding, k=3)
    → Document 1: [0.25, -0.52, 0.11, ...] Similarity: 0.92 ✓ TOP MATCH
    → Document 2: [0.18, -0.45, 0.15, ...] Similarity: 0.85
    → Document 3: [0.20, -0.48, 0.10, ...] Similarity: 0.78

  Step 3 - Build context:
    context = """
    Downtown Garage located at 123 Main Street, open 24/7, capacity 500 spaces, $5/hour
    Airport Parking 2 miles from terminal, open 6am-10pm, capacity 1000 spaces, $3/hour
    Riverside Parking near the river, open 24/7, capacity 200 spaces, $4/hour
    """

  Step 4 - Create prompt:
    prompt = """
    You are a helpful parking information assistant.
    Use the following context to answer the question.

    Context:
    Downtown Garage located at 123 Main Street...
    Airport Parking 2 miles from terminal...
    Riverside Parking near the river...

    Question: Where is downtown parking?
    Answer:"""

  Step 5 - Call LLM:
    llm.invoke(prompt)
    → "The downtown parking is located at 123 Main Street.
         It's a garage with 500 spaces, open 24/7, at $5 per hour."

  Step 6 - Return:
    {
      "answer": "The downtown parking is located at 123 Main Street...",
      "sources": [doc1, doc2, doc3]
    }
```

#### Phase 6: Format & Display Response
```
cli.py format_response():
  response_text = "The downtown parking is located at 123 Main Street..."

  sources = [doc1, doc2, doc3]
  response_text += "\n\nSources:"
  response_text += "\n  [1] Downtown Garage located at 123 Main Street..."
  response_text += "\n  [2] Airport Parking 2 miles from terminal..."
  response_text += "\n  [3] Riverside Parking near the river..."

  Print to user:
  Bot: The downtown parking is located at 123 Main Street...

       Sources:
       [1] Downtown Garage located at 123 Main Street...
       [2] Airport Parking 2 miles from terminal...
       [3] Riverside Parking near the river...
```

---

## Part 5: Production RAG Concepts Used

### 1. **Semantic Search** (Not Keyword Search)
```
Keyword search: "parking" only matches exact word "parking"
Semantic search: "parking" also finds "garage", "lot", "spaces"
                because they're semantically similar
```
**Implementation**: Embeddings + vector similarity

---

### 2. **Retrieval-Augmented Generation (RAG)**
```
Without RAG (Hallucination Risk):
  User: "Where is downtown parking?"
  LLM: Invents an answer from its training data (could be wrong)

With RAG (Grounded Answer):
  User: "Where is downtown parking?"
  System: Retrieves actual documents
  LLM: Answers based on retrieved documents
  Result: Accurate, sourced answer
```
**Implementation**: Vector DB + LLM with context

---

### 3. **Hot vs Cold Data**
```
Cold Data (Static, Reference):
  ├─ Parking information documents
  ├─ Policies and guidelines
  └─ Location details
  └─ Storage: Vector DB (Weaviate)
  └─ Access: Semantic search

Hot Data (Dynamic, Transactional):
  ├─ Parking space availability
  ├─ Reservations
  ├─ User information
  └─ Pricing and schedules
  └─ Storage: SQL DB (SQLite)
  └─ Access: Exact queries
```
**Production concept**: Separate databases by access pattern

---

### 4. **ReAct Agent-Based Routing** (LangGraph)
```
OLD (Inefficient): Every query → Full RAG pipeline (even "hey" triggers retrieval)
NEW (Intelligent): Agent decides → Only retrieve when needed

Agent Decision Tree:
  "Hey"                    → direct_response (NO retrieval)
  "Where is parking?"      → vector_search (static knowledge)
  "How many available?"    → sql_query (real-time data)
  "Book a space"           → start_reservation (workflow)
```

**Key Innovation**: The LLM agent decides WHAT tools to use:
- Greetings/chitchat → No retrieval, no sources shown
- Static info queries → Vector DB search
- Real-time queries → SQL Agent
- Combined queries → Multiple tools, then synthesize

**Implementation**: LangGraph ReAct agent with tool registry

---

### 5. **Guard Rails & Security**
```
Input Validation (Block bad requests):
  ├─ Sensitive data detection
  ├─ Malicious keyword detection
  └─ Blacklist checks

Output Filtering (Don't leak data):
  ├─ Mask emails
  ├─ Mask phone numbers
  ├─ Mask IP addresses
  └─ Mask credit cards
```
**Production concept**: Defense in depth (multiple layers)

---

### 6. **Multi-Step Workflows** (Human-in-the-Loop)
```
Reservation workflow:
  1. Collect name
  2. Collect car number
  3. Collect parking location
  4. Collect start/end time
  5. Submit to admin for approval
  6. Notify user of status
```
**Production concept**: Not everything should be fully automated

---

### 7. **Flexible LLM Integration**
```
Can switch providers without code changes:
  .env: LLM_PROVIDER=ollama  → Uses Ollama
  .env: LLM_PROVIDER=openai  → Uses OpenAI
  .env: LLM_PROVIDER=gemini  → Uses Google Gemini
  .env: LLM_PROVIDER=anthropic → Uses Claude
```
**Implementation**: Provider pattern + configuration

---

### 8. **Evaluation & Metrics**
```python
# From test_rag.py - Production RAG evaluation uses RAGAS framework
Metrics:
  ├─ Faithfulness: Does answer come from retrieved docs?
  ├─ Answer Relevance: Is answer relevant to question?
  ├─ Context Precision: Is retrieved context useful?
  └─ Recall@K: Do we retrieve the right documents?

Key Innovation: Use semantic similarity (cosine distance) not word-overlap
  ├─ Old (Wrong): Word-overlap Jaccard similarity
  ├─ New (Right): Embedding-based cosine similarity
  └─ Result: Metrics now accurately reflect RAG quality
```

**Automatic Report Generation**:
```bash
python test_rag.py                  # Run tests + generate report
python test_rag.py --no-report      # Run tests without report
```

Reports are saved to `reports/` folder with format:
`{provider}_{model}_test_results_{timestamp}.md`

Example: `ollama_llama2_7b_test_results_20260227_143022.md`

---

## Part 5b: Admin Dashboard & API Layer

### Overview
The Admin Dashboard provides a web-based interface for administrators to manage parking reservations. It runs as a FastAPI server alongside the chatbot.

```
┌─────────────────────────────────────────────────────────────┐
│                    Admin Dashboard                          │
│              http://localhost:8000/dashboard/               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Pending    │  │   Approved   │  │   Rejected   │      │
│  │  Requests    │  │    Today     │  │    Today     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Reservation Card                                    │   │
│  │  Name: John Doe | Car: ABC-123 | Parking: Downtown  │   │
│  │  Time: Feb 28 10:00 - Feb 28 18:00                  │   │
│  │                                                      │   │
│  │     [✓ Approve]              [✗ Reject]             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### API Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/dashboard/` | GET | HTML dashboard UI | No |
| `/dashboard/api/reservations/pending` | GET | List pending reservations | No |
| `/dashboard/api/reservations/{id}/approve` | POST | Approve a reservation | Yes* |
| `/dashboard/api/reservations/{id}/reject` | POST | Reject a reservation | Yes* |
| `/dashboard/api/reservations/history` | GET | Recent activity | No |

*When `REQUIRE_API_KEY=true` in `.env`

### Security Features (src/api/security.py)

**1. API Key Authentication**
```python
# Two ways to provide API key:
# Header: X-API-Key: your-secret-key
# Query:  ?api_key=your-secret-key

# Uses constant-time comparison to prevent timing attacks
secrets.compare_digest(provided_key, expected_key)
```

**2. Rate Limiting**
```python
class RateLimiter:
    max_requests = 100      # Per IP
    window_seconds = 60     # Time window
```
- Prevents abuse and DoS attacks
- Per-IP tracking
- Auto-cleanup of old entries

### Reservation File Writer (src/services/reservation_writer.py)

When a reservation is approved, it's written to a text file (MCP-style output):

```
========================================================
CONFIRMED PARKING RESERVATIONS
========================================================
Name                      | Car Number      | Reservation Period                       | Approval Time        | Admin
========================================================
John Doe                  | ABC-123         | 2026-02-28 10:00 to 2026-02-28 18:00    | 2026-02-28 09:30    | Dashboard Admin
```

**Security Features**:
- Input sanitization (prevents injection)
- File locking (thread-safe writes)
- Append-only mode (no overwrites)
- Path traversal prevention

### HITL Integration

The dashboard integrates with the Human-in-the-Loop workflow:

```
Chatbot                    Dashboard                    File System
   │                          │                             │
   ├──creates reservation────►│                             │
   │                          │                             │
   │                    Admin reviews                       │
   │                          │                             │
   │◄────approval sent────────┤                             │
   │                          ├────write to file───────────►│
   │                          │                             │
   ├──notify user─────────────┤                             │
```

---

## Part 6: File Responsibility Matrix

| File | Responsibility |
|------|---|
| `main.py` | Entry point only |
| `src/cli.py` | Interactive user interface + command handling |
| `src/app.py` | Application orchestrator + component initialization |
| `src/rag/embeddings.py` | Text → Vector conversion |
| `src/database/weaviate_db.py` | Static data storage + semantic search |
| `src/database/sql_db.py` | Dynamic data storage + exact queries |
| `src/rag/llm_provider.py` | LLM provider abstraction (Ollama/OpenAI/Gemini/Claude) |
| `src/rag/retriever.py` | RAG pipeline orchestration + hybrid retrieval |
| `src/rag/sql_agent.py` | SQL Agent for intelligent database queries |
| `src/guardrails/filter.py` | Input validation + output filtering |
| `src/agents/workflow.py` | ReAct agent routing via LangGraph state machine |
| `src/agents/state.py` | Conversation state definition + agent tracking |
| `src/agents/tools.py` | Tool definitions (VectorSearchTool, SQLQueryTool) |
| `src/agents/prompts.py` | Agent decision prompts + synthesis prompts |
| `src/api/dashboard.py` | Admin dashboard REST API + HTML web interface |
| `src/api/security.py` | API key authentication + rate limiting |
| `src/services/reservation_writer.py` | File export for confirmed reservations (MCP-style) |
| `test_rag.py` | Comprehensive testing + auto-generated markdown reports |
| `reports/` | Auto-generated test reports (markdown) |
| `data/confirmed_reservations/` | Exported reservation records |

---

## Part 7: Technology Stack & Production Concepts

### Core Stack
- **Python 3.10+** - Language
- **LangChain** - LLM framework
- **LangGraph** - Agentic workflows
- **FastAPI** - Admin Dashboard REST API
- **Weaviate** - Vector database (Docker)
- **SQLite** - Relational database
- **HuggingFace Transformers** - Embeddings

### LLM Options
- **Ollama** (free, local)
- **OpenAI** (ChatGPT)
- **Google Gemini** (Bard)
- **Anthropic Claude** (Claude)

### Production Patterns
1. **Provider Pattern** - Abstract LLM provider
2. **State Machine Pattern** - LangGraph for workflows
3. **RAG Pattern** - Grounded generation
4. **Defense in Depth** - Multiple security layers
5. **Hot/Cold Separation** - Different databases for different patterns
6. **Evaluation Framework** - RAGAS metrics for production RAG

---

## Part 8: Key Takeaways

### What Makes This Production-Ready

1. **Semantic Search** - Finds meaning, not just keywords
2. **Grounded Generation** - LLM answers from actual documents
3. **Security by Default** - Blocks first, validates always
4. **Flexible LLM** - Switch providers without code changes
5. **ReAct Agent Routing** - Intelligent tool selection (only retrieve when needed)
6. **Hybrid Retrieval** - Vector DB for static + SQL Agent for real-time data
7. **Human-in-Loop** - Critical decisions get admin approval
8. **Admin Dashboard** - Web UI for reservation management with real-time updates
9. **API Security** - API key authentication + rate limiting
10. **File Export** - Confirmed reservations written to file (MCP-style)
11. **Evaluation Metrics** - Measure RAG quality scientifically
12. **Auto-Generated Reports** - Test results saved as markdown reports

### Architecture Principles

- **Separation of Concerns** - Each component has one job
- **Configuration Management** - Control via `.env`, not code
- **Explicit State** - Workflow state is visible and trackable
- **Defensive Programming** - Validate at boundaries
- **Monitoring** - Logging at every step for debugging

---

## How to Run Examples

### Start the chatbot
```bash
docker-compose up -d           # Start Weaviate
sleep 10
uv run python main.py          # Start chatbot
```

### Try these queries
```
User: "Where is downtown parking?"
→ Uses RAG to find answer from documents

User: "What are the prices?"
→ Uses RAG to find pricing information

User: "I want to book a space"
→ Triggers reservation workflow

User: "help"
→ Shows available commands
```

### Run tests & evaluation
```bash
uv run python test_rag.py              # Full evaluation with metrics + auto-report
uv run python test_rag.py --no-report  # Tests only, no report file
```

Reports are automatically saved to `reports/` folder.

---

