# Testing Documentation - Parking Chatbot RAG System

This document provides a deep dive into the testing methodology used in `test_rag.py`, explaining what each test measures, how metrics are calculated, and how they relate to the actual RAG system implementation.

---

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Categories](#test-categories)
3. [Test 1: Guardrails (Security)](#test-1-guardrails-security)
4. [Test 2: RAG Metrics (RAGAS Framework)](#test-2-rag-metrics-ragas-framework)
5. [Test 3: Recall@K (Retrieval Quality)](#test-3-recallk-retrieval-quality)
6. [Test 4: Component Initialization](#test-4-component-initialization)
7. [Test 5: Hybrid Retrieval](#test-5-hybrid-retrieval)
8. [Test 6: End-to-End Workflow](#test-6-end-to-end-workflow)
9. [Test 7: Agent Routing](#test-7-agent-routing)
10. [Test 8: Data Architecture](#test-8-data-architecture)
11. [How Tests Map to System Components](#how-tests-map-to-system-components)
12. [Running Tests](#running-tests)

---

## Testing Philosophy

Our testing approach follows a **layered testing strategy**:

```
┌─────────────────────────────────────────────────────────┐
│                    Integration Tests                      │
│    Tests the ACTUAL user flow through the system          │
│    (process_user_message → ReAct agent → response)        │
├─────────────────────────────────────────────────────────┤
│                    Component Tests                        │
│    Tests individual components in isolation               │
│    (embeddings, retriever, guardrails filter)             │
├─────────────────────────────────────────────────────────┤
│                    Unit Tests                             │
│    Tests specific functions with known inputs/outputs     │
│    (check_safety patterns, similarity calculations)       │
└─────────────────────────────────────────────────────────┘
```

**Key Principle**: We test through `process_user_message()` whenever possible to ensure the ReAct agent workflow is actually being exercised.

---

## Test Categories

| Test | Type | Tests Through | Purpose |
|------|------|---------------|---------|
| Guardrails | Unit + Integration | Filter + `process_user_message()` | Security verification |
| RAG Metrics | Integration | `process_user_message()` | Answer quality |
| Recall@K | Component | `retrieve_documents()` | Retrieval ranking |
| Components | Component | Direct initialization | System health |
| Hybrid Retrieval | Integration | `process_user_message()` | SQL + Vector routing |
| E2E Workflow | Integration | `process_user_message()` | Full system flow |
| Agent Routing | Integration | `process_user_message()` | Tool selection |
| Data Architecture | Documentation | N/A | Architecture verification |

---

## Test 1: Guardrails (Security)

### What It Tests

The guardrails test verifies that our security system:
1. **Detects sensitive data** (credit cards, SSN, emails, passwords)
2. **Blocks malicious inputs** (SQL injection, suspicious keywords)
3. **Is actually integrated** into the agent workflow

### Implementation in the RAG System

**File**: `src/guardrails/filter.py`

```python
class DataProtectionFilter:
    def check_safety(self, message: str) -> Tuple[bool, str]:
        """Check input for sensitive data and malicious patterns."""
        # Returns (is_safe, reason)

    def filter_output(self, response: str) -> Tuple[str, bool]:
        """Mask sensitive data in output before returning to user."""
        # Returns (filtered_response, was_modified)
```

**Integration in Workflow** (`src/agents/workflow.py`):

```python
# INPUT FILTERING (line 151)
def _safety_check_node(self, state):
    is_safe, issue = self.guard_rails.check_safety(state.current_message)
    if not is_safe:
        state.safety_issue_detected = True
        state.chatbot_response = f"I cannot process this request. {issue}"
    return state

# OUTPUT FILTERING (line 388)
def _output_filter_node(self, state):
    filtered_response, was_modified = self.guard_rails.filter_output(
        state.chatbot_response
    )
    state.chatbot_response = filtered_response
    return state
```

### How We Test It

#### Part 1: Unit Test (Does the filter work?)

```python
filter_obj = DataProtectionFilter()
is_safe, reason = filter_obj.check_safety("My credit card is 4532-1234-5678-9010")
# Expected: is_safe = False (blocked)
```

**Test Cases**:
| Input | Expected | Pattern Detected |
|-------|----------|------------------|
| `4532-1234-5678-9010` | Blocked | Credit card regex |
| `555-123-4567` | Blocked | Phone number regex |
| `user@example.com` | Blocked | Email regex |
| `123-45-6789` | Blocked | SSN regex |
| `Password: secret` | Blocked | Password keyword |
| `DROP TABLE users` | Blocked | SQL injection |
| `Where is parking?` | Safe | No patterns |

#### Part 2: Integration Test (Does the agent actually use it?)

```python
result = app.process_user_message("My credit card is 4111-1111-1111-1111")
# Expected: Response indicates blocking, no LLM processing occurred
```

**Why Both Tests Matter**:
- Unit test: Verifies the filter patterns are correct
- Integration test: Verifies the workflow actually calls the filter

---

## Test 2: RAG Metrics (RAGAS Framework)

### What It Tests

The RAG metrics test evaluates answer quality using the **RAGAS framework** (Retrieval-Augmented Generation Assessment):

1. **Faithfulness**: Is the answer derived from the retrieved context?
2. **Answer Relevance**: Does the answer address the question?
3. **Context Precision**: How much of the retrieved context was useful?

### The RAG Triad Explained

```
                    Question
                       │
                       ▼
              ┌────────────────┐
              │    Retriever   │
              └────────┬───────┘
                       │
                       ▼
                   Context
                       │
                       ▼
              ┌────────────────┐
              │      LLM       │
              └────────┬───────┘
                       │
                       ▼
                    Answer

RAGAS Metrics:
• Context Precision: Question ↔ Context (Is retrieved context relevant?)
• Faithfulness: Context ↔ Answer (Is answer grounded in context?)
• Answer Relevance: Question ↔ Answer (Does answer address question?)
```

### How Each Metric Is Calculated

#### 1. Faithfulness

**Definition**: What fraction of the answer's claims are supported by the context?

**Implementation** (`test_rag.py`, line ~310):

```python
def calculate_faithfulness(answer: str, context: str) -> float:
    # Split answer into sentences
    answer_sentences = [s.strip() for s in answer.split(".") if s.strip()]

    # Embed the context
    context_embedding = embeddings.embed_query(context)

    faithful_sentences = 0
    for sentence in answer_sentences:
        # Embed each sentence
        sentence_embedding = embeddings.embed_query(sentence)

        # Calculate cosine similarity
        similarity = cosine_similarity(sentence_embedding, context_embedding)

        # If similarity > 0.4, sentence is "faithful" to context
        if similarity > 0.4:
            faithful_sentences += 1

    return faithful_sentences / len(answer_sentences)
```

**Example**:
```
Context: "Downtown Garage is at 123 Main Street. Open 24/7. $5/hour."
Answer: "Downtown parking is at 123 Main Street. It costs $5 per hour."

Sentence 1: "Downtown parking is at 123 Main Street" → similarity 0.85 ✓
Sentence 2: "It costs $5 per hour" → similarity 0.72 ✓

Faithfulness = 2/2 = 1.0
```

#### 2. Answer Relevance

**Definition**: How semantically similar is the answer to the question?

**Implementation** (`test_rag.py`, line ~340):

```python
def calculate_answer_relevance(question: str, answer: str) -> float:
    # Embed question and answer
    question_embedding = embeddings.embed_query(question)
    answer_embedding = embeddings.embed_query(answer)

    # Calculate cosine similarity
    similarity = cosine_similarity(question_embedding, answer_embedding)

    return max(0, min(1, similarity))  # Clamp to [0, 1]
```

**Example**:
```
Question: "Where is downtown parking?"
Answer: "Downtown parking is located at 123 Main Street."

Similarity = 0.82 (high relevance - answer addresses the question)
```

#### 3. Context Precision

**Definition**: What fraction of retrieved documents were useful for the answer?

**Implementation** (`test_rag.py`, line ~362):

```python
def calculate_context_precision(context_docs: List, answer: str) -> float:
    # Embed the answer
    answer_embedding = embeddings.embed_query(answer)

    useful_docs = 0
    for doc in context_docs:
        # Embed each document
        doc_embedding = embeddings.embed_query(doc.page_content)

        # Calculate similarity to answer
        similarity = cosine_similarity(doc_embedding, answer_embedding)

        # If similarity > 0.3, document was "useful"
        if similarity > 0.3:
            useful_docs += 1

    return useful_docs / len(context_docs)
```

**Example**:
```
Retrieved docs: [doc1, doc2, doc3]
Answer: "Downtown parking is at 123 Main Street"

doc1 (Downtown info): similarity 0.75 ✓ useful
doc2 (Airport info): similarity 0.25 ✗ not useful
doc3 (Pricing info): similarity 0.35 ✓ useful

Context Precision = 2/3 = 0.67
```

### How We Test It

```python
# Test goes through the ACTUAL workflow
result = app.process_user_message("Where is downtown parking?")
answer = result.get("response", "")
sources = result.get("sources", [])

# Calculate metrics
faithfulness = calculate_faithfulness(answer, context)
relevance = calculate_answer_relevance(query, answer)
precision = calculate_context_precision(sources, answer)
```

---

## Test 3: Recall@K (Retrieval Quality)

### What It Tests

Recall@K measures **retrieval ranking quality**: Do the top K retrieved documents contain relevant information?

### Understanding Recall@K

```
Query: "downtown parking location"

Retrieved Documents (k=5):
  [1] Downtown Garage at 123 Main Street    → Relevant ✓
  [2] Airport Parking information           → Not Relevant ✗
  [3] Downtown parking hours                → Relevant ✓
  [4] Riverside Parking details             → Not Relevant ✗
  [5] Downtown pricing                      → Relevant ✓

Recall@5 = Relevant docs / Total docs = 3/5 = 0.6
```

### How We Calculate It

**Implementation** (`test_rag.py`, line ~492):

```python
def test_recall_at_k():
    query = "downtown parking location"
    k_values = [1, 3, 5]

    # Embed the query
    query_embedding = embeddings.embed_query(query)

    for k in k_values:
        # Retrieve k documents
        docs = app.rag_retriever.retrieve_documents(query, k=k)

        relevance_scores = []
        for doc in docs:
            # Embed each document
            doc_embedding = embeddings.embed_query(doc.page_content)

            # Calculate similarity to query
            relevance = cosine_similarity(query_embedding, doc_embedding)
            relevance_scores.append(relevance)

        # Count documents with relevance > 0.3 as "relevant"
        relevant_count = sum(1 for score in relevance_scores if score > 0.3)

        recall_at_k = relevant_count / len(docs)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
```

### Why This Matters

| K Value | Use Case |
|---------|----------|
| Recall@1 | Is the TOP result relevant? (Important for direct answers) |
| Recall@3 | Are the top 3 results relevant? (Standard RAG retrieval) |
| Recall@5 | Are we getting enough relevant context? |

**Good Recall@K indicates**:
- Vector embeddings are capturing semantic meaning
- The retriever is ranking relevant documents higher
- The knowledge base has appropriate content

---

## Test 4: Component Initialization

### What It Tests

Verifies all system components initialize correctly:

| Component | What We Check |
|-----------|---------------|
| Embeddings | 384-dimensional vectors produced |
| Weaviate | Vector DB connection established |
| SQLite | Database accessible, parking spaces stored |
| LLM | Language model connected and responding |

### How We Test It

```python
def test_component_initialization():
    with create_test_app() as app:
        # Check embeddings
        embeddings = _get_embeddings()
        test_vec = embeddings.embed_query("test")
        assert len(test_vec) == 384  # MiniLM-L6-v2 produces 384-dim vectors

        # Check Weaviate
        assert app.rag_retriever is not None

        # Check SQLite
        spaces = app.list_parking_spaces()
        assert len(spaces) > 0

        # Check LLM
        assert app.rag_retriever.llm is not None
```

---

## Test 5: Hybrid Retrieval

### What It Tests

Verifies the system correctly routes queries to:
- **Vector DB** (Weaviate) for static information
- **SQL Agent** for real-time data

### Implementation in RAG System

**File**: `src/agents/workflow.py`

The ReAct agent decides which tool to use:

```python
def _agent_decide_node(self, state):
    # Agent analyzes the query and decides:
    # - "vector_search" for static info (locations, rules, policies)
    # - "sql_query" for real-time data (availability, prices, status)
    # - "direct_response" for greetings (no retrieval)
```

### How We Test It

```python
test_queries = [
    # Real-time queries → should use sql_query
    ("How many parking spaces are available?", "sql_query"),
    ("What is the current price?", "sql_query"),

    # Static info queries → should use vector_search
    ("Where is downtown parking located?", "vector_search"),
    ("What are the parking rules?", "vector_search"),
]

for query, expected_route in test_queries:
    result = app.process_user_message(query)  # Goes through ReAct agent
    # Verify response contains appropriate data
```

---

## Test 6: End-to-End Workflow

### What It Tests

Tests the complete flow from user input to response:

```
User Input → Safety Check → Agent Decision → Tool Execution → Response
```

### Test Cases

```python
test_cases = [
    ("What parking spaces are available?", "info"),      # Normal query
    ("My credit card is 4532-1234-5678-9012", "safety"), # Should be blocked
    ("I want to book a parking space", "intent"),        # Reservation flow
]
```

### Verification Logic

```python
for query, expected_type in test_cases:
    result = app.process_user_message(query)

    if expected_type == "safety":
        # Should be blocked - verify safety_issue flag
        assert result.get("safety_issue") == True
    else:
        # Should work normally
        assert len(result.get("response", "")) > 20
```

---

## Test 7: Agent Routing

### What It Tests

Verifies the ReAct agent selects the correct tool for different query types:

| Query Type | Expected Action | Sources Shown? |
|------------|-----------------|----------------|
| Greetings ("Hey") | `direct_response` | No |
| Static info | `vector_search` | Yes |
| Real-time data | `sql_query` | Yes |
| Booking request | `start_reservation` | No |

### How We Test It

```python
test_cases = [
    # (query, expected_behavior, should_have_sources)
    ("Hey", "direct_response", False),
    ("Hello there", "direct_response", False),
    ("Where is downtown parking?", "vector_search", True),
    ("How many spaces available?", "sql_query", True),
    ("I want to book parking", "reservation", False),
]

for query, expected_behavior, should_have_sources in test_cases:
    result = app.process_user_message(query)
    sources = result.get("sources", [])
    has_sources = len(sources) > 0

    # Verify sources match expectation
    assert has_sources == should_have_sources

    # For greetings, verify no parking data in response
    if expected_behavior == "direct_response":
        response = result.get("response", "")
        assert "downtown" not in response.lower()
        assert "available" not in response.lower()
```

### Why This Test Is Important

**Before the ReAct agent** (old behavior):
```
"Hey" → Full RAG pipeline → Retrieves parking docs → Shows sources
        (Wasteful and confusing)
```

**After the ReAct agent** (new behavior):
```
"Hey" → Agent decides "direct_response" → No retrieval → Simple greeting
        (Efficient and natural)
```

---

## Test 8: Data Architecture

### What It Tests

This is a documentation/verification test that displays the system architecture:

```
WEAVIATE (Vector DB) - STATIC DATA
├─ General parking information
├─ Parking locations and features
├─ Booking process information
└─ FAQs and guidelines

SQLITE (SQL Database) - DYNAMIC DATA
├─ Real-time availability
├─ Prices
├─ Reservations
└─ User information
```

### Purpose

- Confirms the hot/cold data separation strategy
- Documents the architecture for developers
- Serves as a sanity check that the design is implemented

---

## How Tests Map to System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           test_rag.py                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  test_guardrails()                                                        │
│       │                                                                   │
│       ├──► Unit Test ──► DataProtectionFilter.check_safety()             │
│       │                        │                                          │
│       │                        └──► src/guardrails/filter.py              │
│       │                                                                   │
│       └──► Integration ──► app.process_user_message()                    │
│                                 │                                         │
│                                 └──► workflow._safety_check_node()        │
│                                 └──► workflow._output_filter_node()       │
│                                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  test_rag_metrics()                                                       │
│       │                                                                   │
│       └──► app.process_user_message() ──► ReAct Agent                    │
│                                              │                            │
│                                              ├──► vector_search tool      │
│                                              │       └──► retriever.py    │
│                                              │                            │
│                                              └──► synthesize              │
│                                                      └──► LLM             │
│                                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  test_recall_at_k()                                                       │
│       │                                                                   │
│       └──► rag_retriever.retrieve_documents()                            │
│                   │                                                       │
│                   └──► weaviate_db.py (similarity_search)                │
│                                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  test_hybrid_retrieval()                                                  │
│       │                                                                   │
│       └──► app.process_user_message() ──► ReAct Agent                    │
│                                              │                            │
│                                              ├──► sql_query tool          │
│                                              │       └──► sql_agent.py    │
│                                              │                            │
│                                              └──► vector_search tool      │
│                                                      └──► retriever.py    │
│                                                                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  test_agent_routing()                                                     │
│       │                                                                   │
│       └──► app.process_user_message() ──► ReAct Agent                    │
│                                              │                            │
│                                              ├──► agent_decide_node       │
│                                              │       └──► _parse_response │
│                                              │                            │
│                                              └──► Verifies tool selection │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Running Tests

### Basic Usage

```bash
# Run all tests with auto-generated report
uv run python test_rag.py

# Run tests without generating report
uv run python test_rag.py --no-report
```

### Output

```
================================================================================
              COMPREHENSIVE RAG & SYSTEM TEST SUITE
================================================================================

================================================================================
              TEST 1: GUARDRAILS - Security & Safety
================================================================================
  Part 1: Filter Component Test (Unit Test)
...
  Part 2: Workflow Integration Test (Agent Actually Blocks)
...

================================================================================
              TEST SUMMARY
================================================================================

Test Category                  Status
--------------------------------------------------
Guardrails                     ✓ PASSED
RAG Metrics                    ✓ PASSED
Recall@K                       ✓ PASSED
Components                     ✓ PASSED
Hybrid Retrieval               ✓ PASSED
E2E Workflow                   ✓ PASSED
Agent Routing                  ✓ PASSED
Data Architecture              ✓ PASSED
--------------------------------------------------

Overall: 8/8 test categories passed

✓ All tests passed! System is production-ready.

📄 Report saved to: reports/ollama_llama2_7b_test_results_20260227_143022.md
```

### Report Format

Reports are saved to `reports/` folder as markdown files:

```
reports/
├── ollama_llama2_7b_test_results_20260227_143022.md
├── ollama_deepseek-r1_1.5b_test_results_20260227_150000.md
└── ...
```

---

## Key Metrics Thresholds

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Faithfulness | > 0.8 | 0.5 - 0.8 | < 0.5 |
| Answer Relevance | > 0.7 | 0.4 - 0.7 | < 0.4 |
| Context Precision | > 0.6 | 0.3 - 0.6 | < 0.3 |
| Recall@3 | > 0.8 | 0.5 - 0.8 | < 0.5 |
| Agent Routing Accuracy | > 80% | 60% - 80% | < 60% |
| Guardrails Detection | 100% | 90% - 100% | < 90% |

---

## Cosine Similarity Explained

Many of our metrics use **cosine similarity** to compare embeddings:

```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude
```

**Visual Explanation**:

```
Cosine Similarity = cos(θ) between two vectors

     vec1
      ↗
     /  θ = small angle → high similarity (close to 1)
    /
   ●────────→ vec2


     vec1
      ↑
      │  θ = 90° → no similarity (0)
      │
   ●────────→ vec2
```

**In practice**:
- Similarity > 0.8: Very similar (same topic)
- Similarity 0.5-0.8: Related (same domain)
- Similarity < 0.5: Different topics

---

## Summary

Our testing strategy ensures:

1. **Security** - Guardrails are tested both as units AND integrated into the workflow
2. **Quality** - RAG metrics (RAGAS) measure answer faithfulness, relevance, and context precision
3. **Retrieval** - Recall@K verifies the vector search returns relevant documents
4. **Routing** - Agent routing tests verify the ReAct agent makes correct tool selections
5. **Integration** - Most tests go through `process_user_message()` to test the actual user flow

**The key insight**: Unit tests verify components work, but integration tests verify the system actually uses them correctly.
