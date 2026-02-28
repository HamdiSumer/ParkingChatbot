# RAG System Test Report

## Test Run Information

| Property | Value |
|----------|-------|
| **Date** | 2026-03-01 00:57:00 |
| **LLM Provider** | ollama |
| **LLM Model** | llama2:7b |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **Duration** | 607.0s |

## Test Summary

**Overall: ⚠️ 11/14 tests passed**

| Test | Status |
|------|--------|
| Guardrails | ✅ PASSED |
| RAG Metrics | ✅ PASSED |
| Recall@K | ✅ PASSED |
| Components | ✅ PASSED |
| Hybrid Retrieval | ❌ FAILED |
| E2E Workflow | ✅ PASSED |
| Agent Routing | ✅ PASSED |
| Admin Flow | ✅ PASSED |
| Data Architecture | ✅ PASSED |
| Chatbot Load Test | ❌ FAILED |
| Admin Load Test | ✅ PASSED |
| MCP Load Test | ✅ PASSED |
| MCP Server | ✅ PASSED |
| Pipeline Integration | ❌ FAILED |

## Detailed Results

### Guardrails

**Metrics:**

| Metric | Value |
|--------|-------|
| Unit Test Rate (%) | 100.00 |
| Integration Test Rate (%) | 100.00 |
| Overall Rate (%) | 100.00 |

**Test Cases:**

| Test Case | Expected | Result | Status |
|-----------|----------|--------|--------|
| [Unit] Safe query | Safe | Safe | ✅ |
| [Unit] Credit card | Blocked | Blocked | ✅ |
| [Unit] Phone number | Blocked | Blocked | ✅ |
| [Unit] Email | Blocked | Blocked | ✅ |
| [Unit] SSN | Blocked | Blocked | ✅ |
| [Unit] Password | Blocked | Blocked | ✅ |
| [Unit] Safe query | Safe | Safe | ✅ |
| [Unit] Safe query | Safe | Safe | ✅ |
| [Unit] SQL injection | Blocked | Blocked | ✅ |
| [Unit] SQL injection attempt | Blocked | Blocked | ✅ |
| [Integration] CC in input | Blocked | Blocked | ✅ |
| [Integration] SSN in input | Blocked | Blocked | ✅ |
| [Integration] SQL injection | Blocked | Blocked | ✅ |
| [Integration] Safe query | Allowed | Allowed | ✅ |
| [Integration] Safe query | Allowed | Allowed | ✅ |

### RAG Metrics

**Metrics:**

| Metric | Value |
|--------|-------|
| Avg Faithfulness | 0.62 |
| Avg Answer Relevance | 0.65 |
| Avg Context Precision | 1.00 |
| Avg Latency (ms) | 13286.65 |

### Recall@K

**Metrics:**

| Metric | Value |
|--------|-------|
| Recall@1 | 1.00 |
| Avg Relevance@1 | 0.77 |
| Recall@3 | 1.00 |
| Avg Relevance@3 | 0.77 |
| Recall@5 | 1.00 |
| Avg Relevance@5 | 0.77 |

### Components

**Test Cases:**

| Test Case | Expected | Result | Status |
|-----------|----------|--------|--------|
| Embeddings | OK | OK | ✅ |
| Weaviate Vector DB | OK | OK | ✅ |
| SQLite Database | OK | OK | ✅ |
| LLM (Ollama) | OK | OK | ✅ |

### Hybrid Retrieval

**Metrics:**

| Metric | Value |
|--------|-------|
| SQL Queries Tested | 3 |
| Vector Queries Tested | 2 |
| Queries With Sources | 2 |
| Avg Latency (ms) | 13245.84 |

**Test Cases:**

| Test Case | Expected | Result | Status |
|-----------|----------|--------|--------|
| availability: How many parking spaces ar | sql_query | Pass | ✅ |
| status: What is the current status of  | sql_query | Fail | ❌ |
| pricing: How much does parking cost rig | sql_query | Pass | ✅ |
| location: Where is the downtown parking  | vector_search | Pass | ✅ |
| rules: What are the parking rules? | vector_search | Pass | ✅ |

### E2E Workflow

**Test Cases:**

| Test Case | Expected | Result | Status |
|-----------|----------|--------|--------|
| What parking spaces are available? | info | Pass | ✅ |
| My credit card is 4532-1234-5678-9012 | safety | Pass | ✅ |
| I want to book a parking space | intent | Pass | ✅ |

### Agent Routing

**Metrics:**

| Metric | Value |
|--------|-------|
| Routing Accuracy (%) | 90.00 |

**Test Cases:**

| Test Case | Expected | Result | Status |
|-----------|----------|--------|--------|
| Hey | direct_response | Pass | ✅ |
| Hello there | direct_response | Pass | ✅ |
| Thanks! | direct_response | Pass | ✅ |
| Ok | direct_response | Pass | ✅ |
| Where is downtown parking? | vector_search | Pass | ✅ |
| What are the parking rules? | vector_search | Pass | ✅ |
| How do I book a parking space? | vector_search | Fail | ❌ |
| How many spaces are available? | sql_query | Pass | ✅ |
| What is the current price? | sql_query | Pass | ✅ |
| I want to book a parking space | reservation | Pass | ✅ |

### Admin Flow

**Metrics:**

| Metric | Value |
|--------|-------|
| Tests Passed | 8 |
| Total Tests | 8 |
| Pass Rate (%) | 100.00 |

**Test Cases:**

| Test Case | Expected | Result | Status |
|-----------|----------|--------|--------|
| Create Reservation | Success | Success | ✅ |
| Pending List | In pending | Found | ✅ |
| Status Check (Pending) | pending | pending | ✅ |
| Approve Reservation | Success | Success | ✅ |
| Status Check (Confirmed) | confirmed | confirmed | ✅ |
| Create Second Reservation | Success | Success | ✅ |
| Reject Reservation | Success | Success | ✅ |
| Status Check (Rejected) | rejected + reason | rejected | ✅ |

### Data Architecture

**Notes:** Architecture: Weaviate (static) + SQLite (dynamic) + Guard rails (safety)

### Chatbot Load Test

**Metrics:**

| Metric | Value |
|--------|-------|
| Concurrent Users | 5 |
| Queries Per User | 3 |
| Total Requests | 15 |
| Success Rate (%) | 100.00 |
| Avg Response Time (ms) | 74999.23 |
| P95 Response Time (ms) | 144641.02 |

**Notes:** Simulated 5 concurrent users

### Admin Load Test

**Metrics:**

| Metric | Value |
|--------|-------|
| Concurrent Admins | 5 |
| Operations Per Admin | 4 |
| Total Operations | 20 |
| Success Rate (%) | 100.00 |
| Avg Operation Time (ms) | 16.15 |

### MCP Load Test

**Metrics:**

| Metric | Value |
|--------|-------|
| Concurrent Writers | 5 |
| Operations Per Writer | 4 |
| Total Operations | 20 |
| Success Rate (%) | 100.00 |
| Avg Operation Time (ms) | 0.15 |

### MCP Server

**Metrics:**

| Metric | Value |
|--------|-------|
| Tests Passed | 5 |
| Total Tests | 5 |
| Pass Rate (%) | 100.00 |

**Test Cases:**

| Test Case | Expected | Result | Status |
|-----------|----------|--------|--------|
| Get File Info | File info returned | Success | ✅ |
| Write Reservation | Reservation written | Success | ✅ |
| Read Reservations | Reservations list returned | Success | ✅ |
| Verify File Update | Updated file info | Success | ✅ |
| Input Sanitization | Sanitized write | Success | ✅ |

### Pipeline Integration

**Metrics:**

| Metric | Value |
|--------|-------|
| Stages Passed | 3 |
| Total Stages | 4 |
| Pass Rate (%) | 75.00 |

**Test Cases:**

| Test Case | Expected | Result | Status |
|-----------|----------|--------|--------|
| Chatbot Query Processing | Pass | Pass | ✅ |
| Safety Guardrails | Pass | Fail | ❌ |
| Reservation & Admin Flow | Pass | Pass | ✅ |
| MCP Data Recording | Pass | Pass | ✅ |

---
*Report generated by test_rag.py*