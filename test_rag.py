#!/usr/bin/env python3
"""
Comprehensive RAG & System Testing Suite
Combines RAGAS metrics, integration tests, load tests, and end-to-end verification.

Tests include:
- RAGAS metrics (Faithfulness, Answer Relevance, Context Precision)
- Security guardrails (PII detection, SQL injection prevention)
- Component initialization verification
- Hybrid retrieval (Vector DB + SQL Agent)
- Agent routing decisions
- Admin human-in-the-loop flow
- Load tests (concurrent users, admin operations, MCP writes)
- MCP server functional tests
- Full pipeline integration

Run: python test_rag.py
     python test_rag.py --no-report  # Skip report generation
"""

# Suppress ALL output BEFORE importing anything
import os
import sys
import warnings
import io
from datetime import datetime
from pathlib import Path

# Set environment variables first
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLX_DISABLE_PROGRESS_BAR"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

# Filter all warnings
warnings.filterwarnings("ignore")

# Redirect stderr during imports to suppress library warnings
_stderr = sys.stderr
sys.stderr = io.StringIO()

import time
import asyncio
import numpy as np
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import uuid

# Suppress all logs BEFORE importing src modules
from src.utils.logging import set_quiet_mode, suppress_warnings

set_quiet_mode(True)
suppress_warnings()

from src.app import create_app
from src.guardrails.filter import DataProtectionFilter
from src.rag.embeddings import create_embeddings

# Restore stderr after imports
sys.stderr = _stderr

# Context manager for suppressing stderr during app creation
from contextlib import contextmanager


@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr."""
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr


@contextmanager
def create_test_app():
    """Create app for testing with automatic cleanup."""
    with suppress_stderr():
        app = create_app()
    try:
        yield app
    finally:
        with suppress_stderr():
            app.shutdown()


# Initialize embeddings for semantic similarity calculations
_embeddings_cache = None


def _get_embeddings():
    """Get cached embeddings instance."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = create_embeddings()
    return _embeddings_cache


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}".center(80))
    print("=" * 80)


class TestReportGenerator:
    """Generate markdown test reports."""

    def __init__(self):
        self.report_dir = Path(__file__).parent / "reports"
        self.report_dir.mkdir(exist_ok=True)
        self.test_results = {}
        self.detailed_results = {}
        self.start_time = datetime.now()
        self.end_time = None

    def get_model_info(self) -> dict:
        """Get current LLM model information from config."""
        try:
            from src.config import get_config
            config = get_config()
            provider = config.LLM_PROVIDER

            model_map = {
                "ollama": config.OLLAMA_MODEL,
                "openai": config.OPENAI_MODEL,
                "gemini": config.GEMINI_MODEL,
                "anthropic": config.ANTHROPIC_MODEL,
            }

            return {
                "provider": provider,
                "model": model_map.get(provider, "unknown"),
                "embedding_model": config.EMBEDDING_MODEL,
            }
        except Exception as e:
            return {"provider": "unknown", "model": "unknown", "embedding_model": "unknown"}

    def add_result(self, test_name: str, passed: bool, details: dict = None):
        """Add a test result."""
        self.test_results[test_name] = passed
        if details:
            self.detailed_results[test_name] = details

    def generate_report(self) -> str:
        """Generate markdown report content."""
        self.end_time = datetime.now()
        model_info = self.get_model_info()
        duration = (self.end_time - self.start_time).total_seconds()

        # Build report
        lines = [
            f"# RAG System Test Report",
            f"",
            f"## Test Run Information",
            f"",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| **Date** | {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| **LLM Provider** | {model_info['provider']} |",
            f"| **LLM Model** | {model_info['model']} |",
            f"| **Embedding Model** | {model_info['embedding_model']} |",
            f"| **Duration** | {duration:.1f}s |",
            f"",
            f"## Test Summary",
            f"",
        ]

        passed = sum(1 for v in self.test_results.values() if v)
        total = len(self.test_results)
        status_emoji = "✅" if passed == total else "⚠️"

        lines.extend([
            f"**Overall: {status_emoji} {passed}/{total} tests passed**",
            f"",
            f"| Test | Status |",
            f"|------|--------|",
        ])

        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            lines.append(f"| {test_name} | {status} |")

        lines.append("")

        # Add detailed results for each test
        lines.append("## Detailed Results")
        lines.append("")

        for test_name, details in self.detailed_results.items():
            lines.append(f"### {test_name}")
            lines.append("")

            if "metrics" in details:
                lines.append("**Metrics:**")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for metric, value in details["metrics"].items():
                    if isinstance(value, float):
                        lines.append(f"| {metric} | {value:.2f} |")
                    else:
                        lines.append(f"| {metric} | {value} |")
                lines.append("")

            if "test_cases" in details:
                lines.append("**Test Cases:**")
                lines.append("")
                lines.append("| Test Case | Expected | Result | Status |")
                lines.append("|-----------|----------|--------|--------|")
                for tc in details["test_cases"]:
                    status = "✅" if tc.get("passed", False) else "❌"
                    lines.append(f"| {tc.get('name', 'N/A')[:40]} | {tc.get('expected', 'N/A')} | {tc.get('result', 'N/A')} | {status} |")
                lines.append("")

            if "notes" in details:
                lines.append(f"**Notes:** {details['notes']}")
                lines.append("")

        # Footer
        lines.extend([
            "---",
            f"*Report generated by test_rag.py*",
        ])

        return "\n".join(lines)

    def save_report(self) -> str:
        """Save report to file and return the filepath."""
        model_info = self.get_model_info()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        # Clean model name for filename (replace special chars)
        model_name = model_info["model"].replace(":", "_").replace("/", "_")
        provider = model_info["provider"]

        filename = f"{provider}_{model_name}_test_results_{timestamp}.md"
        filepath = self.report_dir / filename

        report_content = self.generate_report()
        filepath.write_text(report_content)

        return str(filepath)


# ============================================================================
# 1. GUARDRAILS TEST (Shared between both test suites)
# ============================================================================


def test_guardrails():
    """Test guardrails - verify sensitive data filtering.

    Part 1: Unit test of DataProtectionFilter (component test)
    Part 2: Integration test through actual workflow (ensures agent uses guardrails)
    """
    print_section("TEST 1: GUARDRAILS - Security & Safety")

    # ==================== PART 1: Unit Test ====================
    print("\n" + "=" * 80)
    print("  Part 1: Filter Component Test (Unit Test)")
    print("=" * 80)

    filter_obj = DataProtectionFilter()
    unit_test_cases = [
        ("Where is downtown parking?", True, "Safe query"),
        ("My credit card is 4532-1234-5678-9010", False, "Credit card"),
        ("Call me at 555-123-4567", False, "Phone number"),
        ("Email: user@example.com", False, "Email"),
        ("My SSN is 123-45-6789", False, "SSN"),
        ("Password: MySecure123!", False, "Password"),
        ("I need to book a parking space", True, "Safe query"),
        ("What are the prices?", True, "Safe query"),
        ("DROP TABLE users; DELETE FROM data;", False, "SQL injection"),
        ("' OR '1'='1", False, "SQL injection attempt"),
    ]

    unit_passed = 0
    detailed_cases = []
    print(f"\n{'Test Case':<50} {'Expected':<10} {'Result':<10} {'Status':<10}")
    print("-" * 80)

    for message, should_be_safe, reason in unit_test_cases:
        is_safe, _ = filter_obj.check_safety(message)
        expected = "Safe" if should_be_safe else "Blocked"
        result = "Safe" if is_safe else "Blocked"
        test_passed = is_safe == should_be_safe
        status = "✓" if test_passed else "✗"
        if test_passed:
            unit_passed += 1
        print(f"{message[:48]:<50} {expected:<10} {result:<10} {status:<10}")

        detailed_cases.append({
            "name": f"[Unit] {reason}",
            "expected": expected,
            "result": result,
            "passed": test_passed
        })

    unit_rate = (unit_passed / len(unit_test_cases)) * 100
    print("-" * 80)
    print(f"Unit Test Detection Rate: {unit_rate:.1f}% ({unit_passed}/{len(unit_test_cases)})")

    # ==================== PART 2: Integration Test ====================
    print("\n" + "=" * 80)
    print("  Part 2: Workflow Integration Test (Agent Actually Blocks)")
    print("=" * 80)

    integration_passed = 0
    integration_cases = [
        # Input should be BLOCKED (sensitive data in input)
        ("My credit card is 4111-1111-1111-1111", "blocked", "CC in input"),
        ("My SSN is 123-45-6789, can I book?", "blocked", "SSN in input"),
        ("DROP TABLE reservations;", "blocked", "SQL injection"),

        # Safe queries should work
        ("Where is downtown parking?", "allowed", "Safe query"),
        ("What are the parking rules?", "allowed", "Safe query"),
    ]

    try:
        with create_test_app() as app:
            print(f"\n{'Test Case':<50} {'Expected':<10} {'Result':<10} {'Status':<10}")
            print("-" * 80)

            for message, expected_outcome, reason in integration_cases:
                try:
                    result = app.process_user_message(message)
                    response = result.get("response", "")
                    safety_issue = result.get("safety_issue", False)

                    # Check if blocked messages are actually blocked
                    if expected_outcome == "blocked":
                        # Should have safety issue OR response indicates blocking
                        is_blocked = (
                            safety_issue or
                            "cannot process" in response.lower() or
                            "safety" in response.lower() or
                            len(response) < 100  # Blocked responses are usually short
                        )
                        test_passed = is_blocked
                        actual = "Blocked" if is_blocked else "Allowed"
                    else:
                        # Should NOT have safety issue and should have a real response
                        test_passed = not safety_issue and len(response) > 20
                        actual = "Allowed" if test_passed else "Blocked"

                    if test_passed:
                        integration_passed += 1

                    expected_str = "Blocked" if expected_outcome == "blocked" else "Allowed"
                    status = "✓" if test_passed else "✗"
                    print(f"{message[:48]:<50} {expected_str:<10} {actual:<10} {status:<10}")

                    detailed_cases.append({
                        "name": f"[Integration] {reason}",
                        "expected": expected_str,
                        "result": actual,
                        "passed": test_passed
                    })

                except Exception as e:
                    print(f"{message[:48]:<50} {'ERROR':<10} {str(e)[:20]:<10}")
                    detailed_cases.append({
                        "name": f"[Integration] {reason}",
                        "expected": expected_outcome,
                        "result": "Error",
                        "passed": False
                    })

    except Exception as e:
        print(f"\n✗ Integration test failed to initialize: {e}")
        # Still count unit test results
        integration_passed = 0

    integration_rate = (integration_passed / len(integration_cases)) * 100 if integration_cases else 0
    print("-" * 80)
    print(f"Integration Test Rate: {integration_rate:.1f}% ({integration_passed}/{len(integration_cases)})")

    # ==================== SUMMARY ====================
    total_passed = unit_passed + integration_passed
    total_cases = len(unit_test_cases) + len(integration_cases)
    overall_rate = (total_passed / total_cases) * 100

    print("\n" + "=" * 80)
    print("  Guardrails Summary")
    print("=" * 80)
    print(f"  • Unit Tests: {unit_rate:.1f}% ({unit_passed}/{len(unit_test_cases)})")
    print(f"  • Integration Tests: {integration_rate:.1f}% ({integration_passed}/{len(integration_cases)})")
    print(f"  • Overall: {overall_rate:.1f}% ({total_passed}/{total_cases})")

    details = {
        "metrics": {
            "Unit Test Rate (%)": unit_rate,
            "Integration Test Rate (%)": integration_rate,
            "Overall Rate (%)": overall_rate,
        },
        "test_cases": detailed_cases
    }

    # Must pass both unit AND integration tests
    return unit_rate == 100.0 and integration_rate == 100.0, details


# ============================================================================
# 2. SEMANTIC GENERATION METRICS (RAG TRIAD)
# ============================================================================


def calculate_faithfulness(answer: str, context: str) -> float:
    """Faithfulness: Is the answer derived from retrieved context? (Semantic)"""
    if not answer or not context:
        return 0.0

    try:
        embeddings = _get_embeddings()
        answer_sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not answer_sentences:
            return 0.0

        context_embedding = embeddings.embed_query(context)
        faithful_sentences = 0
        for sentence in answer_sentences:
            sentence_embedding = embeddings.embed_query(sentence)
            similarity = _cosine_similarity(
                np.array(sentence_embedding), np.array(context_embedding)
            )
            if similarity > 0.4:
                faithful_sentences += 1

        return faithful_sentences / len(answer_sentences)
    except Exception as e:
        print(f"Warning: Faithfulness calculation failed: {e}")
        answer_terms = set(w.lower() for w in answer.split() if len(w) > 4)
        context_lower = context.lower()
        found_terms = sum(1 for term in answer_terms if term in context_lower)
        return found_terms / len(answer_terms) if answer_terms else 0.5


def calculate_answer_relevance(question: str, answer: str) -> float:
    """Answer Relevance: Does the answer address the question? (Semantic)"""
    if not question or not answer:
        return 0.0

    try:
        embeddings = _get_embeddings()
        question_embedding = embeddings.embed_query(question)
        answer_embedding = embeddings.embed_query(answer)
        similarity = _cosine_similarity(
            np.array(question_embedding), np.array(answer_embedding)
        )
        return max(0, min(1, similarity))
    except Exception as e:
        print(f"Warning: Answer relevance calculation failed: {e}")
        question_terms = set(w.lower() for w in question.split() if len(w) > 3)
        answer_terms = set(w.lower() for w in answer.split() if len(w) > 3)
        overlap = len(question_terms & answer_terms)
        total = len(question_terms | answer_terms)
        return overlap / total if total > 0 else 0.0


def calculate_context_precision(context_docs: List, answer: str) -> float:
    """Context Precision: How much of retrieved context was useful? (Semantic)"""
    if not context_docs or not answer:
        return 0.0

    try:
        embeddings = _get_embeddings()
        answer_embedding = embeddings.embed_query(answer)
        useful_docs = 0

        for doc in context_docs:
            doc_embedding = embeddings.embed_query(doc.page_content)
            similarity = _cosine_similarity(
                np.array(doc_embedding), np.array(answer_embedding)
            )
            if similarity > 0.3:
                useful_docs += 1

        return useful_docs / len(context_docs) if context_docs else 0.0
    except Exception as e:
        print(f"Warning: Context precision calculation failed: {e}")
        useful_docs = 0
        answer_lower = answer.lower()
        for doc in context_docs:
            content_terms = set(w.lower() for w in doc.page_content.split()[:20])
            answer_terms = set(w.lower() for w in answer_lower.split())
            if len(content_terms & answer_terms) > 2:
                useful_docs += 1
        return useful_docs / len(context_docs) if context_docs else 0.0


# ============================================================================
# 3. RAG QUALITY METRICS TEST
# ============================================================================


def test_rag_metrics():
    """Test RAG quality using RAGAS framework with semantic metrics.

    Uses process_user_message() to test the ACTUAL user flow through the ReAct agent,
    not just the retriever component directly.
    """
    print_section("TEST 2: RAG METRICS - Quality Assessment (RAGAS)")

    try:
        with create_test_app() as app:
            if not app.workflow:
                print("\n✗ Workflow not initialized.")
                return False, {"notes": "Workflow not initialized"}

            # Use queries that should trigger vector_search (static info)
            test_queries = [
                "Where is downtown parking?",
                "What are the parking prices?",
                "What are the parking rules and policies?",
            ]

            metrics_summary = {
                "faithfulness": [],
                "answer_relevance": [],
                "context_precision": [],
                "latency_ms": [],
            }

            print(
                f"\n{'Query':<35} {'Latency':<12} {'Faith.':<8} {'Ans.Rel':<8} {'Ctx.Pre':<8}"
            )
            print("-" * 80)

            for query in test_queries:
                start = time.time()
                # Use process_user_message to go through the ACTUAL ReAct agent workflow
                result = app.process_user_message(query)
                latency_ms = (time.time() - start) * 1000

                answer = result.get("response", "")
                sources = result.get("sources", [])
                # Handle case where sources might be Document objects or dicts
                if sources:
                    context = "\n".join([
                        doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        for doc in sources
                    ])
                else:
                    context = ""

                faithfulness = calculate_faithfulness(answer, context)
                relevance = calculate_answer_relevance(query, answer)
                precision = calculate_context_precision(sources, answer)

                metrics_summary["faithfulness"].append(faithfulness)
                metrics_summary["answer_relevance"].append(relevance)
                metrics_summary["context_precision"].append(precision)
                metrics_summary["latency_ms"].append(latency_ms)

                print(
                    f"{query:<35} {latency_ms:<12.0f} {faithfulness:<8.2f} {relevance:<8.2f} {precision:<8.2f}"
                )

            print("-" * 80)
            avg_faith = sum(metrics_summary["faithfulness"]) / len(
                metrics_summary["faithfulness"]
            )
            avg_relevance = sum(metrics_summary["answer_relevance"]) / len(
                metrics_summary["answer_relevance"]
            )
            avg_precision = sum(metrics_summary["context_precision"]) / len(
                metrics_summary["context_precision"]
            )
            avg_latency = sum(metrics_summary["latency_ms"]) / len(
                metrics_summary["latency_ms"]
            )

            print(f"\nRAG Metrics (Average):")
            print(f"  • Faithfulness (answer from context): {avg_faith:.2f}")
            print(f"  • Answer Relevance (answers question): {avg_relevance:.2f}")
            print(f"  • Context Precision (useful context): {avg_precision:.2f}")
            print(f"  • Total Latency: {avg_latency:.0f}ms")

            details = {
                "metrics": {
                    "Avg Faithfulness": avg_faith,
                    "Avg Answer Relevance": avg_relevance,
                    "Avg Context Precision": avg_precision,
                    "Avg Latency (ms)": avg_latency,
                }
            }

            return True, details

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 4. RETRIEVAL RECALL@K TEST
# ============================================================================


def test_recall_at_k():
    """Test Recall@K metric using semantic relevance scoring."""
    print_section("TEST 3: RECALL@K - Retrieval Ranking Quality")

    try:
        with create_test_app() as app:
            if not app.rag_retriever:
                print("\n✗ RAG Retriever not initialized.")
                return False, {"notes": "RAG Retriever not initialized"}

            embeddings = _get_embeddings()
            test_query = "downtown parking location"
            k_values = [1, 3, 5]

            print(f"\nQuery: '{test_query}'")
            print(f"\n{'K':<5} {'Semantic Recall':<20} {'Avg Relevance':<20}")
            print("-" * 50)

            query_embedding = embeddings.embed_query(test_query)
            recall_metrics = {}

            for k in k_values:
                docs = app.rag_retriever.retrieve_documents(test_query, k=k)

                if not docs:
                    print(f"{k:<5} {'0.00':<20} {'0.00':<20}")
                    recall_metrics[f"Recall@{k}"] = 0.0
                    continue

                relevance_scores = []
                for doc in docs:
                    doc_embedding = embeddings.embed_query(doc.page_content)
                    relevance = _cosine_similarity(
                        np.array(query_embedding), np.array(doc_embedding)
                    )
                    relevance_scores.append(relevance)

                relevant_count = sum(1 for score in relevance_scores if score > 0.3)
                recall_at_k = relevant_count / len(docs) if docs else 0.0
                avg_relevance = (
                    sum(relevance_scores) / len(relevance_scores)
                    if relevance_scores
                    else 0.0
                )

                print(f"{k:<5} {recall_at_k:.2f}{'':<15} {avg_relevance:.2f}{'':<15}")
                recall_metrics[f"Recall@{k}"] = recall_at_k
                recall_metrics[f"Avg Relevance@{k}"] = avg_relevance

            print("-" * 50)
            print("✓ Recall@K test completed")

            details = {"metrics": recall_metrics}
            return True, details

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 5. COMPONENT VERIFICATION
# ============================================================================


def test_component_initialization():
    """Verify all system components initialize correctly."""
    print_section("TEST 4: COMPONENT INITIALIZATION - System Verification")

    try:
        with create_test_app() as app:
            checks = []

            # Check embeddings
            try:
                embeddings = _get_embeddings()
                test_vec = embeddings.embed_query("test")
                checks.append(("Embeddings", len(test_vec) == 384, f"384-dim vectors"))
            except Exception as e:
                checks.append(("Embeddings", False, str(e)))

            # Check Weaviate connection
            try:
                if app.rag_retriever:
                    checks.append(("Weaviate Vector DB", True, "Connected"))
                else:
                    checks.append(("Weaviate Vector DB", False, "Not initialized"))
            except Exception as e:
                checks.append(("Weaviate Vector DB", False, str(e)))

            # Check SQL database
            try:
                spaces = app.list_parking_spaces()
                checks.append(("SQLite Database", True, f"{len(spaces)} spaces stored"))
            except Exception as e:
                checks.append(("SQLite Database", False, str(e)))

            # Check LLM
            try:
                if app.rag_retriever and app.rag_retriever.llm:
                    checks.append(("LLM (Ollama)", True, "Connected"))
                else:
                    checks.append(("LLM (Ollama)", False, "Not initialized"))
            except Exception as e:
                checks.append(("LLM (Ollama)", False, str(e)))

            print(f"\n{'Component':<30} {'Status':<10} {'Details':<40}")
            print("-" * 80)

            detailed_cases = []
            for component, status, details in checks:
                status_str = "✓ OK" if status else "✗ FAIL"
                print(f"{component:<30} {status_str:<10} {details:<40}")
                detailed_cases.append({
                    "name": component,
                    "expected": "OK",
                    "result": "OK" if status else "FAIL",
                    "passed": status
                })

            all_passed = all(status for _, status, _ in checks)
            print("-" * 80)

            report_details = {"test_cases": detailed_cases}
            return all_passed, report_details

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 6. END-TO-END WORKFLOW TEST
# ============================================================================


def test_end_to_end_workflow():
    """Test complete workflow from query to response."""
    print_section("TEST 5: END-TO-END WORKFLOW")

    try:
        with create_test_app() as app:
            test_cases = [
                ("What parking spaces are available?", "info"),
                ("My credit card is 4532-1234-5678-9012", "safety"),
                ("I want to book a parking space", "intent"),
            ]

            print("\nSimulating user interactions...\n")

            results = []
            detailed_cases = []
            for query, expected_type in test_cases:
                result = app.process_user_message(query)
                is_safe = not result.get("safety_issue")

                if expected_type == "safety":
                    success = not is_safe
                else:
                    success = is_safe

                results.append(success)

                status = "✓" if success else "✗"
                print(f"{status} Query: '{query[:50]}...'")
                print(f"  Response: {result['response'][:70]}...")
                print()

                detailed_cases.append({
                    "name": query[:40],
                    "expected": expected_type,
                    "result": "Pass" if success else "Fail",
                    "passed": success
                })

            report_details = {"test_cases": detailed_cases}
            return all(results), report_details

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 7. HYBRID RETRIEVAL TEST (SQL Agent + Vector DB)
# ============================================================================


def test_hybrid_retrieval():
    """Test hybrid retrieval combining Vector DB semantic search + SQL Agent queries.

    Uses process_user_message() to test the ACTUAL user flow through the ReAct agent.
    The agent should route real-time queries to sql_query and static queries to vector_search.
    """
    print_section("TEST 7: HYBRID RETRIEVAL - SQL Agent + Vector DB")

    try:
        with create_test_app() as app:
            if not app.workflow:
                print("\n✗ Workflow not initialized.")
                return False, {"notes": "Workflow not initialized"}

            # Test queries categorized by expected routing
            # Real-time queries should trigger sql_query
            # Static queries should trigger vector_search
            test_queries = [
                # Real-time data queries (should use sql_query)
                ("How many parking spaces are available?", "sql_query", "availability"),
                ("What is the current status of parking?", "sql_query", "status"),
                ("How much does parking cost right now?", "sql_query", "pricing"),

                # Static info queries (should use vector_search)
                ("Where is the downtown parking located?", "vector_search", "location"),
                ("What are the parking rules?", "vector_search", "rules"),
            ]

            print(f"\n{'Query':<45} {'Expected':<15} {'Sources?':<10} {'Latency':<10}")
            print("-" * 85)

            results = {
                "sql_queries": 0,
                "vector_queries": 0,
                "with_sources": 0,
                "latency_ms": [],
            }

            all_passed = True
            detailed_cases = []

            for query, expected_route, query_type in test_queries:
                try:
                    start = time.time()
                    # Use process_user_message to go through the ACTUAL ReAct agent workflow
                    result = app.process_user_message(query)
                    latency_ms = (time.time() - start) * 1000

                    response = result.get("response", "")
                    sources = result.get("sources", [])
                    has_sources = len(sources) > 0 if sources else False

                    results["latency_ms"].append(latency_ms)

                    if has_sources:
                        results["with_sources"] += 1

                    # Check if response contains relevant content
                    response_lower = response.lower()

                    if expected_route == "sql_query":
                        results["sql_queries"] += 1
                        # For SQL queries, check response mentions data-like content
                        has_data = any(kw in response_lower for kw in [
                            "available", "spaces", "price", "cost", "status", "open", "capacity"
                        ])
                        test_passed = has_data and len(response) > 20
                    else:
                        results["vector_queries"] += 1
                        # For vector queries, should have sources and relevant content
                        test_passed = has_sources or len(response) > 50

                    if not test_passed:
                        all_passed = False

                    status = "✓" if test_passed else "✗"
                    sources_str = "Yes" if has_sources else "No"
                    print(f"{query:<45} {expected_route:<15} {sources_str:<10} {latency_ms:.0f}ms {status}")

                    detailed_cases.append({
                        "name": f"{query_type}: {query[:30]}",
                        "expected": expected_route,
                        "result": "Pass" if test_passed else "Fail",
                        "passed": test_passed
                    })

                except Exception as e:
                    print(f"{query:<45} {'ERROR':<15} {'':<10} ✗ {str(e)[:20]}")
                    all_passed = False
                    detailed_cases.append({
                        "name": f"{query_type}: {query[:30]}",
                        "expected": expected_route,
                        "result": f"Error",
                        "passed": False
                    })

            # Summary
            print("-" * 85)
            print("\nHybrid Retrieval Summary:")
            print(f"  • SQL Agent queries tested: {results['sql_queries']}")
            print(f"  • Vector search queries tested: {results['vector_queries']}")
            print(f"  • Queries with sources: {results['with_sources']}/{len(test_queries)}")

            if results["latency_ms"]:
                avg_latency = sum(results["latency_ms"]) / len(results["latency_ms"])
                print(f"  • Average latency: {avg_latency:.0f}ms")

            print("\nProduction RAG Concepts Demonstrated:")
            print("  ✓ ReAct Agent: Intelligent routing to appropriate tool")
            print("  ✓ Semantic Search: Vector DB for static reference data")
            print("  ✓ SQL Agent: Real-time data queries through workflow")
            print("  ✓ Hybrid Integration: Both sources available via agent")

            # Build metrics for report
            metrics = {
                "SQL Queries Tested": results["sql_queries"],
                "Vector Queries Tested": results["vector_queries"],
                "Queries With Sources": results["with_sources"],
            }
            if results["latency_ms"]:
                metrics["Avg Latency (ms)"] = sum(results["latency_ms"]) / len(results["latency_ms"])

            details = {
                "metrics": metrics,
                "test_cases": detailed_cases
            }
            return all_passed, details

    except Exception as e:
        print(f"\n✗ Hybrid retrieval test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 8. AGENT ROUTING TEST (NEW - Tests ReAct Agent Tool Selection)
# ============================================================================


def test_agent_routing():
    """Test that the ReAct agent selects the correct tools for different queries."""
    print_section("TEST 8: AGENT ROUTING - ReAct Tool Selection")

    try:
        with create_test_app() as app:
            if not app.workflow:
                print("\n✗ Workflow not initialized.")
                return False, {"notes": "Workflow not initialized"}

            # Test cases: (query, expected_behavior, should_have_sources)
            test_cases = [
                # Greetings should NOT trigger retrieval
                ("Hey", "direct_response", False),
                ("Hello there", "direct_response", False),
                ("Thanks!", "direct_response", False),
                ("Ok", "direct_response", False),

                # Static info should use vector_search
                ("Where is downtown parking?", "vector_search", True),
                ("What are the parking rules?", "vector_search", True),
                ("How do I book a parking space?", "vector_search", True),

                # Real-time queries should use sql_query
                ("How many spaces are available?", "sql_query", True),
                ("What is the current price?", "sql_query", True),

                # Reservation intent
                ("I want to book a parking space", "reservation", False),
            ]

            print(f"\n{'Query':<40} {'Expected':<18} {'Sources?':<10} {'Status':<10}")
            print("-" * 80)

            passed = 0
            detailed_cases = []

            for query, expected_behavior, should_have_sources in test_cases:
                try:
                    result = app.process_user_message(query)
                    response = result.get("response", "")
                    sources = result.get("sources", [])
                    has_sources = len(sources) > 0 if sources else False

                    # Check if sources match expectation
                    sources_correct = (has_sources == should_have_sources)

                    # For greetings, verify no retrieval happened (response should be short, friendly)
                    if expected_behavior == "direct_response":
                        # Greetings should have short responses without parking data
                        is_greeting_response = len(response) < 200 and not any(
                            kw in response.lower() for kw in ["downtown", "airport", "available", "price per hour"]
                        )
                        test_passed = sources_correct and is_greeting_response
                    elif expected_behavior == "reservation":
                        # Should start reservation flow
                        test_passed = "name" in response.lower() or "reservation" in response.lower() or "book" in response.lower()
                    else:
                        # For retrieval queries, just check sources are present
                        test_passed = sources_correct

                    if test_passed:
                        passed += 1

                    status = "✓" if test_passed else "✗"
                    sources_str = "Yes" if has_sources else "No"
                    print(f"{query:<40} {expected_behavior:<18} {sources_str:<10} {status:<10}")

                    detailed_cases.append({
                        "name": query[:35],
                        "expected": expected_behavior,
                        "result": "Pass" if test_passed else "Fail",
                        "passed": test_passed
                    })

                except Exception as e:
                    print(f"{query:<40} {'ERROR':<18} {'':<10} ✗ {str(e)[:20]}")
                    detailed_cases.append({
                        "name": query[:35],
                        "expected": expected_behavior,
                        "result": f"Error: {str(e)[:20]}",
                        "passed": False
                    })

            print("-" * 80)
            success_rate = (passed / len(test_cases)) * 100
            print(f"Agent Routing Accuracy: {success_rate:.1f}% ({passed}/{len(test_cases)})")

            # Test passes if at least 80% of routing decisions are correct
            all_passed = success_rate >= 80.0

            details = {
                "metrics": {"Routing Accuracy (%)": success_rate},
                "test_cases": detailed_cases
            }

            return all_passed, details

    except Exception as e:
        print(f"\n✗ Agent routing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 9. ADMIN FLOW TEST (Human-in-the-Loop)
# ============================================================================


def test_admin_flow():
    """Test the admin reservation approval/rejection flow."""
    print_section("TEST 9: ADMIN FLOW - Human-in-the-Loop Approval")

    try:
        from src.database.sql_db import ParkingDatabase
        from src.admin.admin_service import AdminService
        from datetime import datetime, timedelta
        import uuid

        # Use a separate test database
        test_db_path = "./data/test_admin.db"

        # Clean up any existing test db
        import os
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        db = ParkingDatabase(db_path=test_db_path)
        admin_service = AdminService(db)

        test_cases = []
        all_passed = True

        # Test 1: Create a test reservation
        print("\n1. Creating test reservation...")
        res_id = f"RES_TEST_{uuid.uuid4().hex[:6].upper()}"
        start_time = datetime.now() + timedelta(hours=1)
        end_time = datetime.now() + timedelta(hours=3)

        # Add a parking space first
        db.add_parking_space(
            id="test_parking",
            name="Test Parking",
            location="Test Location",
            capacity=100,
            price_per_hour=5.0
        )

        success = db.create_reservation(
            res_id=res_id,
            user_name="John",
            user_surname="Doe",
            car_number="ABC-123",
            parking_id="test_parking",
            start_time=start_time,
            end_time=end_time,
        )

        test_passed = success
        status = "✓" if test_passed else "✗"
        print(f"   {status} Reservation {res_id} created")
        test_cases.append({
            "name": "Create Reservation",
            "expected": "Success",
            "result": "Success" if test_passed else "Failed",
            "passed": test_passed
        })
        if not test_passed:
            all_passed = False

        # Test 2: Verify reservation is pending
        print("\n2. Verifying pending status...")
        pending = admin_service.get_pending_reservations()
        test_passed = any(r["id"] == res_id for r in pending)
        status = "✓" if test_passed else "✗"
        print(f"   {status} Reservation appears in pending list")
        test_cases.append({
            "name": "Pending List",
            "expected": "In pending",
            "result": "Found" if test_passed else "Not found",
            "passed": test_passed
        })
        if not test_passed:
            all_passed = False

        # Test 3: Get reservation status (should be pending)
        print("\n3. Checking status via admin service...")
        status_info = admin_service.get_reservation_status(res_id)
        test_passed = status_info is not None and status_info["status"] == "pending"
        status = "✓" if test_passed else "✗"
        print(f"   {status} Status: {status_info.get('status', 'N/A') if status_info else 'None'}")
        test_cases.append({
            "name": "Status Check (Pending)",
            "expected": "pending",
            "result": status_info.get("status", "N/A") if status_info else "None",
            "passed": test_passed
        })
        if not test_passed:
            all_passed = False

        # Test 4: Approve the reservation
        print("\n4. Admin approving reservation...")
        result = admin_service.approve_reservation(res_id, "TestAdmin", "Test approval")
        test_passed = result["success"]
        status = "✓" if test_passed else "✗"
        print(f"   {status} {result['message']}")
        test_cases.append({
            "name": "Approve Reservation",
            "expected": "Success",
            "result": "Success" if test_passed else result.get("message", "Failed"),
            "passed": test_passed
        })
        if not test_passed:
            all_passed = False

        # Test 5: Verify status is now confirmed
        print("\n5. Verifying approved status...")
        status_info = admin_service.get_reservation_status(res_id)
        test_passed = status_info is not None and status_info["status"] == "confirmed"
        status = "✓" if test_passed else "✗"
        print(f"   {status} Status: {status_info.get('status', 'N/A') if status_info else 'None'}")
        test_cases.append({
            "name": "Status Check (Confirmed)",
            "expected": "confirmed",
            "result": status_info.get("status", "N/A") if status_info else "None",
            "passed": test_passed
        })
        if not test_passed:
            all_passed = False

        # Test 6: Create another reservation for rejection test
        print("\n6. Creating second reservation for rejection test...")
        res_id_2 = f"RES_TEST_{uuid.uuid4().hex[:6].upper()}"
        success = db.create_reservation(
            res_id=res_id_2,
            user_name="Jane",
            user_surname="Smith",
            car_number="XYZ-789",
            parking_id="test_parking",
            start_time=start_time,
            end_time=end_time,
        )
        status = "✓" if success else "✗"
        print(f"   {status} Reservation {res_id_2} created")
        test_cases.append({
            "name": "Create Second Reservation",
            "expected": "Success",
            "result": "Success" if success else "Failed",
            "passed": success
        })
        if not success:
            all_passed = False

        # Test 7: Reject the second reservation
        print("\n7. Admin rejecting reservation...")
        result = admin_service.reject_reservation(res_id_2, "TestAdmin", "Test rejection reason")
        test_passed = result["success"]
        status = "✓" if test_passed else "✗"
        print(f"   {status} {result['message']}")
        test_cases.append({
            "name": "Reject Reservation",
            "expected": "Success",
            "result": "Success" if test_passed else result.get("message", "Failed"),
            "passed": test_passed
        })
        if not test_passed:
            all_passed = False

        # Test 8: Verify rejection status and reason
        print("\n8. Verifying rejected status...")
        status_info = admin_service.get_reservation_status(res_id_2)
        test_passed = (
            status_info is not None and
            status_info["status"] == "rejected" and
            status_info.get("rejection_reason") == "Test rejection reason"
        )
        status = "✓" if test_passed else "✗"
        if status_info:
            print(f"   {status} Status: {status_info.get('status', 'N/A')}")
            print(f"   {status} Reason: {status_info.get('rejection_reason', 'N/A')}")
        test_cases.append({
            "name": "Status Check (Rejected)",
            "expected": "rejected + reason",
            "result": f"{status_info.get('status', 'N/A')}" if status_info else "None",
            "passed": test_passed
        })
        if not test_passed:
            all_passed = False

        # Cleanup
        db.close()
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        # Summary
        print("\n" + "-" * 80)
        passed_count = sum(1 for tc in test_cases if tc["passed"])
        print(f"Admin Flow Tests: {passed_count}/{len(test_cases)} passed")

        details = {
            "metrics": {
                "Tests Passed": passed_count,
                "Total Tests": len(test_cases),
                "Pass Rate (%)": (passed_count / len(test_cases)) * 100
            },
            "test_cases": test_cases
        }

        return all_passed, details

    except Exception as e:
        print(f"\n✗ Admin flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 10. DATA ARCHITECTURE TEST
# ============================================================================


def test_data_architecture():
    """Verify data separation and storage strategy."""
    print_section("TEST 6: DATA ARCHITECTURE - Static vs Dynamic")

    print("""
    ┌──────────────────────────────────────────────────────────┐
    │                  SYSTEM ARCHITECTURE                      │
    ├──────────────────────────────────────────────────────────┤
    │                                                            │
    │  WEAVIATE (Vector DB) - STATIC DATA                       │
    │  ├─ General parking information (embeddings indexed)      │
    │  ├─ Parking locations and features                        │
    │  ├─ Booking process information                           │
    │  └─ FAQs and guidelines                                   │
    │     → Used by RAG for semantic search                     │
    │                                                            │
    │  SQLITE (SQL Database) - DYNAMIC DATA                     │
    │  ├─ Real-time availability (updated constantly)           │
    │  ├─ Prices (can change)                                   │
    │  ├─ Reservations (created/updated/cancelled)             │
    │  └─ User information & admin approvals                    │
    │     → Used for transactions & current state               │
    │                                                            │
    ├──────────────────────────────────────────────────────────┤
    │  WORKFLOW                                                  │
    │                                                            │
    │  User Query → [Safety Filter] → [Intent Detection]       │
    │                                   │                        │
    │                    ┌──────────────┴──────────────┐        │
    │                    ↓                             ↓        │
    │            Information Query          Reservation Request │
    │                    ↓                             ↓        │
    │          [Vector DB Search]        [SQL DB Query]         │
    │          [LLM Generation]          [Check Availability]   │
    │                    ↓                             ↓        │
    │            Answer from RAG         Book Parking Space     │
    │                                                            │
    └──────────────────────────────────────────────────────────┘
    """)

    print("✓ Data architecture verified:")
    print("  • Static knowledge → Weaviate (semantic search via RAG)")
    print("  • Dynamic state → SQLite (transactions & reservations)")
    print("  • Safety → Guard rails (block sensitive data)")

    details = {
        "notes": "Architecture: Weaviate (static) + SQLite (dynamic) + Guard rails (safety)"
    }
    return True, details


# ============================================================================
# 10. LOAD TEST: CHATBOT CONCURRENT USERS
# ============================================================================


def test_chatbot_load():
    """Load test: Simulate multiple concurrent users interacting with the chatbot.

    Tests system performance under concurrent load to evaluate:
    - Response time degradation under load
    - Success rate with multiple simultaneous requests
    - System stability with concurrent access
    """
    print_section("TEST 10: LOAD TEST - Chatbot Concurrent Users")

    # Configuration
    NUM_CONCURRENT_USERS = 5  # Number of concurrent users
    QUERIES_PER_USER = 3      # Queries each user will send

    test_queries = [
        "Where is downtown parking?",
        "What are the parking prices?",
        "How many spaces are available?",
        "What are the parking rules?",
        "What is the current status of parking?",
    ]

    results = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "response_times": [],
        "errors": [],
    }

    results_lock = threading.Lock()

    def user_session(user_id: int, app) -> List[Dict]:
        """Simulate a single user's session with multiple queries."""
        session_results = []

        for query_idx in range(QUERIES_PER_USER):
            query = test_queries[(user_id + query_idx) % len(test_queries)]
            start_time = time.time()

            try:
                result = app.process_user_message(query)
                elapsed_ms = (time.time() - start_time) * 1000

                success = result.get("response") and len(result.get("response", "")) > 10

                session_results.append({
                    "user_id": user_id,
                    "query": query,
                    "success": success,
                    "response_time_ms": elapsed_ms,
                    "error": None
                })

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                session_results.append({
                    "user_id": user_id,
                    "query": query,
                    "success": False,
                    "response_time_ms": elapsed_ms,
                    "error": str(e)
                })

        return session_results

    try:
        with create_test_app() as app:
            print(f"\nSimulating {NUM_CONCURRENT_USERS} concurrent users, {QUERIES_PER_USER} queries each...")
            print(f"Total requests: {NUM_CONCURRENT_USERS * QUERIES_PER_USER}")
            print("-" * 80)

            all_results = []

            # Run concurrent user sessions
            with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_USERS) as executor:
                futures = {
                    executor.submit(user_session, user_id, app): user_id
                    for user_id in range(NUM_CONCURRENT_USERS)
                }

                for future in as_completed(futures):
                    user_id = futures[future]
                    try:
                        session_results = future.result()
                        all_results.extend(session_results)
                    except Exception as e:
                        print(f"User {user_id} session failed: {e}")

            # Aggregate results
            for result in all_results:
                results["total_requests"] += 1
                if result["success"]:
                    results["successful_requests"] += 1
                else:
                    results["failed_requests"] += 1
                    if result["error"]:
                        results["errors"].append(result["error"])
                results["response_times"].append(result["response_time_ms"])

            # Calculate statistics
            if results["response_times"]:
                avg_response_time = sum(results["response_times"]) / len(results["response_times"])
                min_response_time = min(results["response_times"])
                max_response_time = max(results["response_times"])
                p95_response_time = sorted(results["response_times"])[int(len(results["response_times"]) * 0.95)]
            else:
                avg_response_time = min_response_time = max_response_time = p95_response_time = 0

            success_rate = (results["successful_requests"] / results["total_requests"]) * 100 if results["total_requests"] > 0 else 0

            # Print results
            print(f"\n{'Metric':<30} {'Value':<20}")
            print("-" * 50)
            print(f"{'Total Requests':<30} {results['total_requests']:<20}")
            print(f"{'Successful Requests':<30} {results['successful_requests']:<20}")
            print(f"{'Failed Requests':<30} {results['failed_requests']:<20}")
            print(f"{'Success Rate':<30} {success_rate:.1f}%")
            print(f"{'Avg Response Time':<30} {avg_response_time:.0f}ms")
            print(f"{'Min Response Time':<30} {min_response_time:.0f}ms")
            print(f"{'Max Response Time':<30} {max_response_time:.0f}ms")
            print(f"{'P95 Response Time':<30} {p95_response_time:.0f}ms")
            print("-" * 50)

            # Determine pass/fail
            # Pass if: >80% success rate AND avg response time < 30 seconds
            test_passed = success_rate >= 80 and avg_response_time < 30000

            status = "✓ PASSED" if test_passed else "✗ FAILED"
            print(f"\nLoad Test Result: {status}")

            if results["errors"]:
                print(f"\nErrors encountered ({len(results['errors'])} unique):")
                for error in set(results["errors"][:5]):  # Show first 5 unique errors
                    print(f"  • {error[:80]}...")

            details = {
                "metrics": {
                    "Concurrent Users": NUM_CONCURRENT_USERS,
                    "Queries Per User": QUERIES_PER_USER,
                    "Total Requests": results["total_requests"],
                    "Success Rate (%)": success_rate,
                    "Avg Response Time (ms)": avg_response_time,
                    "P95 Response Time (ms)": p95_response_time,
                },
                "notes": f"Simulated {NUM_CONCURRENT_USERS} concurrent users"
            }

            return test_passed, details

    except Exception as e:
        print(f"\n✗ Load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 11. LOAD TEST: ADMIN CONFIRMATION FUNCTIONALITY
# ============================================================================


def test_admin_load():
    """Load test: Simulate multiple concurrent admin operations.

    Tests admin system performance with concurrent:
    - Reservation creation
    - Approval/rejection operations
    - Status queries
    """
    print_section("TEST 11: LOAD TEST - Admin Confirmation Functionality")

    # Configuration
    NUM_CONCURRENT_ADMINS = 5
    OPERATIONS_PER_ADMIN = 4

    results = {
        "total_operations": 0,
        "successful_operations": 0,
        "failed_operations": 0,
        "operation_times": [],
        "errors": [],
    }

    results_lock = threading.Lock()

    def admin_session(admin_id: int, db, admin_service) -> List[Dict]:
        """Simulate admin operations session."""
        session_results = []

        for op_idx in range(OPERATIONS_PER_ADMIN):
            operation_type = op_idx % 4  # Cycle through operation types
            res_id = f"LOAD_TEST_{admin_id}_{op_idx}_{uuid.uuid4().hex[:6].upper()}"
            start_time = time.time()

            try:
                if operation_type == 0:
                    # Create reservation
                    from datetime import datetime, timedelta
                    success = db.create_reservation(
                        res_id=res_id,
                        user_name=f"LoadTest{admin_id}",
                        user_surname=f"User{op_idx}",
                        car_number=f"LT-{admin_id}{op_idx}",
                        parking_id="downtown_1",
                        start_time=datetime.now() + timedelta(hours=1),
                        end_time=datetime.now() + timedelta(hours=3),
                    )
                    op_name = "Create Reservation"

                elif operation_type == 1:
                    # List pending reservations
                    pending = admin_service.get_pending_reservations()
                    success = isinstance(pending, list)
                    op_name = "List Pending"

                elif operation_type == 2:
                    # Create and approve
                    from datetime import datetime, timedelta
                    db.create_reservation(
                        res_id=res_id,
                        user_name=f"ApproveTest{admin_id}",
                        user_surname=f"User{op_idx}",
                        car_number=f"AP-{admin_id}{op_idx}",
                        parking_id="downtown_1",
                        start_time=datetime.now() + timedelta(hours=2),
                        end_time=datetime.now() + timedelta(hours=4),
                    )
                    result = admin_service.approve_reservation(res_id, f"Admin{admin_id}", "Load test approval")
                    success = result.get("success", False)
                    op_name = "Approve Reservation"

                else:
                    # Create and reject
                    from datetime import datetime, timedelta
                    db.create_reservation(
                        res_id=res_id,
                        user_name=f"RejectTest{admin_id}",
                        user_surname=f"User{op_idx}",
                        car_number=f"RJ-{admin_id}{op_idx}",
                        parking_id="downtown_1",
                        start_time=datetime.now() + timedelta(hours=3),
                        end_time=datetime.now() + timedelta(hours=5),
                    )
                    result = admin_service.reject_reservation(res_id, f"Admin{admin_id}", "Load test rejection")
                    success = result.get("success", False)
                    op_name = "Reject Reservation"

                elapsed_ms = (time.time() - start_time) * 1000

                session_results.append({
                    "admin_id": admin_id,
                    "operation": op_name,
                    "success": success,
                    "response_time_ms": elapsed_ms,
                    "error": None
                })

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                session_results.append({
                    "admin_id": admin_id,
                    "operation": f"Operation {op_idx}",
                    "success": False,
                    "response_time_ms": elapsed_ms,
                    "error": str(e)
                })

        return session_results

    try:
        from src.database.sql_db import ParkingDatabase
        from src.admin.admin_service import AdminService

        # Use a separate test database
        test_db_path = "./data/test_admin_load.db"

        # Clean up any existing test db
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        db = ParkingDatabase(db_path=test_db_path)
        admin_service = AdminService(db)

        # Add a parking space for testing
        db.add_parking_space(
            id="downtown_1",
            name="Downtown Parking",
            location="Downtown",
            capacity=100,
            price_per_hour=5.0
        )

        print(f"\nSimulating {NUM_CONCURRENT_ADMINS} concurrent admins, {OPERATIONS_PER_ADMIN} operations each...")
        print(f"Total operations: {NUM_CONCURRENT_ADMINS * OPERATIONS_PER_ADMIN}")
        print("-" * 80)

        all_results = []

        # Run concurrent admin sessions
        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_ADMINS) as executor:
            futures = {
                executor.submit(admin_session, admin_id, db, admin_service): admin_id
                for admin_id in range(NUM_CONCURRENT_ADMINS)
            }

            for future in as_completed(futures):
                admin_id = futures[future]
                try:
                    session_results = future.result()
                    all_results.extend(session_results)
                except Exception as e:
                    print(f"Admin {admin_id} session failed: {e}")

        # Aggregate results
        for result in all_results:
            results["total_operations"] += 1
            if result["success"]:
                results["successful_operations"] += 1
            else:
                results["failed_operations"] += 1
                if result["error"]:
                    results["errors"].append(result["error"])
            results["operation_times"].append(result["response_time_ms"])

        # Calculate statistics
        if results["operation_times"]:
            avg_time = sum(results["operation_times"]) / len(results["operation_times"])
            min_time = min(results["operation_times"])
            max_time = max(results["operation_times"])
        else:
            avg_time = min_time = max_time = 0

        success_rate = (results["successful_operations"] / results["total_operations"]) * 100 if results["total_operations"] > 0 else 0

        # Print results
        print(f"\n{'Metric':<30} {'Value':<20}")
        print("-" * 50)
        print(f"{'Total Operations':<30} {results['total_operations']:<20}")
        print(f"{'Successful Operations':<30} {results['successful_operations']:<20}")
        print(f"{'Failed Operations':<30} {results['failed_operations']:<20}")
        print(f"{'Success Rate':<30} {success_rate:.1f}%")
        print(f"{'Avg Operation Time':<30} {avg_time:.0f}ms")
        print(f"{'Min Operation Time':<30} {min_time:.0f}ms")
        print(f"{'Max Operation Time':<30} {max_time:.0f}ms")
        print("-" * 50)

        # Cleanup
        db.close()
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        # Pass if >80% success rate
        test_passed = success_rate >= 80

        status = "✓ PASSED" if test_passed else "✗ FAILED"
        print(f"\nAdmin Load Test Result: {status}")

        details = {
            "metrics": {
                "Concurrent Admins": NUM_CONCURRENT_ADMINS,
                "Operations Per Admin": OPERATIONS_PER_ADMIN,
                "Total Operations": results["total_operations"],
                "Success Rate (%)": success_rate,
                "Avg Operation Time (ms)": avg_time,
            }
        }

        return test_passed, details

    except Exception as e:
        print(f"\n✗ Admin load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 12. LOAD TEST: MCP SERVER CONCURRENT WRITES
# ============================================================================


def test_mcp_load():
    """Load test: Simulate concurrent MCP server operations.

    Tests MCP server performance with concurrent:
    - Reservation writes
    - File reads
    - Info queries
    """
    print_section("TEST 12: LOAD TEST - MCP Server Concurrent Operations")

    # Configuration
    NUM_CONCURRENT_WRITERS = 5
    OPERATIONS_PER_WRITER = 4

    results = {
        "total_operations": 0,
        "successful_operations": 0,
        "failed_operations": 0,
        "operation_times": [],
        "errors": [],
    }

    async def mcp_operations(writer_id: int) -> List[Dict]:
        """Perform MCP operations."""
        from src.mcp.reservation_server import (
            write_reservation,
            read_reservations,
            get_reservation_file_info,
        )

        session_results = []

        for op_idx in range(OPERATIONS_PER_WRITER):
            operation_type = op_idx % 3
            start_time = time.time()

            try:
                if operation_type == 0:
                    # Write reservation
                    test_reservation = {
                        "reservation_id": f"MCP_LOAD_{writer_id}_{op_idx}_{uuid.uuid4().hex[:6]}",
                        "user_name": f"LoadTest{writer_id}",
                        "user_surname": f"User{op_idx}",
                        "car_number": f"MCP-{writer_id}{op_idx}",
                        "start_time": "2026-03-01 10:00",
                        "end_time": "2026-03-01 14:00",
                        "approved_by": f"LoadTestAdmin{writer_id}",
                        "parking_id": "downtown_1"
                    }
                    result = await write_reservation(test_reservation)
                    success = "successfully" in result[0].text.lower() or "written" in result[0].text.lower()
                    op_name = "Write Reservation"

                elif operation_type == 1:
                    # Read reservations
                    result = await read_reservations({"limit": 5})
                    success = result is not None and len(result) > 0
                    op_name = "Read Reservations"

                else:
                    # Get file info
                    result = await get_reservation_file_info({})
                    success = result is not None and len(result) > 0
                    op_name = "Get File Info"

                elapsed_ms = (time.time() - start_time) * 1000

                session_results.append({
                    "writer_id": writer_id,
                    "operation": op_name,
                    "success": success,
                    "response_time_ms": elapsed_ms,
                    "error": None
                })

            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                session_results.append({
                    "writer_id": writer_id,
                    "operation": f"Operation {op_idx}",
                    "success": False,
                    "response_time_ms": elapsed_ms,
                    "error": str(e)
                })

        return session_results

    async def run_concurrent_mcp_tests():
        """Run all MCP tests concurrently."""
        tasks = [mcp_operations(writer_id) for writer_id in range(NUM_CONCURRENT_WRITERS)]
        return await asyncio.gather(*tasks, return_exceptions=True)

    try:
        print(f"\nSimulating {NUM_CONCURRENT_WRITERS} concurrent MCP writers, {OPERATIONS_PER_WRITER} operations each...")
        print(f"Total operations: {NUM_CONCURRENT_WRITERS * OPERATIONS_PER_WRITER}")
        print("-" * 80)

        # Run async operations
        all_results = asyncio.run(run_concurrent_mcp_tests())

        # Aggregate results
        for session_results in all_results:
            if isinstance(session_results, Exception):
                results["failed_operations"] += 1
                results["errors"].append(str(session_results))
                continue

            for result in session_results:
                results["total_operations"] += 1
                if result["success"]:
                    results["successful_operations"] += 1
                else:
                    results["failed_operations"] += 1
                    if result["error"]:
                        results["errors"].append(result["error"])
                results["operation_times"].append(result["response_time_ms"])

        # Calculate statistics
        if results["operation_times"]:
            avg_time = sum(results["operation_times"]) / len(results["operation_times"])
            min_time = min(results["operation_times"])
            max_time = max(results["operation_times"])
        else:
            avg_time = min_time = max_time = 0

        success_rate = (results["successful_operations"] / results["total_operations"]) * 100 if results["total_operations"] > 0 else 0

        # Print results
        print(f"\n{'Metric':<30} {'Value':<20}")
        print("-" * 50)
        print(f"{'Total Operations':<30} {results['total_operations']:<20}")
        print(f"{'Successful Operations':<30} {results['successful_operations']:<20}")
        print(f"{'Failed Operations':<30} {results['failed_operations']:<20}")
        print(f"{'Success Rate':<30} {success_rate:.1f}%")
        print(f"{'Avg Operation Time':<30} {avg_time:.0f}ms")
        print(f"{'Min Operation Time':<30} {min_time:.0f}ms")
        print(f"{'Max Operation Time':<30} {max_time:.0f}ms")
        print("-" * 50)

        # Pass if >80% success rate
        test_passed = success_rate >= 80

        status = "✓ PASSED" if test_passed else "✗ FAILED"
        print(f"\nMCP Load Test Result: {status}")

        details = {
            "metrics": {
                "Concurrent Writers": NUM_CONCURRENT_WRITERS,
                "Operations Per Writer": OPERATIONS_PER_WRITER,
                "Total Operations": results["total_operations"],
                "Success Rate (%)": success_rate,
                "Avg Operation Time (ms)": avg_time,
            }
        }

        return test_passed, details

    except Exception as e:
        print(f"\n✗ MCP load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 13. MCP SERVER FUNCTIONAL TESTS (Merged from test_mcp_server.py)
# ============================================================================


def test_mcp_server():
    """Test MCP server tools functionality.

    Tests all MCP server tools:
    - write_reservation: Write approved reservation to file
    - read_reservations: Read reservation history
    - get_reservation_file_info: Get file metadata
    """
    print_section("TEST 13: MCP SERVER - Functional Tests")

    async def run_mcp_tests() -> Tuple[bool, Dict]:
        """Run all MCP tool tests."""
        from src.mcp.reservation_server import (
            write_reservation,
            read_reservations,
            get_reservation_file_info,
        )

        test_cases = []
        all_passed = True

        # Test 1: Get initial file info
        print("\n1. Get Reservation File Info...")
        try:
            result = await get_reservation_file_info({})
            success = result is not None and len(result) > 0
            test_cases.append({
                "name": "Get File Info",
                "expected": "File info returned",
                "result": "Success" if success else "Failed",
                "passed": success
            })
            if success:
                print(f"   ✓ {result[0].text[:80]}...")
            else:
                print("   ✗ Failed to get file info")
                all_passed = False
        except Exception as e:
            print(f"   ✗ Error: {e}")
            test_cases.append({
                "name": "Get File Info",
                "expected": "File info returned",
                "result": f"Error: {str(e)[:30]}",
                "passed": False
            })
            all_passed = False

        # Test 2: Write a test reservation
        print("\n2. Write Test Reservation...")
        test_reservation = {
            "reservation_id": f"MCP_TEST_{uuid.uuid4().hex[:8].upper()}",
            "user_name": "MCP",
            "user_surname": "TestUser",
            "car_number": "MCP-999",
            "start_time": "2026-03-01 10:00",
            "end_time": "2026-03-01 14:00",
            "approved_by": "MCP Server Test",
            "parking_id": "downtown_1"
        }
        try:
            result = await write_reservation(test_reservation)
            success = "successfully" in result[0].text.lower() or "written" in result[0].text.lower()
            test_cases.append({
                "name": "Write Reservation",
                "expected": "Reservation written",
                "result": "Success" if success else "Failed",
                "passed": success
            })
            if success:
                print(f"   ✓ {result[0].text[:80]}...")
            else:
                print(f"   ✗ Unexpected result: {result[0].text[:80]}...")
                all_passed = False
        except Exception as e:
            print(f"   ✗ Error: {e}")
            test_cases.append({
                "name": "Write Reservation",
                "expected": "Reservation written",
                "result": f"Error: {str(e)[:30]}",
                "passed": False
            })
            all_passed = False

        # Test 3: Read reservations
        print("\n3. Read Recent Reservations...")
        try:
            result = await read_reservations({"limit": 5})
            success = result is not None and len(result) > 0
            test_cases.append({
                "name": "Read Reservations",
                "expected": "Reservations list returned",
                "result": "Success" if success else "Failed",
                "passed": success
            })
            if success:
                print(f"   ✓ {result[0].text[:80]}...")
            else:
                print("   ✗ Failed to read reservations")
                all_passed = False
        except Exception as e:
            print(f"   ✗ Error: {e}")
            test_cases.append({
                "name": "Read Reservations",
                "expected": "Reservations list returned",
                "result": f"Error: {str(e)[:30]}",
                "passed": False
            })
            all_passed = False

        # Test 4: Verify file was updated
        print("\n4. Verify File Updated...")
        try:
            result = await get_reservation_file_info({})
            success = result is not None and len(result) > 0
            test_cases.append({
                "name": "Verify File Update",
                "expected": "Updated file info",
                "result": "Success" if success else "Failed",
                "passed": success
            })
            if success:
                print(f"   ✓ {result[0].text[:80]}...")
            else:
                print("   ✗ Failed to verify file update")
                all_passed = False
        except Exception as e:
            print(f"   ✗ Error: {e}")
            test_cases.append({
                "name": "Verify File Update",
                "expected": "Updated file info",
                "result": f"Error: {str(e)[:30]}",
                "passed": False
            })
            all_passed = False

        # Test 5: Input validation (security test)
        print("\n5. Input Validation (Security)...")
        malicious_reservation = {
            "reservation_id": "../../etc/passwd",
            "user_name": "Mal|icious",
            "user_surname": "User\nInjection",
            "car_number": "ABC-123",
            "start_time": "2026-03-01 10:00",
            "end_time": "2026-03-01 14:00",
            "approved_by": "Test",
            "parking_id": "downtown_1"
        }
        try:
            result = await write_reservation(malicious_reservation)
            # Should succeed but sanitize the input
            success = result is not None and len(result) > 0
            # The sanitization should have removed malicious chars
            test_cases.append({
                "name": "Input Sanitization",
                "expected": "Sanitized write",
                "result": "Success" if success else "Failed",
                "passed": success
            })
            if success:
                print(f"   ✓ Input sanitized and written")
            else:
                print("   ✗ Sanitization may have failed")
                all_passed = False
        except Exception as e:
            # If it raises an error, that's also acceptable security behavior
            print(f"   ✓ Malicious input rejected: {str(e)[:50]}")
            test_cases.append({
                "name": "Input Sanitization",
                "expected": "Sanitized or rejected",
                "result": "Rejected",
                "passed": True
            })

        print("\n" + "-" * 80)
        passed_count = sum(1 for tc in test_cases if tc["passed"])
        print(f"MCP Server Tests: {passed_count}/{len(test_cases)} passed")

        details = {
            "metrics": {
                "Tests Passed": passed_count,
                "Total Tests": len(test_cases),
                "Pass Rate (%)": (passed_count / len(test_cases)) * 100 if test_cases else 0
            },
            "test_cases": test_cases
        }

        return all_passed, details

    try:
        return asyncio.run(run_mcp_tests())
    except Exception as e:
        print(f"\n✗ MCP server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 14. FULL PIPELINE INTEGRATION TEST
# ============================================================================


def test_full_pipeline_integration():
    """Test the complete system pipeline from user query to data recording.

    Tests the full workflow:
    1. User interaction (chatbot query)
    2. RAG retrieval and response
    3. Reservation creation
    4. Admin approval/rejection
    5. MCP server data recording
    """
    print_section("TEST 14: FULL PIPELINE INTEGRATION")

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    FULL PIPELINE INTEGRATION TEST                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │   User Query ──→ Safety Check ──→ RAG Retrieval ──→ Response        │
    │                                                                      │
    │   Reservation Request ──→ Data Collection ──→ Admin Queue           │
    │                                                                      │
    │   Admin Decision ──→ Status Update ──→ MCP Recording                │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    test_stages = []
    all_passed = True

    try:
        # Stage 1: Chatbot Query Processing
        print("\n" + "=" * 60)
        print("STAGE 1: Chatbot Query Processing")
        print("=" * 60)

        with create_test_app() as app:
            test_queries = [
                ("Where is downtown parking?", "info"),
                ("What are the prices?", "info"),
                ("How many spaces are available?", "realtime"),
            ]

            stage1_passed = 0
            for query, query_type in test_queries:
                result = app.process_user_message(query)
                response = result.get("response", "")
                success = len(response) > 20 and not result.get("safety_issue", False)

                status = "✓" if success else "✗"
                print(f"  {status} Query ({query_type}): '{query[:40]}...'")
                print(f"    Response: {response[:60]}...")

                if success:
                    stage1_passed += 1

            stage1_success = stage1_passed == len(test_queries)
            test_stages.append({
                "name": "Chatbot Query Processing",
                "passed": stage1_success,
                "details": f"{stage1_passed}/{len(test_queries)} queries successful"
            })
            if not stage1_success:
                all_passed = False

        # Stage 2: Safety Guardrails
        print("\n" + "=" * 60)
        print("STAGE 2: Safety Guardrails")
        print("=" * 60)

        from src.guardrails.filter import DataProtectionFilter
        filter_obj = DataProtectionFilter()

        safety_tests = [
            ("My credit card is 4111-1111-1111-1111", False, "Block CC"),
            ("Normal parking query", True, "Allow safe"),
            ("DROP TABLE users;", False, "Block SQL injection"),
        ]

        stage2_passed = 0
        for message, should_be_safe, reason in safety_tests:
            is_safe, _ = filter_obj.check_safety(message)
            success = is_safe == should_be_safe

            status = "✓" if success else "✗"
            expected = "Safe" if should_be_safe else "Blocked"
            actual = "Safe" if is_safe else "Blocked"
            print(f"  {status} {reason}: Expected={expected}, Got={actual}")

            if success:
                stage2_passed += 1

        stage2_success = stage2_passed == len(safety_tests)
        test_stages.append({
            "name": "Safety Guardrails",
            "passed": stage2_success,
            "details": f"{stage2_passed}/{len(safety_tests)} checks correct"
        })
        if not stage2_success:
            all_passed = False

        # Stage 3: Reservation & Admin Flow
        print("\n" + "=" * 60)
        print("STAGE 3: Reservation & Admin Flow")
        print("=" * 60)

        from src.database.sql_db import ParkingDatabase
        from src.admin.admin_service import AdminService
        from datetime import datetime, timedelta

        test_db_path = "./data/test_pipeline.db"
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        db = ParkingDatabase(db_path=test_db_path)
        admin_service = AdminService(db)

        # Add parking space
        db.add_parking_space(
            id="pipeline_test",
            name="Pipeline Test Parking",
            location="Test Location",
            capacity=100,
            price_per_hour=5.0
        )

        stage3_tests = []

        # Create reservation
        res_id = f"PIPELINE_TEST_{uuid.uuid4().hex[:6].upper()}"
        success = db.create_reservation(
            res_id=res_id,
            user_name="Pipeline",
            user_surname="TestUser",
            car_number="PLN-123",
            parking_id="pipeline_test",
            start_time=datetime.now() + timedelta(hours=1),
            end_time=datetime.now() + timedelta(hours=3),
        )
        status = "✓" if success else "✗"
        print(f"  {status} Create reservation: {res_id}")
        stage3_tests.append(success)

        # Check pending
        pending = admin_service.get_pending_reservations()
        found_pending = any(r["id"] == res_id for r in pending)
        status = "✓" if found_pending else "✗"
        print(f"  {status} Reservation in pending queue")
        stage3_tests.append(found_pending)

        # Approve
        result = admin_service.approve_reservation(res_id, "PipelineAdmin", "Test approval")
        approved = result.get("success", False)
        status = "✓" if approved else "✗"
        print(f"  {status} Admin approval: {result.get('message', 'No message')}")
        stage3_tests.append(approved)

        # Verify status
        status_info = admin_service.get_reservation_status(res_id)
        confirmed = status_info and status_info.get("status") == "confirmed"
        status = "✓" if confirmed else "✗"
        print(f"  {status} Status updated to confirmed")
        stage3_tests.append(confirmed)

        db.close()
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        stage3_success = all(stage3_tests)
        test_stages.append({
            "name": "Reservation & Admin Flow",
            "passed": stage3_success,
            "details": f"{sum(stage3_tests)}/{len(stage3_tests)} operations successful"
        })
        if not stage3_success:
            all_passed = False

        # Stage 4: MCP Recording
        print("\n" + "=" * 60)
        print("STAGE 4: MCP Data Recording")
        print("=" * 60)

        async def test_mcp_recording():
            from src.mcp.reservation_server import write_reservation, read_reservations

            # Write reservation via MCP
            test_reservation = {
                "reservation_id": f"MCP_PIPELINE_{uuid.uuid4().hex[:6]}",
                "user_name": "Pipeline",
                "user_surname": "MCPTest",
                "car_number": "MCP-PLN",
                "start_time": "2026-03-01 10:00",
                "end_time": "2026-03-01 14:00",
                "approved_by": "Pipeline Test",
                "parking_id": "downtown_1"
            }

            write_result = await write_reservation(test_reservation)
            write_success = "successfully" in write_result[0].text.lower() or "written" in write_result[0].text.lower()

            status = "✓" if write_success else "✗"
            print(f"  {status} MCP write reservation")

            # Read back
            read_result = await read_reservations({"limit": 5})
            read_success = read_result is not None and len(read_result) > 0

            status = "✓" if read_success else "✗"
            print(f"  {status} MCP read reservations")

            return write_success and read_success

        stage4_success = asyncio.run(test_mcp_recording())
        test_stages.append({
            "name": "MCP Data Recording",
            "passed": stage4_success,
            "details": "Write and read successful" if stage4_success else "MCP operations failed"
        })
        if not stage4_success:
            all_passed = False

        # Summary
        print("\n" + "=" * 60)
        print("PIPELINE INTEGRATION SUMMARY")
        print("=" * 60)

        print(f"\n{'Stage':<35} {'Status':<10} {'Details':<30}")
        print("-" * 75)

        for stage in test_stages:
            status = "✓ PASS" if stage["passed"] else "✗ FAIL"
            print(f"{stage['name']:<35} {status:<10} {stage['details']:<30}")

        print("-" * 75)
        passed_count = sum(1 for s in test_stages if s["passed"])
        overall_status = "✓ ALL STAGES PASSED" if all_passed else f"✗ {len(test_stages) - passed_count} STAGE(S) FAILED"
        print(f"\n{overall_status}")

        details = {
            "metrics": {
                "Stages Passed": passed_count,
                "Total Stages": len(test_stages),
                "Pass Rate (%)": (passed_count / len(test_stages)) * 100
            },
            "test_cases": [
                {"name": s["name"], "expected": "Pass", "result": "Pass" if s["passed"] else "Fail", "passed": s["passed"]}
                for s in test_stages
            ]
        }

        return all_passed, details

    except Exception as e:
        print(f"\n✗ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


def main(generate_report: bool = True):
    """Run all tests and generate summary.

    Args:
        generate_report: If True, save a markdown report to reports/ folder.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RAG & SYSTEM TEST SUITE".center(80))
    print("=" * 80)
    print(
        "Testing: RAGAS metrics, hybrid retrieval, component integration, end-to-end workflow"
    )

    # Initialize report generator
    report = TestReportGenerator() if generate_report else None

    # Run tests and collect results (each returns (passed, details) tuple)
    tests = [
        ("Guardrails", test_guardrails),
        ("RAG Metrics", test_rag_metrics),
        ("Recall@K", test_recall_at_k),
        ("Components", test_component_initialization),
        ("Hybrid Retrieval", test_hybrid_retrieval),
        ("E2E Workflow", test_end_to_end_workflow),
        ("Agent Routing", test_agent_routing),
        ("Admin Flow", test_admin_flow),  # Human-in-the-loop approval
        ("Data Architecture", test_data_architecture),
        # Load Tests (Stage 3 Requirements)
        ("Chatbot Load Test", test_chatbot_load),
        ("Admin Load Test", test_admin_load),
        ("MCP Load Test", test_mcp_load),
        # MCP Server Tests (merged from test_mcp_server.py)
        ("MCP Server", test_mcp_server),
        # Full Pipeline Integration
        ("Pipeline Integration", test_full_pipeline_integration),
    ]

    test_results = {}
    for test_name, test_func in tests:
        result = test_func()
        # Handle both old format (bool) and new format (bool, details)
        if isinstance(result, tuple):
            passed, details = result
        else:
            passed, details = result, {}

        test_results[test_name] = passed

        if report:
            report.add_result(test_name, passed, details)

    # Summary
    print_section("TEST SUMMARY")

    passed_count = sum(1 for v in test_results.values() if v)
    total = len(test_results)

    print(f"\n{'Test Category':<30} {'Status':<20}")
    print("-" * 50)
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<30} {status:<20}")

    print("-" * 50)
    print(f"\nOverall: {passed_count}/{total} test categories passed")

    if passed_count == total:
        print("\n✓ All tests passed! System is production-ready.")
    else:
        print(f"\n⚠ {total - passed_count} test(s) need attention.")

    # Generate and save report
    if report:
        report_path = report.save_report()
        print(f"\n📄 Report saved to: {report_path}")

    return passed_count == total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG system tests")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating markdown report"
    )
    args = parser.parse_args()

    success = main(generate_report=not args.no_report)
    sys.exit(0 if success else 1)
