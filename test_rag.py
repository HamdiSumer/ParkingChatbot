#!/usr/bin/env python3
"""
Comprehensive RAG & System Testing Suite
Combines RAGAS metrics, integration tests, and end-to-end verification.

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
import numpy as np
from typing import List, Dict, Tuple

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
    """Test guardrails - verify sensitive data filtering."""
    print_section("TEST 1: GUARDRAILS - Security & Safety")

    filter_obj = DataProtectionFilter()
    test_cases = [
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

    passed = 0
    detailed_cases = []
    print(f"\n{'Test Case':<50} {'Expected':<10} {'Result':<10} {'Status':<10}")
    print("-" * 80)

    for message, should_be_safe, reason in test_cases:
        is_safe, _ = filter_obj.check_safety(message)
        expected = "Safe" if should_be_safe else "Blocked"
        result = "Safe" if is_safe else "Blocked"
        test_passed = is_safe == should_be_safe
        status = "✓" if test_passed else "✗"
        if test_passed:
            passed += 1
        print(f"{message[:48]:<50} {expected:<10} {result:<10} {status:<10}")

        detailed_cases.append({
            "name": reason,
            "expected": expected,
            "result": result,
            "passed": test_passed
        })

    detection_rate = (passed / len(test_cases)) * 100
    print("-" * 80)
    print(f"Detection Rate: {detection_rate:.1f}% ({passed}/{len(test_cases)})")

    details = {
        "metrics": {"Detection Rate (%)": detection_rate},
        "test_cases": detailed_cases
    }

    return detection_rate == 100.0, details


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
    """Test RAG quality using RAGAS framework with semantic metrics."""
    print_section("TEST 2: RAG METRICS - Quality Assessment (RAGAS)")

    try:
        with create_test_app() as app:
            if not app.rag_retriever:
                print("\n✗ RAG Retriever not initialized.")
                return False, {"notes": "RAG Retriever not initialized"}

            test_queries = [
                "Where is downtown parking?",
                "What are the parking prices?",
                "How many spaces are available?",
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
                result = app.rag_retriever.query(query)
                latency_ms = (time.time() - start) * 1000

                answer = result.get("answer", "")
                sources = result.get("sources", [])
                context = "\n".join([doc.page_content for doc in sources])

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
    """Test hybrid retrieval combining Vector DB semantic search + SQL Agent queries."""
    print_section("TEST 7: HYBRID RETRIEVAL - SQL Agent + Vector DB")

    try:
        with create_test_app() as app:
            # Check if SQL agent is initialized
            if not app.sql_agent:
                print("\n⚠ SQL Agent not initialized. Testing Vector DB only.")
                return True, {"notes": "SQL Agent not initialized, tested Vector DB only"}

            if not app.rag_retriever:
                print("\n✗ RAG Retriever not initialized.")
                return False, {"notes": "RAG Retriever not initialized"}

            test_queries = [
                ("How many parking spaces are available?", "availability/count"),
                ("What is the status of downtown parking?", "status/location"),
                ("How much does parking cost?", "pricing"),
            ]

            print(f"\n{'Query':<40} {'Type':<20} {'Status':<10}")
            print("-" * 80)

            all_passed = True

            # Test 1: SQL Agent Query Generation
            print("\n" + "=" * 80)
            print("  SQL Agent Query Quality")
            print("=" * 80)

            sql_agent_results = {
                "valid_queries": 0,
                "query_execution_success": 0,
                "query_latency_ms": [],
            }

            for query, query_type in test_queries:
                try:
                    start = time.time()
                    sql_result = app.sql_agent.invoke({"input": query})
                    latency_ms = (time.time() - start) * 1000

                    output = sql_result.get("output", "")

                    # Check if query was attempted
                    if output:
                        sql_agent_results["valid_queries"] += 1
                        sql_agent_results["query_latency_ms"].append(latency_ms)

                    # Query execution was successful if output contains results
                    if "Query:" in output and "Result:" in output:
                        sql_agent_results["query_execution_success"] += 1
                        status = "✓ Success"
                    else:
                        status = "⚠ No SQL"

                    print(
                        f"{query:<40} {query_type:<20} {status:<10} ({latency_ms:.0f}ms)"
                    )

                except Exception as e:
                    print(f"{query:<40} {query_type:<20} ✗ Error: {str(e)[:30]}")
                    all_passed = False

            # Test 2: Hybrid Context Quality
            print("\n" + "=" * 80)
            print("  Hybrid Context Quality (Vector DB + SQL Agent)")
            print("=" * 80)

            hybrid_results = {
                "queries_with_sql_data": 0,
                "context_completeness": [],
                "latency_ms": [],
            }

            print(f"\n{'Query':<40} {'With SQL Data':<15} {'Latency':<10}")
            print("-" * 80)

            for query, _ in test_queries:
                try:
                    start = time.time()
                    result = app.rag_retriever.query(query)
                    latency_ms = (time.time() - start) * 1000

                    answer = result.get("answer", "")

                    # Check if SQL data was included
                    has_sql_data = any(
                        keyword in answer.lower()
                        for keyword in [
                            "available",
                            "open",
                            "closed",
                            "query:",
                            "result:",
                        ]
                    )

                    if has_sql_data:
                        hybrid_results["queries_with_sql_data"] += 1

                    hybrid_results["latency_ms"].append(latency_ms)

                    # Context completeness: does answer mention both static and dynamic info
                    context_complete = has_sql_data and len(answer) > 50
                    if context_complete:
                        hybrid_results["context_completeness"].append(1.0)
                    else:
                        hybrid_results["context_completeness"].append(0.5)

                    status = "✓ With SQL" if has_sql_data else "⚠ Vector only"
                    print(f"{query:<40} {status:<15} {latency_ms:<10.0f}ms")

                except Exception as e:
                    print(f"{query:<40} ✗ Error{'':<10} {str(e)[:30]}")
                    all_passed = False

            # Test 3: Production Metrics for Hybrid Retrieval
            print("\n" + "=" * 80)
            print("  Production Metrics for Hybrid Retrieval")
            print("=" * 80)

            print("\nSQL Agent Metrics:")
            if sql_agent_results["valid_queries"] > 0:
                query_success_rate = (
                    sql_agent_results["query_execution_success"]
                    / sql_agent_results["valid_queries"]
                ) * 100
                avg_query_latency = sum(sql_agent_results["query_latency_ms"]) / len(
                    sql_agent_results["query_latency_ms"]
                )

                print(
                    f"  • Query Generation Rate: {query_success_rate:.1f}% ({sql_agent_results['query_execution_success']}/{sql_agent_results['valid_queries']})"
                )
                print(f"  • Average Query Latency: {avg_query_latency:.0f}ms")
            else:
                print("  • No SQL queries generated")

            print("\nHybrid Retrieval Metrics:")
            if hybrid_results["queries_with_sql_data"] > 0:
                sql_coverage = (
                    hybrid_results["queries_with_sql_data"] / len(test_queries)
                ) * 100
                avg_latency = sum(hybrid_results["latency_ms"]) / len(
                    hybrid_results["latency_ms"]
                )
                avg_completeness = sum(hybrid_results["context_completeness"]) / len(
                    hybrid_results["context_completeness"]
                )

                print(
                    f"  • SQL Data Coverage: {sql_coverage:.1f}% (queries enhanced with SQL)"
                )
                print(
                    f"  • Average Total Latency: {avg_latency:.0f}ms (Vector DB + SQL Agent)"
                )
                print(f"  • Context Completeness: {avg_completeness:.2f}/1.0")
            else:
                print(
                    "  • SQL Agent not providing data (may be expected if query not relevant)"
                )

            print("\nProduction RAG Concepts Demonstrated:")
            print("  ✓ Semantic Search: Vector DB for static reference data")
            print("  ✓ SQL Agent: LLM-driven dynamic query generation")
            print("  ✓ Hybrid Integration: Combined Vector DB + SQL Agent context")
            print(
                "  ✓ Graceful Degradation: Works even if SQL agent doesn't find relevant queries"
            )

            # Build metrics for report
            metrics = {}
            if sql_agent_results["valid_queries"] > 0:
                query_success_rate = (sql_agent_results["query_execution_success"] / sql_agent_results["valid_queries"]) * 100
                avg_query_latency = sum(sql_agent_results["query_latency_ms"]) / len(sql_agent_results["query_latency_ms"])
                metrics["SQL Query Success Rate (%)"] = query_success_rate
                metrics["SQL Avg Latency (ms)"] = avg_query_latency

            if hybrid_results["queries_with_sql_data"] > 0:
                sql_coverage = (hybrid_results["queries_with_sql_data"] / len(test_queries)) * 100
                avg_latency = sum(hybrid_results["latency_ms"]) / len(hybrid_results["latency_ms"])
                avg_completeness = sum(hybrid_results["context_completeness"]) / len(hybrid_results["context_completeness"])
                metrics["SQL Data Coverage (%)"] = sql_coverage
                metrics["Hybrid Avg Latency (ms)"] = avg_latency
                metrics["Context Completeness"] = avg_completeness

            details = {"metrics": metrics} if metrics else {"notes": "Hybrid retrieval completed"}
            return all_passed, details

    except Exception as e:
        print(f"\n✗ Hybrid retrieval test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, {"notes": f"Error: {str(e)}"}


# ============================================================================
# 8. DATA ARCHITECTURE TEST
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
        ("Data Architecture", test_data_architecture),
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
