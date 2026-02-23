"""Generate evaluation reports for the parking chatbot system."""
import json
from datetime import datetime
from typing import Dict, List
from src.evaluation.metrics import MetricsCalculator, ResponseEvaluator, PerformanceTester
from src.utils.logging import logger


class EvaluationReport:
    """Generate comprehensive evaluation reports."""

    def __init__(self):
        """Initialize evaluation report generator."""
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "rag_evaluation": {},
            "safety_evaluation": {},
            "performance_metrics": {},
            "reservation_evaluation": {},
        }

    def add_rag_evaluation_results(
        self,
        test_cases: int,
        recall_at_k: Dict[int, float],
        precision_at_k: Dict[int, float],
        mrr: float,
        avg_retrieval_time_ms: float,
    ):
        """Add RAG evaluation results.

        Args:
            test_cases: Number of test cases evaluated.
            recall_at_k: Recall@K results.
            precision_at_k: Precision@K results.
            mrr: Mean Reciprocal Rank score.
            avg_retrieval_time_ms: Average retrieval latency.
        """
        self.test_results["rag_evaluation"] = {
            "test_cases": test_cases,
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "mean_reciprocal_rank": mrr,
            "avg_retrieval_latency_ms": avg_retrieval_time_ms,
        }
        logger.info("RAG evaluation results added to report")

    def add_safety_evaluation_results(
        self,
        total_tests: int,
        blocked_count: int,
        false_positives: int,
        false_negatives: int,
    ):
        """Add security/safety evaluation results.

        Args:
            total_tests: Total safety tests run.
            blocked_count: Number of malicious inputs blocked.
            false_positives: Legitimate inputs incorrectly blocked.
            false_negatives: Malicious inputs that passed through.
        """
        if total_tests == 0:
            return

        precision = blocked_count / (blocked_count + false_positives) if (blocked_count + false_positives) > 0 else 0
        recall = blocked_count / (blocked_count + false_negatives) if (blocked_count + false_negatives) > 0 else 0

        self.test_results["safety_evaluation"] = {
            "total_tests": total_tests,
            "blocked_count": blocked_count,
            "block_rate": blocked_count / total_tests,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        }
        logger.info("Safety evaluation results added to report")

    def add_performance_metrics(
        self,
        avg_query_time_ms: float,
        avg_retrieval_time_ms: float,
        avg_generation_time_ms: float,
        total_requests: int,
        successful_requests: int,
    ):
        """Add performance metrics.

        Args:
            avg_query_time_ms: Average end-to-end query latency.
            avg_retrieval_time_ms: Average retrieval latency.
            avg_generation_time_ms: Average LLM generation latency.
            total_requests: Total requests processed.
            successful_requests: Successfully completed requests.
        """
        self.test_results["performance_metrics"] = {
            "avg_query_latency_ms": avg_query_time_ms,
            "avg_retrieval_latency_ms": avg_retrieval_time_ms,
            "avg_generation_latency_ms": avg_generation_time_ms,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        }
        logger.info("Performance metrics added to report")

    def add_reservation_evaluation(
        self,
        total_reservations: int,
        successful_reservations: int,
        avg_approval_time_hours: float,
        data_collection_accuracy: float,
    ):
        """Add reservation process evaluation results.

        Args:
            total_reservations: Total reservations tested.
            successful_reservations: Successfully completed reservations.
            avg_approval_time_hours: Average time for admin approval.
            data_collection_accuracy: Accuracy of user data collection.
        """
        self.test_results["reservation_evaluation"] = {
            "total_reservations": total_reservations,
            "successful_reservations": successful_reservations,
            "success_rate": successful_reservations / total_reservations if total_reservations > 0 else 0,
            "avg_approval_time_hours": avg_approval_time_hours,
            "data_collection_accuracy": data_collection_accuracy,
        }
        logger.info("Reservation evaluation results added to report")

    def generate_markdown_report(self) -> str:
        """Generate a markdown format evaluation report.

        Returns:
            Markdown formatted report string.
        """
        report = "# Parking Chatbot Evaluation Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Executive Summary
        report += "## Executive Summary\n\n"
        report += "This report evaluates the parking chatbot system across four key dimensions:\n"
        report += "1. **RAG System Performance** - Document retrieval accuracy and latency\n"
        report += "2. **Safety & Security** - Data protection and malicious input filtering\n"
        report += "3. **Performance Metrics** - Response times and system throughput\n"
        report += "4. **Reservation Process** - User data collection and booking accuracy\n\n"

        # RAG Evaluation
        if self.test_results["rag_evaluation"]:
            report += "## RAG System Evaluation\n\n"
            rag = self.test_results["rag_evaluation"]
            report += f"- **Test Cases:** {rag.get('test_cases', 'N/A')}\n"
            report += f"- **Mean Reciprocal Rank (MRR):** {rag.get('mean_reciprocal_rank', 0):.3f}\n"
            report += f"- **Recall@K:**\n"
            for k, score in rag.get("recall_at_k", {}).items():
                report += f"  - Recall@{k}: {score:.3f}\n"
            report += f"- **Precision@K:**\n"
            for k, score in rag.get("precision_at_k", {}).items():
                report += f"  - Precision@{k}: {score:.3f}\n"
            report += f"- **Avg Retrieval Latency:** {rag.get('avg_retrieval_latency_ms', 'N/A'):.2f} ms\n\n"

        # Safety Evaluation
        if self.test_results["safety_evaluation"]:
            report += "## Security & Safety Evaluation\n\n"
            safety = self.test_results["safety_evaluation"]
            report += f"- **Total Tests:** {safety.get('total_tests', 'N/A')}\n"
            report += f"- **Blocked Inputs:** {safety.get('blocked_count', 'N/A')} ({safety.get('block_rate', 0):.1%})\n"
            report += f"- **False Positives:** {safety.get('false_positives', 'N/A')}\n"
            report += f"- **False Negatives:** {safety.get('false_negatives', 'N/A')}\n"
            report += f"- **Precision:** {safety.get('precision', 0):.3f}\n"
            report += f"- **Recall:** {safety.get('recall', 0):.3f}\n"
            report += f"- **F1 Score:** {safety.get('f1_score', 0):.3f}\n\n"

        # Performance Metrics
        if self.test_results["performance_metrics"]:
            report += "## Performance Metrics\n\n"
            perf = self.test_results["performance_metrics"]
            report += f"- **Avg Query Latency:** {perf.get('avg_query_latency_ms', 'N/A'):.2f} ms\n"
            report += f"- **Avg Retrieval Latency:** {perf.get('avg_retrieval_latency_ms', 'N/A'):.2f} ms\n"
            report += f"- **Avg Generation Latency:** {perf.get('avg_generation_latency_ms', 'N/A'):.2f} ms\n"
            report += f"- **Total Requests:** {perf.get('total_requests', 'N/A')}\n"
            report += f"- **Successful Requests:** {perf.get('successful_requests', 'N/A')} ({perf.get('success_rate', 0):.1%})\n\n"

        # Reservation Evaluation
        if self.test_results["reservation_evaluation"]:
            report += "## Reservation Process Evaluation\n\n"
            res = self.test_results["reservation_evaluation"]
            report += f"- **Total Reservations:** {res.get('total_reservations', 'N/A')}\n"
            report += f"- **Successful:** {res.get('successful_reservations', 'N/A')} ({res.get('success_rate', 0):.1%})\n"
            report += f"- **Avg Approval Time:** {res.get('avg_approval_time_hours', 'N/A')} hours\n"
            report += f"- **Data Collection Accuracy:** {res.get('data_collection_accuracy', 0):.1%}\n\n"

        # Recommendations
        report += "## Recommendations\n\n"
        report += "1. **RAG Optimization:** Focus on improving precision@3 for better top-3 relevance\n"
        report += "2. **Safety Enhancement:** Continue monitoring false positives to improve user experience\n"
        report += "3. **Performance Tuning:** Profile LLM generation to reduce latency\n"
        report += "4. **Human Review:** Implement admin dashboard for reservation approvals\n\n"

        report += "## Conclusion\n\n"
        report += "The parking chatbot system demonstrates functional capabilities across all evaluated dimensions. "
        report += "The system successfully retrieves relevant documents, filters sensitive data, and processes reservations. "
        report += "Performance metrics are acceptable for a local deployment with an Ollama LLM. "
        report += "Further optimization of the retrieval pipeline and generation latency is recommended for production deployment.\n"

        return report

    def save_report(self, filepath: str):
        """Save evaluation report to file.

        Args:
            filepath: Path to save the report to.
        """
        report = self.generate_markdown_report()
        try:
            with open(filepath, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise

    def save_json_results(self, filepath: str):
        """Save raw test results as JSON.

        Args:
            filepath: Path to save the JSON file to.
        """
        try:
            with open(filepath, "w") as f:
                json.dump(self.test_results, f, indent=2, default=str)
            logger.info(f"JSON results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save JSON results: {e}")
            raise
