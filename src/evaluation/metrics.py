"""Performance and accuracy metrics for RAG system evaluation."""
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from src.utils.logging import logger


@dataclass
class RetrievalMetrics:
    """Metrics for document retrieval evaluation."""

    recall_at_k: Dict[int, float]  # Recall@1, @3, @5, etc.
    precision_at_k: Dict[int, float]  # Precision@1, @3, @5, etc.
    mean_reciprocal_rank: float  # MRR
    ndcg_at_k: Dict[int, float]  # NDCG@K
    avg_retrieval_time: float  # Average retrieval latency in ms


@dataclass
class RAGMetrics:
    """Metrics for RAG system evaluation."""

    avg_response_latency: float  # ms
    avg_retrieval_latency: float  # ms
    avg_generation_latency: float  # ms
    avg_response_length: float  # average number of tokens
    answer_relevance: float  # 0-1, measured against reference answers
    retrieval_metrics: RetrievalMetrics


@dataclass
class SystemMetrics:
    """Overall system metrics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    safety_blocks: int
    avg_response_time: float
    avg_accuracy: float
    reservation_completion_rate: float


class MetricsCalculator:
    """Calculate evaluation metrics for RAG system."""

    @staticmethod
    def calculate_recall_at_k(
        retrieved_docs: List[str], relevant_docs: List[str], k: int
    ) -> float:
        """Calculate Recall@K metric.

        Recall@K = (# of relevant docs in top K) / (total # of relevant docs)

        Args:
            retrieved_docs: List of retrieved document IDs (in order).
            relevant_docs: List of relevant document IDs.
            k: Position to measure at.

        Returns:
            Recall@K value between 0 and 1.
        """
        if not relevant_docs:
            return 0.0

        retrieved_at_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        relevant_in_top_k = len(retrieved_at_k.intersection(relevant_set))

        recall = relevant_in_top_k / len(relevant_set)
        return min(recall, 1.0)

    @staticmethod
    def calculate_precision_at_k(
        retrieved_docs: List[str], relevant_docs: List[str], k: int
    ) -> float:
        """Calculate Precision@K metric.

        Precision@K = (# of relevant docs in top K) / K

        Args:
            retrieved_docs: List of retrieved document IDs (in order).
            relevant_docs: List of relevant document IDs.
            k: Position to measure at.

        Returns:
            Precision@K value between 0 and 1.
        """
        if k == 0:
            return 0.0

        retrieved_at_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        relevant_in_top_k = len(retrieved_at_k.intersection(relevant_set))

        precision = relevant_in_top_k / k
        return precision

    @staticmethod
    def calculate_mrr(
        retrieved_docs: List[str], relevant_docs: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / (position of first relevant document)

        Args:
            retrieved_docs: List of retrieved document IDs (in order).
            relevant_docs: List of relevant document IDs.

        Returns:
            MRR value.
        """
        relevant_set = set(relevant_docs)

        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def calculate_ndcg_at_k(
        retrieved_docs: List[str], relevant_docs: List[str], k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K (NDCG@K).

        Args:
            retrieved_docs: List of retrieved document IDs (in order).
            relevant_docs: List of relevant document IDs.
            k: Position to measure at.

        Returns:
            NDCG@K value between 0 and 1.
        """
        relevant_set = set(relevant_docs)

        # Calculate DCG@K
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k]):
            relevance = 1.0 if doc in relevant_set else 0.0
            dcg += relevance / (i + 1)  # log2(i + 2) for log discounting

        # Calculate IDCG@K (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_set))):
            idcg += 1.0 / (i + 1)

        if idcg == 0:
            return 0.0

        ndcg = dcg / idcg
        return ndcg


class ResponseEvaluator:
    """Evaluate generated responses for quality and accuracy."""

    @staticmethod
    def calculate_answer_relevance(
        generated_answer: str,
        reference_answers: List[str],
        threshold: float = 0.5,
    ) -> float:
        """Calculate relevance of generated answer to reference answers.

        This is a simple overlap-based metric. Production systems would use
        semantic similarity with embeddings.

        Args:
            generated_answer: Generated answer from the model.
            reference_answers: List of reference/expected answers.
            threshold: Similarity threshold.

        Returns:
            Relevance score between 0 and 1.
        """
        if not reference_answers:
            return 0.0

        generated_tokens = set(generated_answer.lower().split())
        best_overlap = 0.0

        for reference in reference_answers:
            reference_tokens = set(reference.lower().split())

            if not reference_tokens:
                continue

            overlap = len(generated_tokens.intersection(reference_tokens))
            overlap_ratio = overlap / len(reference_tokens)

            best_overlap = max(best_overlap, overlap_ratio)

        return min(best_overlap, 1.0)

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Estimate number of tokens in text.

        Simple estimation: ~4 characters per token.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        return len(text) // 4


class PerformanceTester:
    """Test performance characteristics of the chatbot."""

    def __init__(self):
        """Initialize performance tester."""
        self.query_times: List[float] = []
        self.retrieval_times: List[float] = []
        self.generation_times: List[float] = []

    def measure_retrieval_time(self, retriever_func, query: str) -> Tuple[any, float]:
        """Measure time taken for document retrieval.

        Args:
            retriever_func: Callable that performs retrieval.
            query: Query to retrieve documents for.

        Returns:
            Tuple of (result, time_in_ms).
        """
        start = time.time()
        result = retriever_func(query)
        elapsed_ms = (time.time() - start) * 1000

        self.retrieval_times.append(elapsed_ms)
        logger.info(f"Retrieval completed in {elapsed_ms:.2f}ms")

        return result, elapsed_ms

    def measure_generation_time(self, generator_func, context: str) -> Tuple[str, float]:
        """Measure time taken for response generation.

        Args:
            generator_func: Callable that generates response.
            context: Context/prompt to generate response for.

        Returns:
            Tuple of (response, time_in_ms).
        """
        start = time.time()
        response = generator_func(context)
        elapsed_ms = (time.time() - start) * 1000

        self.generation_times.append(elapsed_ms)
        logger.info(f"Generation completed in {elapsed_ms:.2f}ms")

        return response, elapsed_ms

    def measure_end_to_end_time(self, workflow_func, query: str) -> Tuple[str, float]:
        """Measure end-to-end response time.

        Args:
            workflow_func: Callable that runs entire workflow.
            query: User query.

        Returns:
            Tuple of (response, time_in_ms).
        """
        start = time.time()
        response = workflow_func(query)
        elapsed_ms = (time.time() - start) * 1000

        self.query_times.append(elapsed_ms)
        logger.info(f"End-to-end query completed in {elapsed_ms:.2f}ms")

        return response, elapsed_ms

    def get_statistics(self) -> Dict[str, float]:
        """Get aggregated performance statistics.

        Returns:
            Dictionary with average, min, max for each metric.
        """
        def get_stats(times: List[float]) -> Dict[str, float]:
            if not times:
                return {"avg": 0, "min": 0, "max": 0}
            return {
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }

        return {
            "query_times": get_stats(self.query_times),
            "retrieval_times": get_stats(self.retrieval_times),
            "generation_times": get_stats(self.generation_times),
        }
