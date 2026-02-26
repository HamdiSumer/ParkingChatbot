"""RAG retriever for querying static parking information with hybrid retrieval."""
from typing import List, Dict, Any
import re
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.llm_provider import create_llm
from src.utils.logging import logger


def _clean_response(response: str) -> str:
    """Remove thinking tags from reasoning model outputs (e.g., deepseek-r1).

    Args:
        response: Raw response from LLM that may contain <think>...</think> tags.

    Returns:
        Cleaned response with thinking tags removed.
    """
    # Remove <think>...</think> tags and everything inside
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    # Clean up extra whitespace
    cleaned = '\n'.join(line.strip() for line in cleaned.split('\n') if line.strip())
    return cleaned.strip()


class ParkingRAGRetriever:
    """RAG retriever for parking information queries with hybrid retrieval.

    Combines:
    - Vector DB semantic search for static reference data
    - SQL Agent for dynamic data queries (inspects schema, decides what to query)
    """

    def __init__(self, vector_store, llm=None, db=None, sql_agent=None):
        """Initialize RAG retriever.

        Args:
            vector_store: Vector database instance (Weaviate/Milvus).
            llm: Language model instance. If None, creates LLM based on config.
            db: SQL database instance for hybrid retrieval (optional).
            sql_agent: Pre-built SQL agent. If None and db provided, creates one.
        """
        self.vector_store = vector_store
        self.db = db
        self.sql_agent = sql_agent
        self.llm = llm or create_llm(temperature=0.3)  # Lower temp for accurate info
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
        )

        # Use custom chain for hybrid retrieval (always preferred when SQL agent is available)
        # RetrievalQA doesn't support injecting hybrid context, so we use custom chain
        self._use_retrieval_qa = False
        self._setup_custom_chain()

        logger.info("RAG retriever initialized with hybrid retrieval support")

    def _setup_custom_chain(self):
        """Setup custom RAG chain with re-ranking support."""
        prompt_template = """You are a helpful parking information assistant.
        Use the following context to answer the question. If you don't know the answer, say so.

        If real-time SQL data is provided, use it to give current information.

        Context:
        {context}

        Question: {question}

        Answer:"""

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

        # Setup re-ranking prompt
        self.rerank_prompt = PromptTemplate(
            template="""Rate the relevance of the following document to the question on a scale of 0-10.
Only output a single number (0-10), nothing else.

Question: {question}

Document: {document}

Relevance score (0-10):""",
            input_variables=["question", "document"]
        )
        self.rerank_chain = self.rerank_prompt | self.llm | StrOutputParser()

    def _rerank_documents(self, question: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """Re-rank documents using LLM-based relevance scoring.

        Args:
            question: User's question.
            documents: List of retrieved documents.
            top_k: Number of top documents to return after re-ranking.

        Returns:
            Re-ranked list of documents (most relevant first).
        """
        if not documents or len(documents) <= 1:
            return documents

        try:
            scored_docs = []
            for doc in documents:
                try:
                    score_str = self.rerank_chain.invoke({
                        "question": question,
                        "document": doc.page_content[:500]  # Limit content length
                    }).strip()

                    # Extract numeric score
                    score = float(''.join(c for c in score_str if c.isdigit() or c == '.') or '5')
                    score = max(0, min(10, score))  # Clamp to 0-10
                    scored_docs.append((score, doc))
                    logger.debug(f"Re-rank score {score}: {doc.page_content[:50]}...")
                except Exception as e:
                    logger.warning(f"Re-ranking failed for document: {e}")
                    scored_docs.append((5.0, doc))  # Default score

            # Sort by score (descending) and return top_k
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            reranked = [doc for _, doc in scored_docs[:top_k]]

            logger.info(f"Re-ranked {len(documents)} documents, returning top {len(reranked)}")
            return reranked

        except Exception as e:
            logger.warning(f"Re-ranking failed, returning original order: {e}")
            return documents[:top_k]

    def _build_hybrid_context(self, question: str, vector_docs: List[Document]) -> str:
        """Build hybrid context combining vector DB and SQL agent results.

        Args:
            question: User's question.
            vector_docs: Documents retrieved from vector DB.

        Returns:
            Combined context string for the LLM.
        """
        # Start with vector DB context (static reference data)
        vector_context = "\n".join([doc.page_content for doc in vector_docs])
        context = "=== STATIC REFERENCE DATA (from vector DB) ===\n" + vector_context

        # Add SQL agent results if available
        if self.sql_agent:
            try:
                sql_results = self._query_sql_agent(question)
                if sql_results and sql_results.strip():
                    context += "\n\n=== REAL-TIME DATA (from SQL database) ===\n" + sql_results
                    logger.info("Added SQL agent results to context")
            except Exception as e:
                logger.warning(f"Failed to get SQL agent results: {e}")

        return context

    def _query_sql_agent(self, question: str) -> str:
        """Query the SQL agent to get relevant database information.

        The SQL agent inspects the database schema and decides what queries
        would help answer the user's question.

        Args:
            question: User's question.

        Returns:
            SQL query results as a formatted string.
        """
        try:
            # Invoke the SQL agent with the user's question
            result = self.sql_agent.invoke({"input": question})

            # Extract the result text
            if isinstance(result, dict):
                sql_result = result.get("output", "")
            else:
                sql_result = str(result)

            return sql_result if sql_result else ""

        except Exception as e:
            logger.error(f"SQL agent error: {e}")
            return ""

    def query(self, question: str, use_reranking: bool = True) -> Dict[str, Any]:
        """Query the RAG system for parking information using hybrid retrieval.

        Hybrid retrieval combines:
        - Vector DB semantic search (static reference data)
        - LLM-based re-ranking for improved relevance
        - SQL Agent (real-time parking data, intelligently queried)

        Args:
            question: User query about parking information.
            use_reranking: Whether to re-rank documents (default True).

        Returns:
            Dictionary containing answer and source documents.
        """
        try:
            # Retrieve initial candidates from vector DB (get more for re-ranking)
            k_initial = 6 if use_reranking else 3
            documents = self.vector_store.as_retriever(
                search_kwargs={"k": k_initial}
            ).invoke(question)

            # Re-rank documents for better relevance
            if use_reranking and len(documents) > 1:
                documents = self._rerank_documents(question, documents, top_k=3)

            # Build hybrid context (Vector DB + SQL Agent)
            context = self._build_hybrid_context(question, documents)

            # Generate answer using LLM with hybrid context
            answer = self.chain.invoke({
                "context": context,
                "question": question
            })

            # Clean thinking tags from reasoning models
            answer = _clean_response(answer)
            logger.info(f"Query processed with hybrid retrieval: {question[:50]}...")

            return {
                "answer": answer,
                "sources": documents,
            }
        except Exception as e:
            logger.error(f"Error during hybrid RAG query: {e}")
            return {"answer": "I couldn't process your query. Please try again.", "sources": []}

    def retrieve_documents(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve documents similar to the query.

        Args:
            query: Search query.
            k: Number of documents to retrieve.

        Returns:
            List of relevant documents.
        """
        try:
            return self.retriever.invoke(query)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def vector_search_only(self, query: str, k: int = 3, use_reranking: bool = True) -> Dict[str, Any]:
        """Perform vector search without SQL agent (used by agent tools).

        Args:
            query: Search query.
            k: Number of documents to retrieve.
            use_reranking: Whether to re-rank documents.

        Returns:
            Dict with 'documents' and 'formatted' context.
        """
        try:
            k_initial = k * 2 if use_reranking else k
            documents = self.vector_store.as_retriever(
                search_kwargs={"k": k_initial}
            ).invoke(query)

            if use_reranking and len(documents) > 1:
                documents = self._rerank_documents(query, documents, top_k=k)

            formatted = "\n\n".join([doc.page_content for doc in documents])
            logger.info(f"Vector search returned {len(documents)} documents")

            return {
                "documents": documents,
                "formatted": formatted,
                "success": True
            }
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {"documents": [], "formatted": "", "success": False, "error": str(e)}

    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the provided context.

        Args:
            question: User's question.
            context: Pre-built context from tools.

        Returns:
            Generated answer string.
        """
        try:
            answer = self.chain.invoke({
                "context": context,
                "question": question
            })
            return _clean_response(answer)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I couldn't generate an answer. Please try again."
