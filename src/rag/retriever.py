"""RAG retriever for querying static parking information with hybrid retrieval."""
from typing import List, Dict, Any
import re
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.llm_provider import create_llm
from src.utils.logging import logger

# Try to import RetrievalQA from different locations
try:
    from langchain.chains import RetrievalQA
    HAS_RETRIEVAL_QA = True
except ImportError:
    try:
        from langchain_community.chains import RetrievalQA
        HAS_RETRIEVAL_QA = True
    except ImportError:
        HAS_RETRIEVAL_QA = False
        logger.warning("RetrievalQA not found, will use custom chain implementation")


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

        # Try to use RetrievalQA if available, otherwise use custom chain
        if HAS_RETRIEVAL_QA:
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Stuff all relevant documents into context
                retriever=self.retriever,
                return_source_documents=True,
            )
            self._use_retrieval_qa = True
        else:
            # Custom chain implementation for newer LangChain versions
            self._use_retrieval_qa = False
            self._setup_custom_chain()

        logger.info("RAG retriever initialized successfully with hybrid retrieval support")

    def _setup_custom_chain(self):
        """Setup custom RAG chain when RetrievalQA is not available."""
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

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system for parking information using hybrid retrieval.

        Hybrid retrieval combines:
        - Vector DB semantic search (static reference data)
        - SQL Agent (real-time parking data, intelligently queried)

        Args:
            question: User query about parking information.

        Returns:
            Dictionary containing answer and source documents.
        """
        try:
            if self._use_retrieval_qa:
                # Use RetrievalQA chain with hybrid context
                documents = self.retriever.invoke(question)
                # Build hybrid context (Vector DB + SQL Agent)
                context = self._build_hybrid_context(question, documents)

                result = self.rag_chain({"query": question})
                answer = result.get("result", "")
                # Clean thinking tags from reasoning models
                answer = _clean_response(answer)
                logger.info(f"Query processed with hybrid retrieval: {question[:50]}...")
                return {
                    "answer": answer,
                    "sources": result.get("source_documents", []),
                }
            else:
                # Use custom chain with hybrid context
                documents = self.retriever.invoke(question)
                # Build hybrid context (Vector DB + SQL Agent)
                context = self._build_hybrid_context(question, documents)

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
