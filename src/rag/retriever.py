"""RAG retriever for querying static parking information."""
from typing import List, Dict, Any
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


class ParkingRAGRetriever:
    """RAG retriever for parking information queries."""

    def __init__(self, vector_store, llm=None):
        """Initialize RAG retriever.

        Args:
            vector_store: Vector database instance (Milvus).
            llm: Language model instance. If None, creates LLM based on config.
        """
        self.vector_store = vector_store
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

        logger.info("RAG retriever initialized successfully")

    def _setup_custom_chain(self):
        """Setup custom RAG chain when RetrievalQA is not available."""
        prompt_template = """You are a helpful parking information assistant.
        Use the following context to answer the question. If you don't know the answer, say so.

        Context:
        {context}

        Question: {question}

        Answer:"""

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system for parking information.

        Args:
            question: User query about parking information.

        Returns:
            Dictionary containing answer and source documents.
        """
        try:
            if self._use_retrieval_qa:
                # Use RetrievalQA chain
                result = self.rag_chain({"query": question})
                logger.info(f"Query processed: {question[:50]}...")
                return {
                    "answer": result.get("result", ""),
                    "sources": result.get("source_documents", []),
                }
            else:
                # Use custom chain
                documents = self.retriever.invoke(question)
                context = "\n".join([doc.page_content for doc in documents])
                answer = self.chain.invoke({
                    "context": context,
                    "question": question
                })
                logger.info(f"Query processed: {question[:50]}...")
                return {
                    "answer": answer,
                    "sources": documents,
                }
        except Exception as e:
            logger.error(f"Error during RAG query: {e}")
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
