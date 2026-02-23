"""Pinecone vector database integration for static parking data."""
from typing import List
from langchain_community.vectorstores import Pinecone
from langchain_core.documents import Document
from src.config import get_config
from src.rag.embeddings import create_embeddings
from src.utils.logging import logger

config = get_config()


def create_pinecone_connection(embeddings=None):
    """Create a connection to Pinecone vector database.

    Args:
        embeddings: Embeddings instance. If None, creates new one.

    Returns:
        Pinecone vector store instance.
    """
    if not config.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in .env")

    if embeddings is None:
        embeddings = create_embeddings()

    try:
        import pinecone

        # Initialize Pinecone
        pinecone.init(
            api_key=config.PINECONE_API_KEY,
            environment=config.PINECONE_ENVIRONMENT,
        )

        vector_store = Pinecone.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
        )

        logger.info(f"✓ Connected to Pinecone: {config.PINECONE_INDEX_NAME}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Pinecone: {e}")
        raise


def ingest_static_data_pinecone(vector_store, documents: List[Document]):
    """Ingest static parking data into Pinecone.

    Args:
        vector_store: Pinecone instance.
        documents: List of LangChain Document objects.

    Returns:
        Number of documents ingested.
    """
    try:
        # Upsert documents to Pinecone
        vector_store.add_documents(documents)
        logger.info(f"✓ Ingested {len(documents)} documents into Pinecone")
        return len(documents)
    except Exception as e:
        logger.error(f"Failed to ingest documents to Pinecone: {e}")
        raise


def clear_pinecone_index(index_name: str):
    """Clear all vectors from a Pinecone index.

    Args:
        index_name: Name of the index to clear.
    """
    try:
        import pinecone

        pinecone.init(
            api_key=config.PINECONE_API_KEY,
            environment=config.PINECONE_ENVIRONMENT,
        )

        index = pinecone.Index(index_name)
        index.delete(delete_all=True)
        logger.info(f"✓ Cleared Pinecone index: {index_name}")
    except Exception as e:
        logger.error(f"Failed to clear Pinecone index: {e}")
        raise
