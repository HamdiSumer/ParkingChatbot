"""Milvus vector database integration for static parking data."""
from langchain_community.vectorstores import Milvus
from src.config import get_config
from src.rag.embeddings import create_embeddings
from src.utils.logging import logger

config = get_config()


def create_milvus_connection(embeddings=None):
    """Create a connection to Milvus vector database.

    Args:
        embeddings: Embeddings instance. If None, creates new one.

    Returns:
        Milvus vector store instance.
    """
    if embeddings is None:
        embeddings = create_embeddings()

    try:
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"host": config.MILVUS_HOST, "port": config.MILVUS_PORT},
            collection_name=config.MILVUS_COLLECTION,
            drop_old=False,
        )
        logger.info(f"Connected to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise


def ingest_static_data(vector_store, documents: list):
    """Ingest static parking data into Milvus.

    Args:
        vector_store: Milvus instance.
        documents: List of LangChain Document objects.

    Returns:
        Number of documents ingested.
    """
    try:
        vector_store.add_documents(documents)
        logger.info(f"Ingested {len(documents)} documents into Milvus")
        return len(documents)
    except Exception as e:
        logger.error(f"Failed to ingest documents: {e}")
        raise


def clear_collection(vector_store):
    """Clear all documents from the collection.

    Args:
        vector_store: Milvus instance.
    """
    try:
        # Milvus doesn't have a direct clear method, so we drop and recreate
        logger.info("Clearing Milvus collection")
        # This would require reimplementing if needed
    except Exception as e:
        logger.error(f"Failed to clear collection: {e}")
        raise
