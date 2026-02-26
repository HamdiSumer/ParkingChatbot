"""Weaviate vector database integration for static parking data."""
from typing import List
from langchain_community.vectorstores import Weaviate
from langchain_core.documents import Document
from src.config import get_config
from src.rag.embeddings import create_embeddings
from src.utils.logging import logger

config = get_config()


def create_weaviate_connection(embeddings=None):
    """Create a connection to Weaviate vector database.

    Args:
        embeddings: Embeddings instance. If None, creates new one.

    Returns:
        Weaviate vector store instance.
    """
    if embeddings is None:
        embeddings = create_embeddings()

    try:
        import weaviate
        import requests
        import time

        # Parse the Weaviate host URL
        weaviate_url = config.WEAVIATE_HOST

        # Verify Weaviate is ready before connecting (HTTP health check)
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{weaviate_url}/v1/.well-known/ready", timeout=2)
                if response.status_code == 200:
                    # Give Weaviate extra time to fully initialize
                    time.sleep(2)
                    break
            except:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise ConnectionError(f"Weaviate not ready at {weaviate_url}")

        # Connect to Weaviate (v3 API) with extended timeout
        try:
            # Try with increased timeout
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client = weaviate.Client(
                    weaviate_url,
                    timeout_config=(30, 30)  # (connection_timeout, read_timeout)
                )
        except TypeError:
            # Fallback if timeout_config parameter not supported
            client = weaviate.Client(weaviate_url)

        # Verify connection
        if not client.is_ready():
            raise ConnectionError(f"Weaviate not ready at {weaviate_url}")

        # Create schema if it doesn't exist (with no vectorizer, use custom embeddings)
        try:
            schema = client.schema.get()
            # Check if our index exists
            class_exists = any(c.get("class") == config.WEAVIATE_INDEX_NAME for c in schema.get("classes", []))

            if not class_exists:
                # Create class with 'none' vectorizer (we provide vectors locally)
                class_obj = {
                    "class": config.WEAVIATE_INDEX_NAME,
                    "vectorizer": "none",  # Use custom vectors
                    "vectorIndexConfig": {
                        "size": 384  # Dimension of our embeddings
                    },
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"]
                        },
                        {
                            "name": "metadata_source",
                            "dataType": ["text"]
                        },
                        {
                            "name": "metadata_page",
                            "dataType": ["int"]
                        }
                    ]
                }
                client.schema.create_class(class_obj)
                logger.info(f"Created Weaviate class: {config.WEAVIATE_INDEX_NAME}")
        except Exception as e:
            logger.warning(f"Could not create schema: {e}")

        # Create vector store using LangChain with custom embeddings
        vector_store = Weaviate(
            client=client,
            index_name=config.WEAVIATE_INDEX_NAME,
            text_key="text",
            embedding=embeddings,
            by_text=False,  # Use vector search, not text search
        )

        logger.info(f"✓ Connected to Weaviate: {weaviate_url}")
        logger.info(f"✓ Index: {config.WEAVIATE_INDEX_NAME}")
        logger.info(f"✓ Using custom embeddings (no remote vectorization)")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise


def ingest_static_data_weaviate(vector_store, documents: List[Document]):
    """Ingest static parking data into Weaviate.

    Args:
        vector_store: Weaviate instance.
        documents: List of LangChain Document objects.

    Returns:
        Number of documents ingested.
    """
    try:
        # Add documents to Weaviate
        vector_store.add_documents(documents)
        logger.info(f"✓ Ingested {len(documents)} documents into Weaviate")
        return len(documents)
    except Exception as e:
        logger.error(f"Failed to ingest documents to Weaviate: {e}")
        raise


def clear_weaviate_index(client, index_name: str):
    """Clear all vectors from a Weaviate index.

    Args:
        client: Weaviate client (v4 API).
        index_name: Name of the index to clear.
    """
    try:
        client.collections.delete(index_name)
        logger.info(f"✓ Cleared Weaviate index: {index_name}")
    except Exception as e:
        logger.error(f"Failed to clear Weaviate index: {e}")
        raise
