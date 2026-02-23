"""Embedding models for RAG."""
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import get_config
from src.utils.logging import logger

config = get_config()


def create_embeddings():
    """Create HuggingFace embeddings for document vectorization.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},  # Use CPU for local deployment
        )
        logger.info(f"Embeddings initialized with model: {config.EMBEDDING_MODEL}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        raise


def embed_text(embeddings, text: str) -> list:
    """Embed a single text string.

    Args:
        embeddings: HuggingFaceEmbeddings instance.
        text: Text to embed.

    Returns:
        Embedding vector as list.
    """
    return embeddings.embed_query(text)


def embed_texts(embeddings, texts: list) -> list:
    """Embed multiple text strings.

    Args:
        embeddings: HuggingFaceEmbeddings instance.
        texts: List of texts to embed.

    Returns:
        List of embedding vectors.
    """
    return embeddings.embed_documents(texts)
