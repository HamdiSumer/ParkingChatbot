"""Ollama LLM integration for local language models."""
from langchain_community.llms import Ollama
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from src.config import get_config
from src.utils.logging import logger

config = get_config()


def create_ollama_llm(temperature: float = 0.7, top_p: float = 0.9):
    """Create and configure Ollama LLM instance.

    Args:
        temperature: Controls randomness (0-1). Higher values = more creative.
        top_p: Nucleus sampling parameter.

    Returns:
        Ollama LLM instance configured with streaming.
    """
    try:
        llm = Ollama(
            base_url=config.OLLAMA_HOST,
            model=config.OLLAMA_MODEL,
            temperature=temperature,
            top_p=top_p,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        logger.info(f"Ollama LLM initialized with model: {config.OLLAMA_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM: {e}")
        raise


def create_ollama_llm_silent(temperature: float = 0.7, top_p: float = 0.9):
    """Create Ollama LLM without streaming output (for programmatic use).

    Args:
        temperature: Controls randomness (0-1). Higher values = more creative.
        top_p: Nucleus sampling parameter.

    Returns:
        Ollama LLM instance without streaming.
    """
    try:
        llm = Ollama(
            base_url=config.OLLAMA_HOST,
            model=config.OLLAMA_MODEL,
            temperature=temperature,
            top_p=top_p,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM: {e}")
        raise
