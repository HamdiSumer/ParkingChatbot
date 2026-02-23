"""LLM provider factory supporting multiple LLM backends."""
from typing import Any
from src.config import get_config
from src.utils.logging import logger

config = get_config()


def create_llm(temperature: float = 0.3) -> Any:
    """Create LLM instance based on configured provider.

    Args:
        temperature: Controls randomness (0-1). Higher values = more creative.

    Returns:
        LLM instance (Ollama, OpenAI, Gemini, or Anthropic).

    Raises:
        ValueError: If provider not configured or credentials missing.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "ollama":
        return _create_ollama_llm(temperature)
    elif provider == "openai":
        return _create_openai_llm(temperature)
    elif provider == "gemini":
        return _create_gemini_llm(temperature)
    elif provider == "anthropic":
        return _create_anthropic_llm(temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _create_ollama_llm(temperature: float) -> Any:
    """Create Ollama LLM instance."""
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(
            base_url=config.OLLAMA_HOST,
            model=config.OLLAMA_MODEL,
            temperature=temperature,
        )
        logger.info(f"✓ Ollama LLM initialized: {config.OLLAMA_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        raise


def _create_openai_llm(temperature: float) -> Any:
    """Create OpenAI LLM instance."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            temperature=temperature,
        )
        logger.info(f"✓ OpenAI LLM initialized: {config.OPENAI_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI: {e}")
        raise


def _create_gemini_llm(temperature: float) -> Any:
    """Create Google Gemini LLM instance."""
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in .env")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=temperature,
        )
        logger.info(f"✓ Gemini LLM initialized: {config.GEMINI_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        raise


def _create_anthropic_llm(temperature: float) -> Any:
    """Create Anthropic Claude LLM instance."""
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")

    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            api_key=config.ANTHROPIC_API_KEY,
            model=config.ANTHROPIC_MODEL,
            temperature=temperature,
        )
        logger.info(f"✓ Anthropic LLM initialized: {config.ANTHROPIC_MODEL}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic: {e}")
        raise
