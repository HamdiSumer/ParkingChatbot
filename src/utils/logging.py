"""Logging configuration."""
import logging
import warnings
from src.config import get_config

config = get_config()

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Suppress third-party library logs
    logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL)
    logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('langchain').setLevel(logging.CRITICAL)
    logging.getLogger('torch').setLevel(logging.CRITICAL)
    logging.getLogger('transformers').setLevel(logging.CRITICAL)

    # Suppress deprecation warnings if LOG_LEVEL is CRITICAL
    if config.LOG_LEVEL == 'CRITICAL':
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

    return logging.getLogger(__name__)

logger = setup_logging()
