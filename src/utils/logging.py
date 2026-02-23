"""Logging configuration."""
import logging
from src.config import get_config

config = get_config()

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()
