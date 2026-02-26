"""Logging configuration."""
import logging
import warnings
import sys
from contextlib import contextmanager
from src.config import get_config

config = get_config()

# Global flag for quiet mode
_quiet_mode = False


def setup_logging(level: str = None):
    """Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses config.LOG_LEVEL.
    """
    log_level = level or config.LOG_LEVEL

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Suppress third-party library logs
    logging.getLogger('sentence_transformers').setLevel(logging.CRITICAL)
    logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    logging.getLogger('langchain').setLevel(logging.CRITICAL)
    logging.getLogger('langchain_core').setLevel(logging.CRITICAL)
    logging.getLogger('langchain_community').setLevel(logging.CRITICAL)
    logging.getLogger('torch').setLevel(logging.CRITICAL)
    logging.getLogger('transformers').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)
    logging.getLogger('weaviate').setLevel(logging.CRITICAL)

    return logging.getLogger(__name__)


def set_quiet_mode(quiet: bool = True):
    """Enable or disable quiet mode (suppresses all logs).

    Args:
        quiet: If True, suppress all logs. If False, restore normal logging.
    """
    global _quiet_mode
    _quiet_mode = quiet

    if quiet:
        # Set all loggers to CRITICAL
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger('src').setLevel(logging.CRITICAL)
        # Suppress all warnings
        warnings.filterwarnings('ignore')
    else:
        # Restore to config level
        logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL))
        logging.getLogger('src').setLevel(getattr(logging, config.LOG_LEVEL))


@contextmanager
def quiet_logs():
    """Context manager to temporarily suppress all logs.

    Usage:
        with quiet_logs():
            # Code here runs without logging
            app = create_app()
    """
    original_level = logging.getLogger().level
    original_src_level = logging.getLogger('src').level

    try:
        set_quiet_mode(True)
        yield
    finally:
        logging.getLogger().setLevel(original_level)
        logging.getLogger('src').setLevel(original_src_level)


def suppress_warnings():
    """Suppress deprecation and resource warnings."""
    import os
    # Suppress all Python warnings
    warnings.filterwarnings('ignore')

    # Disable tqdm progress bars
    os.environ['TQDM_DISABLE'] = '1'

    # Suppress huggingface/transformers verbosity
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Suppress MLX verbosity (for sentence-transformers)
    os.environ['MLX_DISABLE_PROGRESS_BAR'] = '1'


logger = setup_logging()
