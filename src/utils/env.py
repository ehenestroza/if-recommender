"""
Load environment variables from .env and configure logging for all scripts.

Call configure_logging() once at the top of each script before any other imports
that trigger HuggingFace network activity.
"""

import logging
import os
from pathlib import Path


def configure_logging(level: int = logging.INFO) -> None:
    """
    Set up root logger and silence noisy third-party loggers.

    Loads .env from the project root (if present) so that HF_TOKEN and other
    environment variables are available before any HuggingFace code runs.
    """
    # Load .env from project root (two levels up from this file: src/utils/ → src/ → project/)
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv not installed; rely on shell environment

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    # Silence per-request HTTP logs from the HuggingFace hub client.
    # Errors and warnings are still surfaced.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
