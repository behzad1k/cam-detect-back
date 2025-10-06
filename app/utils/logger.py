import logging
import sys
from pathlib import Path
from app.config import settings


def setup_logging():
  """Configure application logging"""

  # Create logs directory
  log_dir = Path("logs")
  log_dir.mkdir(exist_ok=True)

  # Configure logging format
  log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

  # Set log level
  log_level = getattr(logging, settings.LOG_LEVEL.upper())

  # Configure handlers
  handlers = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(log_dir / "app.log")
  ]

  # Basic config
  logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=handlers
  )

  # Silence noisy loggers
  logging.getLogger("ultralytics").setLevel(logging.WARNING)
  logging.getLogger("torch").setLevel(logging.WARNING)
  logging.getLogger("PIL").setLevel(logging.WARNING)

  logger = logging.getLogger(__name__)
  logger.info(f"✅ Logging configured at {log_level} level")