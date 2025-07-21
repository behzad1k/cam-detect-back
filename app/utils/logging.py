import os
import logging
from pathlib import Path


def setup_logging():
  """Setup logging configuration"""
  log_level = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO

  # Create logs directory if it doesn't exist
  log_dir = Path("logs")
  log_dir.mkdir(exist_ok=True)

  # Configure logging
  logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
      logging.StreamHandler(),  # Console output
      logging.FileHandler(log_dir / "app.log")  # File output
    ]
  )

  # Set specific loggers
  logging.getLogger("ultralytics").setLevel(logging.WARNING)
  logging.getLogger("torch").setLevel(logging.WARNING)

  return logging.getLogger(__name__)


def get_logger(name: str = None):
  """Get a logger instance"""
  return logging.getLogger(name or __name__)
