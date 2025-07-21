import uvicorn
import logging
from app.main import app
from app.core.config import settings
from app.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
  logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
  logger.info(f"Debug mode: {settings.DEBUG}")
  logger.info(f"Auto-reload: {settings.RELOAD}")
  logger.info(f"Device: {settings.DEVICE}")

  uvicorn.run(
    "app.main:app",
    host=settings.HOST,
    port=settings.PORT,
    reload=settings.RELOAD,
    log_level="debug" if settings.DEBUG else "info"
  )
