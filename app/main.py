from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.api.routes import router
from app.models.model_manager import model_manager
from app.websocket.handlers import websocket_endpoint
from app.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
  title=settings.TITLE,
  root_path=settings.ROOT_PATH,
  description=settings.DESCRIPTION,
  version=settings.VERSION,
  debug=settings.DEBUG,
)

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=settings.ALLOWED_ORIGINS,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Include API routes
app.include_router(router)


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
  await websocket_endpoint(websocket)


# Startup event
@app.on_event("startup")
async def startup_event():
  """Load all models on application startup"""
  logger.info("Starting up... Loading all models")
  load_results = model_manager.load_all_models()

  # Log loading results
  for model_name, success in load_results.items():
    if success:
      logger.info(f"Successfully loaded model: {model_name}")
    else:
      logger.error(f"Failed to load model: {model_name}")

  logger.info("Model loading completed")
