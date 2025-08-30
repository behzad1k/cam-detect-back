from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio

# Your existing imports
from app.core.config import settings
from app.models.model_manager import ModelManager
from app.websocket.connection_manager import ConnectionManager
from app.api.routes import router
from app.utils.logging import setup_logging

# New imports for background learning
from app.background_learning.core import BackgroundLearningSystem
from app.background_learning.integration import (
    EnhancedFrameProcessor,
    BackgroundLearningWebSocketHandler,
    create_background_learning_routes,
    BackgroundLearningConfig
)

# Initialize settings and logging
settings = get_settings()
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

# Global instances
model_manager = ModelManager()
connection_manager = ConnectionManager()
enhanced_processor = EnhancedFrameProcessor(model_manager)
bg_websocket_handler = BackgroundLearningWebSocketHandler(connection_manager, enhanced_processor)

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    try:
        # Save all background models before shutdown
        for camera_id, bg_system in enhanced_processor.background_systems.items():
            try:
                bg_system.save_model(f"models/background_{camera_id}.pkl")
                logger.info(f"Saved background model for camera {camera_id}")
            except Exception as e:
                logger.error(f"Failed to save background model for {camera_id}: {e}")

        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Include your existing API routes
app.include_router(router)

# Add background learning routes
create_background_learning_routes(app, enhanced_processor)

# Enhanced WebSocket endpoint
@app.websocket("/ws/enhanced")
async def websocket_endpoint(websocket: WebSocket, camera_id: str = "default"):
    """Enhanced WebSocket endpoint with background learning"""
    await bg_websocket_handler.handle_connection(websocket, camera_id)

# Keep your existing WebSocket endpoint for backward compatibility
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
