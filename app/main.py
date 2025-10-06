# app/main.py - Updated for Multi-Camera Support
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
from .api.routes import router as main_router
from app.api.tracking_routes import router as tracking_router

# NEW: Import the multi-camera handler
from .websocket.multi_camera_handlers import multi_camera_handler

# Initialize settings and logging
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

# Include routers
app.include_router(main_router)
app.include_router(tracking_router)

# Global instances
model_manager = ModelManager()
connection_manager = ConnectionManager()


# NEW: Multi-Camera WebSocket endpoint
@app.websocket("/ws")
async def multi_camera_websocket_endpoint(websocket: WebSocket):
  """Enhanced WebSocket endpoint supporting multiple camera streams"""

  await multi_camera_handler.handle_websocket(websocket)


# OPTIONAL: Keep the old endpoint for backward compatibility
@app.websocket("/ws/legacy")
async def legacy_websocket_endpoint(websocket: WebSocket):
  """Legacy single-camera WebSocket endpoint (backward compatibility)"""
  # You can keep your old unified_handler here if needed
  from .websocket.handlers import unified_handler
  await unified_handler.handle_websocket(websocket)


# Additional endpoints for camera management
@app.get("/cameras/status")
async def get_cameras_status():
  """Get status of all active cameras"""
  active_cameras = []
  for camera_id, camera_conn in multi_camera_handler.camera_connections.items():
    active_cameras.append({
      "camera_id": camera_id,
      "name": camera_conn.camera_name,
      "location": camera_conn.camera_location,
      "is_streaming": camera_conn.is_streaming,
      "tracking_enabled": multi_camera_handler.camera_tracking_enabled.get(camera_id, False),
      "stats": multi_camera_handler.camera_stats.get(camera_id, {})
    })

  return {
    "total_cameras": len(active_cameras),
    "streaming_cameras": len([c for c in active_cameras if c["is_streaming"]]),
    "cameras": active_cameras
  }


@app.get("/cameras/{camera_id}/stats")
async def get_camera_stats(camera_id: str):
  """Get detailed stats for a specific camera"""
  if camera_id not in multi_camera_handler.camera_connections:
    return {"error": "Camera not found"}, 404

  camera_conn = multi_camera_handler.camera_connections[camera_id]
  stats = multi_camera_handler.camera_stats.get(camera_id, {})

  return {
    "camera_id": camera_id,
    "name": camera_conn.camera_name,
    "location": camera_conn.camera_location,
    "is_streaming": camera_conn.is_streaming,
    "tracking_enabled": multi_camera_handler.camera_tracking_enabled.get(camera_id, False),
    "models": multi_camera_handler.camera_models.get(camera_id, []),
    "stats": stats,
    "config": multi_camera_handler.camera_configs.get(camera_id, {})
  }


# Include your existing API routes
app.include_router(router)


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
  logger.info("Multi-camera WebSocket handler ready")


@app.on_event("shutdown")
async def shutdown_event():
  """Shutdown event handler"""
  try:
    # Clean up all camera connections
    logger.info("Shutting down multi-camera handler...")

    # Stop all active camera streams
    for camera_id in list(multi_camera_handler.camera_connections.keys()):
      if multi_camera_handler.camera_tracking_enabled.get(camera_id, False):
        multi_camera_handler.tracker_manager.remove_tracker(camera_id)

    logger.info("Multi-camera handler shutdown completed")

  except Exception as e:
    logger.error(f"Error during shutdown: {e}")


# Health check endpoint with camera info
@app.get("/health")
async def enhanced_health_check():
  """Enhanced health check with camera information"""
  return {
    "status": "healthy",
    "device": str(settings.DEVICE),
    "models_available": model_manager.get_available_models(),
    "models_loaded": model_manager.get_loaded_models(),
    "active_cameras": len(multi_camera_handler.camera_connections),
    "streaming_cameras": len([
      c for c in multi_camera_handler.camera_connections.values()
      if c.is_streaming
    ]),
    "tracking_available": multi_camera_handler.tracking_available,
    "active_connections": len(multi_camera_handler.active_connections)
  }