# app/main.py - CLEAN VERSION USING ENRICHED SCHEMAS

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    alerts,
    cameras,
    health,
    http_camera_proxy,
    ptz_control,
    websocket,
)
from app.config import settings
from app.core.detection.yolo_detector import detector
from app.database.base import init_db
from app.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
# SeeDeep.AI REST API & WebSocket Documentation

Real-time multi-camera object detection and tracking system with alerts.

## Features
- 📹 **Camera Management** - Add, configure, and manage multiple cameras
- 🎯 **Real-time Object Detection** - Detect objects using YOLO models
- 🔍 **Object Tracking** - Track objects across frames with unique IDs
- 🚨 **Alert System** - Speed, tracking, and distance alerts
- 📧 **Email Notifications** - Automated alert emails
- 🎥 **PTZ Camera Control** - ONVIF-based pan-tilt-zoom control
- 📡 **WebSocket Streaming** - Real-time video and data streaming

## WebSocket Endpoint
**URL:** `ws://{host}:{port}/ws/camera/{camera_id}`

Real-time streaming of:
- Video frames (base64-encoded)
- Object detections
- Tracking data
- Speed and distance measurements
- Alerts

## Available Models
- **general_detection** - Person, car, bicycle, motorcycle, etc.
- **face_detection** - Faces with/without masks
- **cap_detection** - Hard hats and safety caps
- **weapon_detection** - Weapons (knife, pistol)
- **fire_detection** - Fire and smoke

## Quick Start
1. **Add Camera:** `POST /api/v1/cameras`
2. **Calibrate (optional):** `POST /api/v1/cameras/{id}/calibrate`
3. **Configure Alerts:** `PUT /api/v1/alerts/{id}/config`
4. **Connect WebSocket:** `ws://localhost:8000/ws/camera/{id}`

## Documentation Formats
- **Interactive Swagger UI:** `/docs` (this page)
- **ReDoc:** `/redoc` (alternative UI)
- **OpenAPI JSON:** Auto-generated spec
- **Static Markdown:** Run `python generate_docs.py`
    """,
    debug=settings.DEBUG,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(cameras.router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket.router)
app.include_router(http_camera_proxy.router, prefix="/api/v1", tags=["camera-proxy"])
app.include_router(ptz_control.router, prefix=settings.API_V1_PREFIX)
app.include_router(alerts.router, prefix=settings.API_V1_PREFIX)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting SeeDeep.AI...")

    logger.info("\n" + "=" * 60)
    logger.info("🖥️  RUNTIME DEVICE CHECK")
    logger.info("=" * 60)

    device = settings.DEVICE
    logger.info(f"Configured device: {device}")

    if device.type == "cuda":
        logger.info(f"✅ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        logger.info(f"✅ Using Apple Silicon GPU (MPS)")
    else:
        logger.info(f"⚠️  Using CPU (no GPU acceleration)")

    logger.info("=" * 60 + "\n")
    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

    # Preload models (optional)
    logger.info("📦 Loading detection models...")
    for model_name in settings.AVAILABLE_MODELS.keys():
        try:
            detector.load_model(model_name)
        except Exception as e:
            logger.warning(f"⚠️ Could not preload {model_name}: {e}")

    logger.info("✅ Application started successfully")
    logger.info(f"📖 API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"📖 Alternative Docs: http://{settings.HOST}:{settings.PORT}/redoc")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down SeeDeep.AI...")
