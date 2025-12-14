import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import cameras, health, http_camera_proxy, onvif, ptz_control, websocket
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
    description="Real-time multi-camera object detection and tracking system with PTZ control",
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
app.include_router(onvif.router, prefix=settings.API_V1_PREFIX)
app.include_router(websocket.router)
app.include_router(http_camera_proxy.router, prefix="/api/v1", tags=["camera-proxy"])
app.include_router(ptz_control.router, prefix=settings.API_V1_PREFIX)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting SeeDeep.AI...")

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


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down SeeDeep.AI...")
