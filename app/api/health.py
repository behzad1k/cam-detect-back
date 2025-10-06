from fastapi import APIRouter
from app.config import settings
from app.services.stream_manager import stream_manager

router = APIRouter(tags=["health"])


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running"
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(settings.DEVICE),
        "available_models": list(settings.AVAILABLE_MODELS.keys()),
        "active_streams": stream_manager.get_active_count()
    }