from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.database.base import get_db
from app.schemas.camera import (
    CameraCreate, CameraUpdate, CameraCalibration, CameraResponse
)
from app.services.camera_service import camera_service
from app.config import settings

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.post("/", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_data: CameraCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new camera"""
    camera = await camera_service.create_camera(db, camera_data)
    return camera


@router.get("/", response_model=List[CameraResponse])
async def list_cameras(
    active_only: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """List all cameras"""
    cameras = await camera_service.get_all_cameras(db, active_only)
    return cameras


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific camera"""
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found"
        )
    return camera


@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: str,
    camera_data: CameraUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a camera"""
    camera = await camera_service.update_camera(db, camera_id, camera_data)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found"
        )
    return camera


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a camera"""
    success = await camera_service.delete_camera(db, camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found"
        )


@router.post("/{camera_id}/calibrate", response_model=CameraResponse)
async def calibrate_camera(
    camera_id: str,
    calibration_data: CameraCalibration,
    db: AsyncSession = Depends(get_db)
):
    """Calibrate a camera for pixel-to-meter conversion"""
    camera = await camera_service.calibrate_camera(db, camera_id, calibration_data)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Calibration failed"
        )
    return camera


@router.get("/{camera_id}/models")
async def get_available_models(camera_id: str):
    """Get available detection models"""
    return {
        "camera_id": camera_id,
        "available_models": list(settings.AVAILABLE_MODELS.keys())
    }