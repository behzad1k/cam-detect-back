from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import cv2
import base64
from pydantic import BaseModel
from app.database.base import get_db
from app.schemas.camera import (
  CameraCreate, CameraUpdate, CameraCalibration, CameraResponse, FeatureConfiguration
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
    print(camera_data)
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

@router.patch("/{camera_id}/features", response_model=CameraResponse)
async def update_camera_features(
    camera_id: str,
    features: FeatureConfiguration,
    db: AsyncSession = Depends(get_db)
):
    """Update camera feature configuration"""
    camera = await camera_service.update_features(db, camera_id, features)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found"
        )
    return camera


class TestConnectionRequest(BaseModel):
  rtsp_url: str


@router.post("/test-connection")
async def test_camera_connection(request: TestConnectionRequest):
  """Test camera connection and return a preview frame"""
  try:
    cap = cv2.VideoCapture(request.rtsp_url)

    if not cap.isOpened():
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Failed to connect to camera"
      )

    # Read a frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Failed to read frame from camera"
      )

    # Resize frame for preview
    frame = cv2.resize(frame, (640, 480))

    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
      "success": True,
      "preview_frame": frame_base64,
      "width": 640,
      "height": 480
    }

  except Exception as e:
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Connection test failed: {str(e)}"
    )
