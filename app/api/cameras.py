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
import asyncio

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
  """Update a camera - FIXED with retry logic for concurrent access"""
  import logging
  logger = logging.getLogger(__name__)

  logger.info(f"📝 Received update request for camera {camera_id}")
  logger.info(f"📝 Update data: {camera_data.dict(exclude_unset=True)}")

  # CRITICAL FIX: Add retry logic for database locks
  max_retries = 3
  retry_delay = 0.5

  for attempt in range(max_retries):
    try:
      camera = await camera_service.update_camera(db, camera_id, camera_data)
      if not camera:
        raise HTTPException(
          status_code=status.HTTP_404_NOT_FOUND,
          detail=f"Camera {camera_id} not found"
        )
      logger.info(f"✅ Camera {camera_id} updated successfully on attempt {attempt + 1}")
      return camera

    except Exception as e:
      error_msg = str(e).lower()

      # Check if it's a database lock error
      if 'locked' in error_msg or 'busy' in error_msg:
        if attempt < max_retries - 1:
          logger.warning(f"⚠️ Database locked, retrying... (attempt {attempt + 1}/{max_retries})")
          await asyncio.sleep(retry_delay)
          retry_delay *= 2  # Exponential backoff
          continue
        else:
          logger.error(f"❌ Database locked after {max_retries} attempts")
          raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is busy. Please try again."
          )
      else:
        # Other error, don't retry
        logger.error(f"❌ Update error: {e}")
        raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=str(e)
        )

  # Should never reach here
  raise HTTPException(
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    detail="Update failed"
  )


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
  """Update camera feature configuration - FIXED with retry logic"""
  import logging
  logger = logging.getLogger(__name__)

  logger.info(f"📝 Received feature update for camera {camera_id}")
  logger.info(f"📝 Features: {features.dict(exclude_unset=True)}")

  max_retries = 3
  retry_delay = 0.5

  for attempt in range(max_retries):
    try:
      camera = await camera_service.update_features(db, camera_id, features)
      if not camera:
        raise HTTPException(
          status_code=status.HTTP_404_NOT_FOUND,
          detail=f"Camera {camera_id} not found"
        )
      logger.info(f"✅ Features updated successfully on attempt {attempt + 1}")
      return camera

    except Exception as e:
      error_msg = str(e).lower()
      if 'locked' in error_msg or 'busy' in error_msg:
        if attempt < max_retries - 1:
          logger.warning(f"⚠️ Database locked, retrying...")
          await asyncio.sleep(retry_delay)
          retry_delay *= 2
          continue
        else:
          raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is busy. Please try again."
          )
      else:
        logger.error(f"❌ Feature update error: {e}")
        raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=str(e)
        )


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

@router.patch("/{camera_id}/detection-classes", response_model=CameraResponse)
async def update_detection_classes(
  camera_id: str,
  classes: List[str],
  db: AsyncSession = Depends(get_db)
):
  """Update detection classes for a camera"""
  import logging
  logger = logging.getLogger(__name__)

  logger.info(f"📝 Received detection class update for camera {camera_id}")
  logger.info(f"📝 Classes: {classes}")

  max_retries = 3
  retry_delay = 0.5

  for attempt in range(max_retries):
    try:
      camera = await camera_service.update_detection_classes(db, camera_id, classes)
      if not camera:
        raise HTTPException(
          status_code=status.HTTP_404_NOT_FOUND,
          detail=f"Camera {camera_id} not found"
        )
      logger.info(f"✅ Detection classes updated successfully")
      return camera

    except Exception as e:
      error_msg = str(e).lower()
      if 'locked' in error_msg or 'busy' in error_msg:
        if attempt < max_retries - 1:
          logger.warning(f"⚠️ Database locked, retrying...")
          await asyncio.sleep(retry_delay)
          retry_delay *= 2
          continue
        else:
          raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is busy. Please try again."
          )
      else:
        logger.error(f"❌ Detection class update error: {e}")
        raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=str(e)
        )


@router.get("/{camera_id}/calibration")
async def get_calibration_info(
  camera_id: str,
  db: AsyncSession = Depends(get_db)
):
  """Get current calibration information"""
  camera = await camera_service.get_camera(db, camera_id)
  if not camera:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Camera {camera_id} not found"
    )

  return {
    "is_calibrated": camera.is_calibrated,
    "pixels_per_meter": camera.pixels_per_meter,
    "calibration_mode": camera.calibration_mode,
    "calibration_points": camera.calibration_points,
    "width": camera.width,
    "height": camera.height,
    "camera_id": camera.id,
    "name": camera.name
  }


@router.delete("/{camera_id}/calibration")
async def clear_calibration(
  camera_id: str,
  db: AsyncSession = Depends(get_db)
):
  """Clear camera calibration"""
  camera = await camera_service.get_camera(db, camera_id)
  if not camera:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Camera {camera_id} not found"
    )

  camera.is_calibrated = False
  camera.pixels_per_meter = None
  camera.calibration_mode = None
  camera.calibration_points = None

  await db.commit()
  await db.refresh(camera)

  return {
    "success": True,
    "message": "Calibration cleared successfully"
  }


@router.post("/{camera_id}/calibration/test")
async def test_calibration(
  camera_id: str,
  calibration_data: CameraCalibration,
  db: AsyncSession = Depends(get_db)
):
  """Test calibration without saving (preview mode)"""
  from app.core.calibration.calibrator import calibrator

  camera = await camera_service.get_camera(db, camera_id)
  if not camera:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Camera {camera_id} not found"
    )

  pixels_per_meter = None

  if calibration_data.mode == "reference_object":
    pixels_per_meter = calibrator.calibrate_reference_object(
      [p.dict() for p in calibration_data.points]
    )

  if pixels_per_meter:
    return {
      "success": True,
      "pixels_per_meter": pixels_per_meter,
      "mode": calibration_data.mode,
      "points_count": len(calibration_data.points)
    }
  else:
    return {
      "success": False,
      "error": "Calibration test failed"
    }


@router.get("/{camera_id}/frame")
async def get_camera_frame(
  camera_id: str,
  db: AsyncSession = Depends(get_db)
):
  """Get a single frame from camera for calibration"""
  camera = await camera_service.get_camera(db, camera_id)
  if not camera:
    raise HTTPException(
      status_code=status.HTTP_404_NOT_FOUND,
      detail=f"Camera {camera_id} not found"
    )

  if not camera.rtsp_url:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="Camera has no RTSP URL configured"
    )

  try:
    cap = cv2.VideoCapture(camera.rtsp_url)

    if not cap.isOpened():
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Failed to connect to camera"
      )

    ret, frame = cap.read()
    cap.release()

    if not ret:
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Failed to read frame from camera"
      )

    # Resize if needed
    max_width = 1920
    if frame.shape[1] > max_width:
      scale = max_width / frame.shape[1]
      new_width = int(frame.shape[1] * scale)
      new_height = int(frame.shape[0] * scale)
      frame = cv2.resize(frame, (new_width, new_height))

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
      "success": True,
      "frame": frame_base64,
      "width": frame.shape[1],
      "height": frame.shape[0],
      "camera_id": camera_id
    }

  except Exception as e:
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"Failed to capture frame: {str(e)}"
    )
