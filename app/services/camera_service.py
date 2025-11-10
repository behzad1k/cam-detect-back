# app/services/camera_service.py - FIXED VERSION WITH DEBUG LOGGING

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import logging

from app.database.models import Camera
from app.schemas.camera import (
  CameraCreate, CameraUpdate, CameraCalibration, FeatureConfiguration
)
from app.core.calibration.calibrator import calibrator

logger = logging.getLogger(__name__)


class CameraService:
  """Service for camera CRUD operations with enhanced features"""

  @staticmethod
  def _detect_models_from_classes(selected_classes: List[str]) -> List[str]:
    """Auto-detect required models based on selected classes"""
    # Import model definitions
    from app.config import settings

    # Map classes to models (this should match your MODEL_DEFINITIONS)
    class_to_model = {
      'Hardhat': 'ppe_detection',
      'Mask': 'ppe_detection',
      'NO-Hardhat': 'ppe_detection',
      'NO-Mask': 'ppe_detection',
      'NO-Safety Vest': 'ppe_detection',
      'Person': 'ppe_detection',
      'Safety Cone': 'ppe_detection',
      'Safety Vest': 'ppe_detection',
      'Machinery': 'ppe_detection',
      'General': 'ppe_detection',
      'no_mask': 'face_detection',
      'mask': 'face_detection',
      'no_cap': 'cap_detection',
      'cap': 'cap_detection',
      'pistol': 'weapon_detection',
      'knife': 'weapon_detection',
      'person': 'general_detection',
      'bicycle': 'general_detection',
      'car': 'general_detection',
      'motorcycle': 'general_detection',
      'smoke': 'fire_detection',
      'fire': 'fire_detection',
    }

    detected_models = set()
    for cls in selected_classes:
      model = class_to_model.get(cls)
      if model and model in settings.AVAILABLE_MODELS:
        detected_models.add(model)

    result = list(detected_models)
    logger.info(f"Auto-detected models from classes {selected_classes}: {result}")
    return result

  @staticmethod
  async def create_camera(db: AsyncSession, camera_data: CameraCreate) -> Camera:
    """Create a new camera with auto-model detection"""

    # Auto-detect models from selected classes
    if camera_data.selected_classes and not camera_data.active_models:
      camera_data.active_models = CameraService._detect_models_from_classes(
        camera_data.selected_classes
      )

    # Build RTSP URL if not provided
    if not camera_data.rtsp_url and camera_data.ipAddress:
      camera_data.rtsp_url = f"{camera_data.protocol}://{camera_data.ipAddress}:{camera_data.port}/stream"
      logger.info(f"Built RTSP URL: {camera_data.rtsp_url}")

    # Create camera object
    camera = Camera(
      name=camera_data.name,
      location=camera_data.location,
      rtsp_url=camera_data.rtsp_url,
      width=camera_data.width,
      height=camera_data.height,
      fps=camera_data.fps,
      features=camera_data.features or {},
      active_models=camera_data.active_models or []
    )

    logger.info(f"Creating camera with RTSP URL: {camera.rtsp_url}")
    logger.info(f"Creating camera with active models: {camera.active_models}")

    # Perform calibration if provided
    if camera_data.calibration and len(camera_data.calibration.points) >= 2:
      pixels_per_meter = None

      if camera_data.calibration.mode == "reference_object":
        pixels_per_meter = calibrator.calibrate_reference_object(
          [p.dict() for p in camera_data.calibration.points]
        )

      if pixels_per_meter:
        camera.is_calibrated = True
        camera.pixels_per_meter = pixels_per_meter
        camera.calibration_mode = camera_data.calibration.mode
        camera.calibration_points = [p.dict() for p in camera_data.calibration.points]
        logger.info(f"✅ Camera calibrated on creation: {pixels_per_meter:.2f} px/m")

    db.add(camera)
    await db.commit()
    await db.refresh(camera)

    logger.info(f"✅ Created camera: {camera.name} ({camera.id})")
    logger.info(f"   RTSP URL: {camera.rtsp_url}")
    logger.info(f"   Active Models: {camera.active_models}")
    return camera

  @staticmethod
  async def get_camera(db: AsyncSession, camera_id: str) -> Optional[Camera]:
    """Get camera by ID"""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    return result.scalar_one_or_none()

  @staticmethod
  async def get_all_cameras(db: AsyncSession, active_only: bool = False) -> List[Camera]:
    """Get all cameras"""
    query = select(Camera)
    if active_only:
      query = query.where(Camera.is_active == True)

    result = await db.execute(query)
    return result.scalars().all()

  @staticmethod
  async def update_camera(
    db: AsyncSession,
    camera_id: str,
    camera_data: CameraUpdate
  ) -> Optional[Camera]:
    """Update camera - FIXED with extensive logging"""
    logger.info(f"=" * 80)
    logger.info(f"🔄 UPDATE CAMERA REQUEST RECEIVED")
    logger.info(f"Camera ID: {camera_id}")
    logger.info(f"Raw update data: {camera_data}")

    # Get current camera state
    camera = await CameraService.get_camera(db, camera_id)
    if not camera:
      logger.error(f"❌ Camera {camera_id} not found in database")
      return None

    logger.info(f"📊 Current camera state:")
    logger.info(f"   Name: {camera.name}")
    logger.info(f"   Location: {camera.location}")
    logger.info(f"   Active Models: {camera.active_models}")
    logger.info(f"   Features: {camera.features}")

    # Get only fields that are being updated
    update_data = camera_data.dict(exclude_unset=True)
    logger.info(f"📝 Fields to update: {list(update_data.keys())}")
    logger.info(f"📝 Update values: {update_data}")

    # CRITICAL: If active_models is not in the update data, preserve it
    if 'active_models' not in update_data:
      logger.info(f"✅ Preserving active_models: {camera.active_models}")
    else:
      logger.info(f"🔄 Updating active_models from {camera.active_models} to {update_data['active_models']}")

    # Apply updates one by one with logging
    for key, value in update_data.items():
      if hasattr(camera, key):
        old_value = getattr(camera, key)
        setattr(camera, key, value)
        logger.info(f"   ✓ Updated {key}: {old_value} → {value}")
      else:
        logger.warning(f"   ⚠️ Field {key} does not exist on Camera model")

    try:
      # CRITICAL FIX: Flush changes before commit
      await db.flush()
      await db.commit()
      await db.refresh(camera)

      logger.info(f"✅ Camera {camera_id} updated successfully in database")
      logger.info(f"📊 New camera state:")
      logger.info(f"   Name: {camera.name}")
      logger.info(f"   Location: {camera.location}")
      logger.info(f"   Active Models: {camera.active_models}")
      logger.info(f"   Features: {camera.features}")
      logger.info(f"=" * 80)

      return camera

    except Exception as e:
      logger.error(f"❌ COMMIT FAILED: {e}")
      logger.error(f"Exception type: {type(e)}")
      logger.error(f"Exception details: {str(e)}")
      await db.rollback()
      logger.info(f"🔄 Changes rolled back")
      logger.info(f"=" * 80)
      raise

  @staticmethod
  async def update_features(
    db: AsyncSession,
    camera_id: str,
    features: FeatureConfiguration
  ) -> Optional[Camera]:
    """Update camera feature configuration - FIXED with logging"""
    logger.info(f"=" * 80)
    logger.info(f"🔄 UPDATE FEATURES REQUEST RECEIVED")
    logger.info(f"Camera ID: {camera_id}")

    camera = await CameraService.get_camera(db, camera_id)
    if not camera:
      logger.error(f"❌ Camera {camera_id} not found")
      return None

    logger.info(f"📊 Current features: {camera.features}")

    feature_dict = features.dict(exclude_unset=True)
    logger.info(f"📝 Feature updates: {feature_dict}")

    current_features = camera.features or {}
    current_features.update(feature_dict)
    camera.features = current_features

    logger.info(f"📊 New features: {camera.features}")

    try:
      await db.flush()
      await db.commit()
      await db.refresh(camera)

      logger.info(f"✅ Features updated successfully for camera {camera_id}")
      logger.info(f"=" * 80)
      return camera

    except Exception as e:
      logger.error(f"❌ FEATURE UPDATE FAILED: {e}")
      await db.rollback()
      logger.info(f"=" * 80)
      raise

  @staticmethod
  async def delete_camera(db: AsyncSession, camera_id: str) -> bool:
    """Delete camera"""
    camera = await CameraService.get_camera(db, camera_id)
    if not camera:
      return False

    await db.delete(camera)
    await db.commit()

    logger.info(f"🗑️ Deleted camera: {camera_id}")
    return True

  @staticmethod
  async def update_detection_classes(
    db: AsyncSession,
    camera_id: str,
    detection_classes: List[str]
  ) -> Optional[Camera]:
    """Update detection classes for a camera"""
    logger.info(f"Updating detection classes for camera {camera_id}")

    camera = await CameraService.get_camera(db, camera_id)
    if not camera:
      logger.error(f"Camera {camera_id} not found")
      return None

    current_features = camera.features or {}
    current_features["detection_classes"] = detection_classes

    # Auto-detect models from classes
    camera.active_models = CameraService._detect_models_from_classes(detection_classes)
    camera.features = current_features

    try:
      await db.flush()
      await db.commit()
      await db.refresh(camera)

      logger.info(f"✅ Updated detection classes for camera {camera_id}")
      logger.info(f"   Classes: {detection_classes}")
      logger.info(f"   Active models: {camera.active_models}")
      return camera

    except Exception as e:
      logger.error(f"❌ Detection class update failed: {e}")
      await db.rollback()
      raise

  @staticmethod
  async def calibrate_camera(
    db: AsyncSession,
    camera_id: str,
    calibration_data: CameraCalibration
  ) -> Optional[Camera]:
    """Calibrate a camera"""
    camera = await CameraService.get_camera(db, camera_id)
    if not camera:
      return None

    pixels_per_meter = None

    if calibration_data.mode == "reference_object":
      pixels_per_meter = calibrator.calibrate_reference_object(
        [p.dict() for p in calibration_data.points]
      )

    if pixels_per_meter:
      camera.is_calibrated = True
      camera.pixels_per_meter = pixels_per_meter
      camera.calibration_mode = calibration_data.mode
      camera.calibration_points = [p.dict() for p in calibration_data.points]

      await db.commit()
      await db.refresh(camera)

      logger.info(f"✅ Calibrated camera {camera_id}: {pixels_per_meter:.2f} px/m")
      return camera
    else:
      logger.error(f"❌ Calibration failed for camera {camera_id}")
      return None


camera_service = CameraService()