# app/services/camera_service.py - FIXED WITH FIELD FILTERING

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.calibration.calibrator import calibrator
from app.database.models import Camera
from app.schemas.camera import (
    CameraCalibration,
    FeatureConfiguration,
)

logger = logging.getLogger(__name__)


class CameraService:
    # Valid Camera model fields (add new connection fields)
    VALID_CAMERA_FIELDS = {
        "id",
        "name",
        "location",
        "rtsp_url",
        # NEW connection fields
        "ip_address",
        "username",
        "password",
        "rtsp_port",
        "onvif_port",
        "http_port",
        "channel",
        "subtype",
        "stream_path",
        "manufacturer",
        # Resolution & settings
        "width",
        "height",
        "fps",
        # Calibration
        "is_calibrated",
        "pixels_per_meter",
        "calibration_mode",
        "calibration_points",
        # Features & models
        "features",
        "active_models",
        # Metadata
        "created_at",
        "updated_at",
        "is_active",
    }

    @staticmethod
    def filter_valid_fields(data: dict) -> dict:
        """
        Filter camera data to only include valid Camera model fields
        This prevents errors when frontend sends extra fields
        """
        valid_data = {}

        for key, value in data.items():
            if key in CameraService.VALID_CAMERA_FIELDS:
                valid_data[key] = value
            else:
                logger.debug(f"Ignoring invalid field: {key} = {value}")

        return valid_data

    @staticmethod
    def parse_camera_url(url: str) -> Dict[str, Any]:
        """
        Parse camera URL and extract connection components

        Examples:
        - rtsp://admin:Behzad8690@192.168.1.12:554/cam/realmonitor?channel=1&subtype=1
        - rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
        - onvif://admin:password@192.168.1.12:80/onvif/device_service
        - http://admin:password@192.168.1.12/cgi-bin/mjpg/video.cgi
        """
        parsed = urlparse(url)

        result = {
            "ip_address": parsed.hostname,
            "username": parsed.username or "admin",
            "password": parsed.password or "",
            "rtsp_port": 554,
            "onvif_port": 80,
            "http_port": 80,
            "channel": "1",
            "subtype": "0",
            "stream_path": None,
            "manufacturer": "unknown",
        }

        # Set port based on scheme
        if parsed.scheme == "rtsp":
            result["rtsp_port"] = parsed.port or 554
        elif parsed.scheme == "onvif":
            result["onvif_port"] = parsed.port or 80
        elif parsed.scheme in ["http", "https"]:
            result["http_port"] = parsed.port or 80

        # Parse path and query for stream details
        path = parsed.path
        query = parsed.query

        # Detect Dahua/Amcrest
        if "cam/realmonitor" in path:
            result["manufacturer"] = "dahua"
            result["stream_path"] = "cam/realmonitor"

            if query:
                params = dict(
                    param.split("=") for param in query.split("&") if "=" in param
                )
                result["channel"] = params.get("channel", "1")
                result["subtype"] = params.get("subtype", "0")

        # Detect Hikvision
        elif "Streaming/Channels" in path:
            result["manufacturer"] = "hikvision"
            result["stream_path"] = path.strip("/")

            match = re.search(r"/Streaming/Channels/(\d+)", path)
            if match:
                stream_id = match.group(1)
                result["subtype"] = "0" if stream_id == "101" else "1"
                result["channel"] = "1"

        # Detect Axis
        elif "axis-media" in path.lower():
            result["manufacturer"] = "axis"
            result["stream_path"] = path.strip("/")

        # Detect Foscam
        elif "videoMain" in path or "videoSub" in path:
            result["manufacturer"] = "foscam"
            result["stream_path"] = path.strip("/")
            result["subtype"] = "1" if "videoSub" in path else "0"

        # ONVIF generic
        elif parsed.scheme == "onvif":
            result["manufacturer"] = "onvif"

        # Custom/Unknown
        else:
            result["manufacturer"] = "custom"
            result["stream_path"] = path.strip("/")

        logger.info(
            f"📹 Parsed camera URL: {result['manufacturer']} @ {result['ip_address']}"
        )
        return result

    @staticmethod
    async def create_camera(db: AsyncSession, camera_data: dict) -> Camera:
        """
        Create a new camera with connection details
        Parses rtsp_url if provided and extracts connection components
        """
        logger.info(f"📹 Creating camera with data: {list(camera_data.keys())}")

        # If rtsp_url is provided, parse it to extract components
        if camera_data.get("rtsp_url") and not camera_data.get("ip_address"):
            parsed = CameraService.parse_camera_url(camera_data["rtsp_url"])
            camera_data.update(parsed)
            logger.info(f"📝 Parsed URL - extracted fields: {list(parsed.keys())}")

        # Filter to only valid Camera model fields
        valid_data = CameraService.filter_valid_fields(camera_data)
        logger.info(f"✅ Valid fields for Camera model: {list(valid_data.keys())}")

        # Create camera instance
        camera = Camera(**valid_data)

        db.add(camera)
        await db.commit()
        await db.refresh(camera)

        logger.info(f"✅ Camera created: {camera.name} (ID: {camera.id})")
        logger.info(f"   IP: {camera.ip_address}")
        logger.info(f"   Manufacturer: {camera.manufacturer}")
        logger.info(f"   Channel: {camera.channel}, Subtype: {camera.subtype}")

        return camera

    @staticmethod
    async def update_camera(
        db: AsyncSession, camera_id: str, camera_data: dict
    ) -> Optional[Camera]:
        """
        Update camera with new data
        If rtsp_url is updated, also update component fields
        """
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        camera = result.scalar_one_or_none()

        if not camera:
            return None

        # If rtsp_url is being updated, parse it
        if "rtsp_url" in camera_data and camera_data["rtsp_url"]:
            parsed = CameraService.parse_camera_url(camera_data["rtsp_url"])
            camera_data.update(parsed)

        # Filter to only valid fields
        valid_data = CameraService.filter_valid_fields(camera_data)

        # Update fields
        for key, value in valid_data.items():
            if hasattr(camera, key):
                setattr(camera, key, value)

        await db.commit()
        await db.refresh(camera)

        logger.info(f"✅ Camera updated: {camera.name} ({camera.ip_address})")
        return camera

    @staticmethod
    async def get_camera(db: AsyncSession, camera_id: str) -> Optional[Camera]:
        """Get camera by ID"""
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all_cameras(db: AsyncSession) -> List[Camera]:
        """Get all cameras"""
        result = await db.execute(select(Camera))
        return result.scalars().all()

    @staticmethod
    async def delete_camera(db: AsyncSession, camera_id: str) -> bool:
        """Delete camera"""
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        camera = result.scalar_one_or_none()

        if not camera:
            return False

        await db.delete(camera)
        await db.commit()

        logger.info(f"✅ Camera deleted: {camera.name}")
        return True

    @staticmethod
    def build_rtsp_url(camera: Camera) -> str:
        """
        Build RTSP URL from camera connection fields
        This is the reverse of parse_camera_url
        """
        if camera.manufacturer in ["dahua", "amcrest"]:
            return f"rtsp://{camera.username}:{camera.password}@{camera.ip_address}:{camera.rtsp_port}/cam/realmonitor?channel={camera.channel}&subtype={camera.subtype}"

        elif camera.manufacturer == "hikvision":
            stream_id = "101" if camera.subtype == "0" else "102"
            return f"rtsp://{camera.username}:{camera.password}@{camera.ip_address}:{camera.rtsp_port}/Streaming/Channels/{stream_id}"

        elif camera.stream_path:
            return f"rtsp://{camera.username}:{camera.password}@{camera.ip_address}:{camera.rtsp_port}/{camera.stream_path}"

        else:
            # Fallback to stored rtsp_url
            return camera.rtsp_url or ""

    @staticmethod
    def build_onvif_url(camera: Camera) -> str:
        """Build ONVIF URL from camera connection fields"""
        return f"onvif://{camera.username}:{camera.password}@{camera.ip_address}:{camera.onvif_port}/onvif/device_service"

    @staticmethod
    def build_http_url(camera: Camera, endpoint: str = "snapshot") -> str:
        """
        Build HTTP URL for camera (for snapshots, MJPEG, etc.)
        """
        if camera.manufacturer in ["dahua", "amcrest"]:
            if endpoint == "snapshot":
                return f"http://{camera.username}:{camera.password}@{camera.ip_address}/cgi-bin/snapshot.cgi?channel={camera.channel}"
            elif endpoint == "mjpeg":
                return f"http://{camera.username}:{camera.password}@{camera.ip_address}/cgi-bin/mjpg/video.cgi?channel={camera.channel}&subtype={camera.subtype}"

        elif camera.manufacturer == "hikvision":
            if endpoint == "snapshot":
                return f"http://{camera.username}:{camera.password}@{camera.ip_address}/ISAPI/Streaming/channels/{camera.channel}01/picture"

        # Generic fallback
        return (
            f"http://{camera.username}:{camera.password}@{camera.ip_address}/{endpoint}"
        )

    @staticmethod
    def _detect_models_from_classes(selected_classes: List[str]) -> List[str]:
        """Auto-detect required models based on selected classes"""
        # Import model definitions
        from app.config import settings

        # Map classes to models (this should match your MODEL_DEFINITIONS)
        class_to_model = {
            "Hardhat": "ppe_detection",
            "Mask": "ppe_detection",
            "NO-Hardhat": "ppe_detection",
            "NO-Mask": "ppe_detection",
            "NO-Safety Vest": "ppe_detection",
            "Person": "ppe_detection",
            "Safety Cone": "ppe_detection",
            "Safety Vest": "ppe_detection",
            "Machinery": "ppe_detection",
            "General": "ppe_detection",
            "no_mask": "face_detection",
            "mask": "face_detection",
            "no_cap": "cap_detection",
            "cap": "cap_detection",
            "pistol": "weapon_detection",
            "knife": "weapon_detection",
            "person": "general_detection",
            "bicycle": "general_detection",
            "car": "general_detection",
            "motorcycle": "general_detection",
            "smoke": "fire_detection",
            "fire": "fire_detection",
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
    async def update_features(
        db: AsyncSession, camera_id: str, features: FeatureConfiguration
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
    async def update_detection_classes(
        db: AsyncSession, camera_id: str, detection_classes: List[str]
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
        camera.active_models = CameraService._detect_models_from_classes(
            detection_classes
        )
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
        db: AsyncSession, camera_id: str, calibration_data: CameraCalibration
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

            logger.info(
                f"✅ Calibrated camera {camera_id}: {pixels_per_meter:.2f} px/m"
            )
            return camera
        else:
            logger.error(f"❌ Calibration failed for camera {camera_id}")
            return None


camera_service = CameraService()
