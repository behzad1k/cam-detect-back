# app/schemas/camera.py

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CameraBase(BaseModel):
    """Base camera schema with common fields"""

    name: str = Field(..., description="Camera display name", example="Entrance Camera")
    location: Optional[str] = Field(
        None, description="Camera location description", example="Main Entrance"
    )
    rtsp_url: Optional[str] = Field(
        None,
        description="RTSP/HTTP stream URL or webcam://0 for local webcam",
        example="rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
    )
    width: int = Field(640, description="Frame width in pixels", example=1920)
    height: int = Field(480, description="Frame height in pixels", example=1080)
    fps: int = Field(15, description="Frames per second", example=30)


class CalibrationPoint(BaseModel):
    """Single calibration point with pixel and real-world coordinates"""

    pixel_x: float = Field(..., description="X coordinate in pixels", example=100.0)
    pixel_y: float = Field(..., description="Y coordinate in pixels", example=200.0)
    real_x: float = Field(
        ..., description="Real-world X coordinate in meters", example=0.0
    )
    real_y: float = Field(
        ..., description="Real-world Y coordinate in meters", example=0.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "pixel_x": 100.0,
                "pixel_y": 200.0,
                "real_x": 0.0,
                "real_y": 0.0,
            }
        }


class CameraCalibration(BaseModel):
    """Camera calibration configuration"""

    mode: str = Field(
        ...,
        description="Calibration mode: 'reference_object' or 'perspective_transform'",
        example="reference_object",
    )
    points: List[CalibrationPoint] = Field(
        ...,
        description="Calibration points (min 2 for reference_object, 4 for perspective_transform)",
    )
    reference_width_meters: Optional[float] = Field(
        None, description="Known width for reference object in meters", example=2.0
    )
    reference_height_meters: Optional[float] = Field(
        None, description="Known height for reference object in meters", example=None
    )

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "reference_object",
                "points": [
                    {"pixel_x": 100, "pixel_y": 200, "real_x": 0, "real_y": 0},
                    {"pixel_x": 350, "pixel_y": 200, "real_x": 2, "real_y": 0},
                ],
                "reference_width_meters": 2.0,
                "reference_height_meters": None,
            }
        }


class FeatureConfiguration(BaseModel):
    """Camera feature configuration"""

    detection: Optional[bool] = Field(
        None, description="Enable/disable object detection", example=True
    )
    tracking: Optional[bool] = Field(
        None, description="Enable/disable object tracking", example=True
    )
    speed: Optional[bool] = Field(
        None,
        description="Enable/disable speed calculation (requires calibration)",
        example=True,
    )
    distance: Optional[bool] = Field(
        None,
        description="Enable/disable distance calculation (requires calibration)",
        example=True,
    )
    counting: Optional[bool] = Field(
        None, description="Enable/disable object counting", example=False
    )
    class_filters: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Per-model class filters (deprecated, use detection_classes)",
        example={"general_detection": ["person", "car"], "face_detection": ["mask"]},
    )
    tracking_classes: Optional[List[str]] = Field(
        None,
        description="Which classes to track (empty = track all detected)",
        example=["person", "car"],
    )
    speed_classes: Optional[List[str]] = Field(
        None,
        description="Which classes to calculate speed for (empty = all tracked)",
        example=["car", "motorcycle"],
    )
    distance_classes: Optional[List[str]] = Field(
        None,
        description="Which classes to calculate distance for (empty = all tracked)",
        example=["person"],
    )
    detection_classes: Optional[List[str]] = Field(
        None,
        description="Which classes to detect across all models (empty = all classes)",
        example=["person", "car", "motorcycle", "mask", "no_mask"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detection": True,
                "tracking": True,
                "speed": True,
                "distance": True,
                "counting": False,
                "detection_classes": ["person", "car", "motorcycle"],
                "tracking_classes": ["person", "car"],
                "speed_classes": ["car", "motorcycle"],
                "distance_classes": ["person"],
            }
        }


class CameraCreate(CameraBase):
    """Schema for creating a new camera"""

    features: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "detection": True,
            "tracking": False,
            "speed": False,
            "distance": False,
            "counting": False,
            "class_filters": {},
            "tracking_classes": [],
            "speed_classes": [],
            "distance_classes": [],
            "detection_classes": [],
        },
        description="Feature configuration for the camera",
    )
    active_models: Optional[List[str]] = Field(
        default_factory=list,
        description="Detection models to use (auto-determined from detection_classes if not specified)",
        example=["general_detection", "face_detection"],
    )
    selected_classes: Optional[List[str]] = Field(
        default_factory=list,
        description="Selected classes (frontend compatibility)",
        example=["person", "car"],
    )
    calibration: Optional[CameraCalibration] = Field(
        None, description="Optional calibration data for the camera"
    )
    alert_email: Optional[str] = Field(
        None,
        description="Email address for alert notifications",
        example="alerts@company.com",
    )

    # Connection settings (auto-parsed from rtsp_url if not provided)
    protocol: Optional[str] = Field(
        None, description="Protocol: rtsp, http, onvif", example="rtsp"
    )
    ipAddress: Optional[str] = Field(
        None, description="Camera IP address", example="192.168.1.12"
    )
    port: Optional[str] = Field(None, description="Connection port", example="554")

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "name": "Entrance Camera",
                "location": "Main Entrance",
                "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "features": {
                    "detection": True,
                    "tracking": True,
                    "speed": True,
                    "distance": False,
                    "counting": False,
                    "detection_classes": ["person", "car", "motorcycle"],
                    "tracking_classes": ["person"],
                    "speed_classes": ["car", "motorcycle"],
                    "distance_classes": [],
                },
                "active_models": ["general_detection"],
                "alert_email": "alerts@company.com",
            }
        }


class CameraUpdate(BaseModel):
    """Schema for updating a camera (all fields optional)"""

    name: Optional[str] = Field(
        None, description="Camera display name", example="Updated Camera Name"
    )
    location: Optional[str] = Field(
        None, description="Camera location", example="New Location"
    )
    rtsp_url: Optional[str] = Field(
        None, description="Stream URL", example="rtsp://..."
    )
    features: Optional[Dict[str, Any]] = Field(
        None, description="Feature configuration"
    )
    active_models: Optional[List[str]] = Field(
        None, description="Active detection models"
    )
    is_active: Optional[bool] = Field(
        None, description="Camera active status", example=True
    )
    alert_email: Optional[str] = Field(None, description="Alert email address")
    alert_config: Optional[Dict[str, Any]] = Field(
        None, description="Alert configuration"
    )

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "name": "Updated Camera Name",
                "location": "New Location",
                "features": {"detection": True, "tracking": True, "speed": True},
                "is_active": True,
            }
        }


class CameraResponse(CameraBase):
    """Schema for camera response"""

    id: str = Field(
        ...,
        description="Unique camera identifier",
        example="f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f",
    )

    # Connection details (parsed from rtsp_url)
    ip_address: Optional[str] = Field(
        None, description="Camera IP address", example="192.168.1.12"
    )
    username: Optional[str] = Field(
        None, description="Camera username", example="admin"
    )
    rtsp_port: Optional[int] = Field(None, description="RTSP port", example=554)
    onvif_port: Optional[int] = Field(None, description="ONVIF port", example=80)

    # Calibration
    is_calibrated: bool = Field(
        ..., description="Whether camera is calibrated", example=True
    )
    pixels_per_meter: Optional[float] = Field(
        None, description="Calibration ratio", example=125.5
    )
    calibration_mode: Optional[str] = Field(
        None, description="Calibration method used", example="reference_object"
    )
    calibration_points: Optional[List[Dict[str, float]]] = Field(
        None, description="Calibration points"
    )

    # Features and models
    features: Dict[str, Any] = Field(..., description="Enabled features")
    active_models: List[str] = Field(
        ..., description="Active detection models", example=["general_detection"]
    )

    # Alerts
    alert_email: Optional[str] = Field(
        None, description="Email for alerts", example="alerts@company.com"
    )
    alert_config: Optional[Dict[str, Any]] = Field(
        None, description="Alert configuration"
    )

    # Metadata
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    is_active: bool = Field(..., description="Camera active status", example=True)

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "f8e7d6c5-b4a3-4291-8e7f-1a2b3c4d5e6f",
                "name": "Entrance Camera",
                "location": "Main Entrance",
                "rtsp_url": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
                "ip_address": "192.168.1.12",
                "username": "admin",
                "rtsp_port": 554,
                "onvif_port": 80,
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "is_calibrated": True,
                "pixels_per_meter": 125.5,
                "calibration_mode": "reference_object",
                "calibration_points": [
                    {"pixel_x": 100, "pixel_y": 200, "real_x": 0, "real_y": 0},
                    {"pixel_x": 350, "pixel_y": 200, "real_x": 2, "real_y": 0},
                ],
                "features": {
                    "detection": True,
                    "tracking": True,
                    "speed": True,
                    "distance": True,
                    "detection_classes": ["person", "car"],
                },
                "active_models": ["general_detection"],
                "alert_email": "alerts@company.com",
                "alert_config": {
                    "speed_alerts": [],
                    "tracking_alerts": [],
                    "distance_alerts": [],
                    "email_enabled": True,
                    "cooldown_seconds": 60,
                },
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T13:00:00Z",
                "is_active": True,
            }
        }
