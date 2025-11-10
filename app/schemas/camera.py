# app/schemas/camera.py - COMPLETE REPLACEMENT

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class CameraBase(BaseModel):
  name: str
  location: Optional[str] = None
  rtsp_url: Optional[str] = None  # Make sure this is properly defined
  width: int = 640
  height: int = 480
  fps: int = 15  # Default to 15 FPS


class CalibrationPoint(BaseModel):
  pixel_x: float
  pixel_y: float
  real_x: float
  real_y: float


class CameraCalibration(BaseModel):
  mode: str  # "reference_object" or "perspective_transform"
  points: List[CalibrationPoint]
  reference_width_meters: Optional[float] = None
  reference_height_meters: Optional[float] = None


class CameraCreate(CameraBase):
  # Feature configuration
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
      "detection_classes": []  # NEW: Classes to detect
    }
  )
  # Models will be auto-detected from selected classes
  active_models: Optional[List[str]] = Field(default_factory=list)

  # Selected classes (frontend sends this)
  selected_classes: Optional[List[str]] = Field(default_factory=list)

  # Calibration data
  calibration: Optional[CameraCalibration] = None

  # Connection settings
  protocol: Optional[str] = "rtsp"
  ipAddress: Optional[str] = None
  port: Optional[str] = "554"

  class Config:
    extra = "allow"  # Allow extra fields from frontend


class CameraUpdate(BaseModel):
  name: Optional[str] = None
  location: Optional[str] = None
  rtsp_url: Optional[str] = None
  features: Optional[Dict[str, Any]] = None
  active_models: Optional[List[str]] = None
  is_active: Optional[bool] = None

  class Config:
    extra = "allow"


class CameraResponse(CameraBase):
  id: str
  is_calibrated: bool
  pixels_per_meter: Optional[float]
  calibration_mode: Optional[str]
  features: Dict[str, Any]
  active_models: List[str]
  created_at: datetime
  is_active: bool

  class Config:
    from_attributes = True


class FeatureConfiguration(BaseModel):
  detection: Optional[bool] = None
  tracking: Optional[bool] = None
  speed: Optional[bool] = None
  distance: Optional[bool] = None
  counting: Optional[bool] = None
  class_filters: Optional[Dict[str, List[str]]] = None
  tracking_classes: Optional[List[str]] = None
  speed_classes: Optional[List[str]] = None
  distance_classes: Optional[List[str]] = None