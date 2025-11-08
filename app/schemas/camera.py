from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class CameraBase(BaseModel):
  name: str
  location: Optional[str] = None
  rtsp_url: Optional[str] = None
  width: int = 640
  height: int = 480
  fps: int = 30


class CameraCreate(CameraBase):
  features: Dict[str, Any] = Field(
    default_factory=lambda: {
      "detection": True,
      "tracking": False,
      "speed": False,
      "distance": False,
      "counting": False,
      # Class filters per model
      "class_filters": {},
      # Classes to track
      "tracking_classes": [],
      # Classes to calculate speed for
      "speed_classes": [],
      # Classes to calculate distance for
      "distance_classes": []
    }
  )
  active_models: List[str] = Field(default_factory=list)

  # Optional calibration during creation
  calibration: Optional['CameraCalibration'] = None


class CameraUpdate(BaseModel):
  name: Optional[str] = None
  location: Optional[str] = None
  rtsp_url: Optional[str] = None
  features: Optional[Dict[str, Any]] = None
  active_models: Optional[List[str]] = None
  is_active: Optional[bool] = None


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
  """Model for updating camera feature configuration"""
  detection: Optional[bool] = None
  tracking: Optional[bool] = None
  speed: Optional[bool] = None
  distance: Optional[bool] = None
  counting: Optional[bool] = None

  # Model-specific class filters
  class_filters: Optional[Dict[str, List[str]]] = None

  # Feature-specific class selections
  tracking_classes: Optional[List[str]] = None
  speed_classes: Optional[List[str]] = None
  distance_classes: Optional[List[str]] = None