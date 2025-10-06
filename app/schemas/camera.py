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
  features: Dict[str, bool] = Field(
    default_factory=lambda: {
      "detection": True,
      "tracking": False,
      "speed": False,
      "counting": False
    }
  )
  active_models: List[str] = Field(default_factory=list)


class CameraUpdate(BaseModel):
  name: Optional[str] = None
  location: Optional[str] = None
  rtsp_url: Optional[str] = None
  features: Optional[Dict[str, bool]] = None
  active_models: Optional[List[str]] = None
  is_active: Optional[bool] = None


class CameraCalibration(BaseModel):
  mode: str  # "reference_object" or "perspective_transform"
  points: List[Dict[str, float]]  # [{"pixel_x": x, "pixel_y": y, "real_x": x, "real_y": y}]
  reference_width_meters: Optional[float] = None
  reference_height_meters: Optional[float] = None


class CameraResponse(CameraBase):
  id: str
  is_calibrated: bool
  pixels_per_meter: Optional[float]
  calibration_mode: Optional[str]
  features: Dict[str, bool]
  active_models: List[str]
  created_at: datetime
  is_active: bool

  class Config:
    from_attributes = True