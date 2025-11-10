from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON, Integer
from sqlalchemy.sql import func
from app.database.base import Base
import uuid


class Camera(Base):
  __tablename__ = "cameras"

  id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
  name = Column(String(64), nullable=False)
  location = Column(String(64), nullable=True)
  rtsp_url = Column(String(255), nullable=True)

  # Resolution
  width = Column(Integer, default=640)
  height = Column(Integer, default=480)
  fps = Column(Integer, default=30)

  # Calibration
  is_calibrated = Column(Boolean, default=False)
  pixels_per_meter = Column(Float, nullable=True)
  calibration_mode = Column(String(64), nullable=True)  # reference_object, perspective_transform
  calibration_points = Column(JSON, nullable=True)

  # Features enabled
  features = Column(JSON, default=dict)  # {"detection": true, "tracking": true, "speed": true}

  # Models to use
  active_models = Column(JSON, default=list)  # ["ppe_detection", "fire_detection"]

  # Metadata
  created_at = Column(DateTime(timezone=True), server_default=func.now())
  updated_at = Column(DateTime(timezone=True), onupdate=func.now())
  is_active = Column(Boolean, default=True)

  def __repr__(self):
    return f"<Camera {self.name} ({self.id})>"