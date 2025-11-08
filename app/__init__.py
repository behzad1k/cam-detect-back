# app/__init__.py
"""SeeDeep.AI - Multi-Camera Object Detection & Tracking System"""

__version__ = "2.0.0"

# app/database/__init__.py
from app.database.base import Base, get_db, init_db
from app.database.models import Camera

__all__ = ["Base", "get_db", "init_db", "Camera"]

# app/schemas/__init__.py
from app.schemas.camera import CameraCreate, CameraUpdate, CameraResponse, CameraCalibration
from app.schemas.detection import Detection, ModelResult, DetectionResponse
from app.schemas.tracking import TrackedObject, TrackingResponse, ZoneDefinition, TrackingConfig

__all__ = [
    "CameraCreate", "CameraUpdate", "CameraResponse", "CameraCalibration",
    "Detection", "ModelResult", "DetectionResponse",
    "TrackedObject", "TrackingResponse", "ZoneDefinition", "TrackingConfig"
]

# app/services/__init__.py
from app.services.camera_service import camera_service
from app.services.stream_manager import stream_manager

__all__ = ["camera_service", "stream_manager"]

# app/core/__init__.py
"""Core processing modules"""

# app/core/detection/__init__.py
from app.core.detection.yolo_detector import detector

__all__ = ["detector"]

# app/core/tracking/__init__.py
from app.core.tracking.tracker import ObjectTracker
from app.core.tracking.speed_calculator import speed_calculator

__all__ = ["ObjectTracker", "speed_calculator"]

# app/core/calibration/__init__.py
from app.core.calibration.calibrator import calibrator

__all__ = ["calibrator"]

# app/api/__init__.py
"""API routes"""

# app/utils/__init__.py
"""Utility functions"""