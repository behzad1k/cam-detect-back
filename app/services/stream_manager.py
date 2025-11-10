from typing import Dict, Optional
import logging

from app.core.detection.yolo_detector import detector
from app.core.tracking.tracker import ObjectTracker
from app.core.tracking.speed_calculator import speed_calculator
from app.database.models import Camera

logger = logging.getLogger(__name__)



class CameraStream:
  """Represents an active camera stream with its processing pipeline"""

  def __init__(self, camera: Camera):
    self.camera = camera
    self.tracker: Optional[ObjectTracker] = None

    # Initialize tracker if tracking is enabled
    if camera.features.get("tracking", False):
      self.tracker = ObjectTracker()
      logger.info(f"📹 Initialized tracker for camera {camera.id}")

  def update_features(self, features: dict):
    """Update enabled features"""
    self.camera.features = features

    # Initialize or remove tracker based on features
    # CRITICAL: Don't recreate if it already exists!
    if features.get("tracking", False) and not self.tracker:
      self.tracker = ObjectTracker()
      logger.info(f"📹 Created tracker for camera {self.camera.id}")
    elif not features.get("tracking", False) and self.tracker:
      self.tracker = None
      logger.info(f"📹 Removed tracker for camera {self.camera.id}")




class StreamManager:
  """Manages multiple camera streams simultaneously"""

  def __init__(self):
    self.active_streams: Dict[str, CameraStream] = {}

  def add_stream(self, camera: Camera) -> bool:
    """Add a new camera stream"""
    try:
      if camera.id in self.active_streams:
        logger.warning(f"Stream {camera.id} already active")
        return True

      stream = CameraStream(camera)
      self.active_streams[camera.id] = stream

      logger.info(f"✅ Added stream for camera {camera.id} ({camera.name})")
      return True

    except Exception as e:
      logger.error(f"❌ Failed to add stream {camera.id}: {e}")
      return False

  def remove_stream(self, camera_id: str) -> bool:
    """Remove a camera stream"""
    if camera_id in self.active_streams:
      del self.active_streams[camera_id]
      logger.info(f"🔴 Removed stream {camera_id}")
      return True
    return False

  def get_stream(self, camera_id: str) -> Optional[CameraStream]:
    """Get a camera stream"""
    return self.active_streams.get(camera_id)

  def update_stream_features(self, camera_id: str, features: dict):
    """Update features for a stream"""
    stream = self.get_stream(camera_id)
    if stream:
      stream.update_features(features)

  def is_active(self, camera_id: str) -> bool:
    """Check if a stream is active"""
    return camera_id in self.active_streams

  def get_active_count(self) -> int:
    """Get number of active streams"""
    return len(self.active_streams)


# Global stream manager instance
stream_manager = StreamManager()