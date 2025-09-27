import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
from collections import deque
from ..core.calibration import CameraCalibration

@dataclass
class TrackedObject:
  track_id: int
  class_name: str
  confidence: float
  bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
  center: Tuple[float, float]
  real_world_center: Tuple[float, float]  # in meters
  velocity: Tuple[float, float] = (0.0, 0.0)  # m/s
  speed: float = 0.0  # m/s
  path: List[Tuple[float, float]] = None
  last_seen: float = 0.0
  distance_traveled: float = 0.0

  def __post_init__(self):
    if self.path is None:
      self.path = deque(maxlen=30)  # Keep last 30 positions


class ObjectTracker:
  def __init__(self, calibration: CameraCalibration, max_disappeared: int = 10, max_distance: float = 100):
    self.calibration = calibration
    self.max_disappeared = max_disappeared
    self.max_distance = max_distance
    self.next_id = 0
    self.tracked_objects: Dict[int, TrackedObject] = {}
    self.disappeared_count: Dict[int, int] = {}

  def update(self, detections: List[Dict], fps: float = 30.0) -> List[TrackedObject]:
    """
    Update tracker with new detections
    detections: List of detection dicts with keys: x1, y1, x2, y2, confidence, label
    """
    current_time = time.time()
    dt = 1.0 / fps  # Time between frames

    # Convert detections to centers
    detection_centers = []
    for detection in detections:
      x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
      center_x = (x1 + x2) / 2
      center_y = (y1 + y2) / 2
      detection_centers.append((center_x, center_y, detection))

    # If we have no existing tracked objects, create new ones
    if len(self.tracked_objects) == 0:
      for center_x, center_y, detection in detection_centers:
        self._create_new_tracked_object(detection, (center_x, center_y), current_time)
    else:
      # Calculate distance matrix between existing objects and new detections
      object_ids = list(self.tracked_objects.keys())
      distance_matrix = np.zeros((len(object_ids), len(detection_centers)))

      for i, obj_id in enumerate(object_ids):
        obj = self.tracked_objects[obj_id]
        for j, (center_x, center_y, _) in enumerate(detection_centers):
          distance = np.sqrt((obj.center[0] - center_x) ** 2 + (obj.center[1] - center_y) ** 2)
          distance_matrix[i, j] = distance

      # Hungarian algorithm (simplified version)
      used_detection_indices = set()
      used_object_indices = set()

      # Find minimum distances for assignment
      for _ in range(min(len(object_ids), len(detection_centers))):
        min_distance = np.inf
        min_i, min_j = -1, -1

        for i in range(len(object_ids)):
          if i in used_object_indices:
            continue
          for j in range(len(detection_centers)):
            if j in used_detection_indices:
              continue
            if distance_matrix[i, j] < min_distance and distance_matrix[i, j] < self.max_distance:
              min_distance = distance_matrix[i, j]
              min_i, min_j = i, j

        if min_i != -1 and min_j != -1:
          # Update existing object
          obj_id = object_ids[min_i]
          center_x, center_y, detection = detection_centers[min_j]
          self._update_tracked_object(obj_id, detection, (center_x, center_y), current_time, dt)
          used_object_indices.add(min_i)
          used_detection_indices.add(min_j)

      # Create new objects for unmatched detections
      for j, (center_x, center_y, detection) in enumerate(detection_centers):
        if j not in used_detection_indices:
          self._create_new_tracked_object(detection, (center_x, center_y), current_time)

      # Mark unmatched objects as disappeared
      for i, obj_id in enumerate(object_ids):
        if i not in used_object_indices:
          self.disappeared_count[obj_id] = self.disappeared_count.get(obj_id, 0) + 1

    # Remove objects that have disappeared for too long
    to_remove = []
    for obj_id, count in self.disappeared_count.items():
      if count > self.max_disappeared:
        to_remove.append(obj_id)

    for obj_id in to_remove:
      del self.tracked_objects[obj_id]
      del self.disappeared_count[obj_id]

    return list(self.tracked_objects.values())

  def _create_new_tracked_object(self, detection: Dict, center: Tuple[float, float], current_time: float):
    """Create a new tracked object"""
    real_world_center = self.calibration.pixel_to_meters(*center)

    tracked_obj = TrackedObject(
      track_id=self.next_id,
      class_name=detection['label'],
      confidence=detection['confidence'],
      bbox=(detection['x1'], detection['y1'], detection['x2'], detection['y2']),
      center=center,
      real_world_center=real_world_center,
      last_seen=current_time,
      path=deque(maxlen=30)
    )
    tracked_obj.path.append(real_world_center)

    self.tracked_objects[self.next_id] = tracked_obj
    self.next_id += 1

  def _update_tracked_object(self, obj_id: int, detection: Dict, center: Tuple[float, float], current_time: float,
                             dt: float):
    """Update an existing tracked object"""
    obj = self.tracked_objects[obj_id]

    # Update basic properties
    obj.class_name = detection['label']
    obj.confidence = detection['confidence']
    obj.bbox = (detection['x1'], detection['y1'], detection['x2'], detection['y2'])

    # Calculate velocity and speed
    new_real_world_center = self.calibration.pixel_to_meters(*center)

    if len(obj.path) > 0:
      prev_center = obj.path[-1]
      dx = new_real_world_center[0] - prev_center[0]
      dy = new_real_world_center[1] - prev_center[1]
      obj.velocity = (dx / dt, dy / dt)
      obj.speed = np.sqrt(dx ** 2 + dy ** 2) / dt
      obj.distance_traveled += np.sqrt(dx ** 2 + dy ** 2)

    # Update position and path
    obj.center = center
    obj.real_world_center = new_real_world_center
    obj.path.append(new_real_world_center)
    obj.last_seen = current_time

    # Reset disappeared count
    if obj_id in self.disappeared_count:
      del self.disappeared_count[obj_id]

  def get_tracking_info(self) -> Dict:
    """Get current tracking statistics"""
    return {
      "total_tracked": len(self.tracked_objects),
      "active_tracks": [
        {
          "track_id": obj.track_id,
          "class": obj.class_name,
          "speed_ms": round(obj.speed, 2),
          "speed_kmh": round(obj.speed * 3.6, 2),
          "distance_traveled": round(obj.distance_traveled, 2),
          "confidence": round(obj.confidence, 2)
        }
        for obj in self.tracked_objects.values()
      ]
    }
