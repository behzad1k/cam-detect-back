import cv2
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
from dataclasses import dataclass, field
import time
import uuid

from app.schemas.detection import Detection


@dataclass
class TrackedObject:
  track_id: str
  class_id: int
  class_name: str
  bbox: Tuple[float, float, float, float]
  confidence: float
  centroid: Tuple[float, float]
  velocity: Tuple[float, float] = (0.0, 0.0)
  age: int = 1
  hits: int = 1
  time_since_update: int = 0
  last_seen: float = field(default_factory=time.time)
  trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
  distance_traveled: float = 0.0


class ObjectTracker:
  """Centroid-based object tracker with Kalman filtering"""

  def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
    self.max_disappeared = max_disappeared
    self.max_distance = max_distance
    self.objects: Dict[str, TrackedObject] = {}
    self.disappeared: Dict[str, int] = {}
    self.kalman_filters: Dict[str, cv2.KalmanFilter] = {}

  def _create_kalman_filter(self) -> cv2.KalmanFilter:
    """Create Kalman filter for position and velocity tracking"""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
    return kf

  def register(self, centroid: Tuple[float, float], detection: Detection) -> str:
    """Register a new object"""
    track_id = str(uuid.uuid4())[:8]

    obj = TrackedObject(
      track_id=track_id,
      class_id=detection.class_id,
      class_name=detection.label,
      bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
      confidence=detection.confidence,
      centroid=centroid
    )
    obj.trajectory.append(centroid)

    self.objects[track_id] = obj
    self.disappeared[track_id] = 0

    # Initialize Kalman filter
    kf = self._create_kalman_filter()
    kf.statePre = np.array([centroid[0], centroid[1], 0, 0], dtype=np.float32)
    self.kalman_filters[track_id] = kf

    return track_id

  def deregister(self, track_id: str):
    """Remove an object"""
    self.objects.pop(track_id, None)
    self.disappeared.pop(track_id, None)
    self.kalman_filters.pop(track_id, None)

  def update(self, detections: List[Detection]) -> Dict[str, TrackedObject]:
    """Update tracker with new detections"""

    # Mark all as disappeared if no detections
    if not detections:
      for track_id in list(self.disappeared.keys()):
        self.disappeared[track_id] += 1
        self.objects[track_id].time_since_update += 1
        self.objects[track_id].age += 1  # FIX: Increment age

        if self.disappeared[track_id] > self.max_disappeared:
          self.deregister(track_id)
      return self.objects

    # Calculate centroids
    input_centroids = []
    for det in detections:
      cx = (det.x1 + det.x2) / 2.0
      cy = (det.y1 + det.y2) / 2.0
      input_centroids.append((cx, cy))

    # Register all if no existing objects
    if not self.objects:
      for i, det in enumerate(detections):
        self.register(input_centroids[i], det)
      return self.objects

    # Match existing objects with new detections
    object_ids = list(self.objects.keys())
    object_centroids = [self.objects[oid].centroid for oid in object_ids]

    # Calculate distance matrix
    D = np.linalg.norm(
      np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
      axis=2
    )

    # Hungarian assignment
    rows = D.min(axis=1).argsort()
    cols = D.argmin(axis=1)[rows]

    used_rows = set()
    used_cols = set()

    # Update matched objects
    for row, col in zip(rows, cols):
      if row in used_rows or col in used_cols:
        continue
      if D[row, col] > self.max_distance:
        continue

      track_id = object_ids[row]
      det = detections[col]
      new_centroid = input_centroids[col]

      # Calculate velocity
      old_centroid = self.objects[track_id].centroid
      velocity = (
        new_centroid[0] - old_centroid[0],
        new_centroid[1] - old_centroid[1]
      )

      # Calculate distance traveled
      distance = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

      # Update object
      self.objects[track_id].centroid = new_centroid
      self.objects[track_id].velocity = velocity
      self.objects[track_id].bbox = (det.x1, det.y1, det.x2, det.y2)
      self.objects[track_id].confidence = det.confidence
      self.objects[track_id].hits += 1
      self.objects[track_id].time_since_update = 0
      self.objects[track_id].last_seen = time.time()
      self.objects[track_id].trajectory.append(new_centroid)
      self.objects[track_id].distance_traveled += distance
      self.objects[track_id].age += 1  # FIX: Increment age on update

      # Update Kalman filter
      kf = self.kalman_filters[track_id]
      measurement = np.array([[np.float32(new_centroid[0])], [np.float32(new_centroid[1])]])
      kf.correct(measurement)
      kf.predict()

      self.disappeared[track_id] = 0
      used_rows.add(row)
      used_cols.add(col)

    # Handle unmatched objects
    unused_rows = set(range(D.shape[0])) - used_rows
    for row in unused_rows:
      track_id = object_ids[row]
      self.disappeared[track_id] += 1
      self.objects[track_id].time_since_update += 1
      self.objects[track_id].age += 1  # FIX: Increment age when disappeared

      if self.disappeared[track_id] > self.max_disappeared:
        self.deregister(track_id)

    # Register new objects
    unused_cols = set(range(D.shape[1])) - used_cols
    for col in unused_cols:
      self.register(input_centroids[col], detections[col])

    return self.objects
