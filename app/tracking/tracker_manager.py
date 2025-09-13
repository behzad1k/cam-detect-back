import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import uuid
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class TrackerType(Enum):
  CENTROID = "centroid"
  KALMAN = "kalman"
  DEEP_SORT = "deep_sort"
  BYTE_TRACK = "byte_track"


@dataclass
class Detection:
  x1: float
  y1: float
  x2: float
  y2: float
  confidence: float
  class_id: int
  class_name: str


@dataclass
class TrackedObject:
  track_id: str
  class_id: int
  class_name: str
  bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
  confidence: float
  centroid: Tuple[float, float]
  velocity: Tuple[float, float]
  age: int
  hits: int
  time_since_update: int
  last_seen: float
  trajectory: deque
  state_history: deque
  metadata: Dict[str, Any]


class CentroidTracker:
  """Centroid-based object tracker with Kalman filtering"""

  def __init__(self, max_disappeared=30, max_distance=100, use_kalman=True):
    self.next_id = 0
    self.objects = {}
    self.disappeared = {}
    self.max_disappeared = max_disappeared
    self.max_distance = max_distance
    self.use_kalman = use_kalman

    # Kalman filters for each tracked object
    self.kalman_filters = {}

  def _create_kalman_filter(self):
    """Create a Kalman filter for position and velocity tracking"""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
    return kf

  def register(self, centroid, detection: Detection):
    """Register a new object"""
    track_id = str(uuid.uuid4())[:8]

    tracked_obj = TrackedObject(
      track_id=track_id,
      class_id=detection.class_id,
      class_name=detection.class_name,
      bbox=(detection.x1, detection.y1, detection.x2, detection.y2),
      confidence=detection.confidence,
      centroid=centroid,
      velocity=(0, 0),
      age=1,
      hits=1,
      time_since_update=0,
      last_seen=time.time(),
      trajectory=deque(maxlen=100),
      state_history=deque(maxlen=50),
      metadata={}
    )

    tracked_obj.trajectory.append(centroid)
    self.objects[track_id] = tracked_obj
    self.disappeared[track_id] = 0

    if self.use_kalman:
      kf = self._create_kalman_filter()
      kf.statePre = np.array([centroid[0], centroid[1], 0, 0], dtype=np.float32)
      self.kalman_filters[track_id] = kf

    return track_id

  def deregister(self, track_id):
    """Remove an object from tracking"""
    if track_id in self.objects:
      del self.objects[track_id]
      del self.disappeared[track_id]
      if track_id in self.kalman_filters:
        del self.kalman_filters[track_id]

  def update(self, detections: List[Detection]) -> Dict[str, TrackedObject]:
    """Update tracker with new detections"""
    if len(detections) == 0:
      # Mark all existing objects as disappeared
      for track_id in list(self.disappeared.keys()):
        self.disappeared[track_id] += 1
        self.objects[track_id].time_since_update += 1

        if self.disappeared[track_id] > self.max_disappeared:
          self.deregister(track_id)
      return self.objects

    # Calculate centroids
    input_centroids = []
    for det in detections:
      cx = int((det.x1 + det.x2) / 2.0)
      cy = int((det.y1 + det.y2) / 2.0)
      input_centroids.append((cx, cy))

    if len(self.objects) == 0:
      # Register all detections as new objects
      for i, detection in enumerate(detections):
        self.register(input_centroids[i], detection)
    else:
      # Match existing objects with new detections
      object_centroids = [obj.centroid for obj in self.objects.values()]
      object_ids = list(self.objects.keys())

      # Calculate distance matrix
      D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] -
                         np.array(input_centroids), axis=2)

      # Find minimum values and sort by distance
      rows = D.min(axis=1).argsort()
      cols = D.argmin(axis=1)[rows]

      used_row_indices = set()
      used_col_indices = set()

      # Update existing objects
      for (row, col) in zip(rows, cols):
        if row in used_row_indices or col in used_col_indices:
          continue

        if D[row, col] > self.max_distance:
          continue

        track_id = object_ids[row]
        detection = detections[col]
        new_centroid = input_centroids[col]

        # Update object
        old_centroid = self.objects[track_id].centroid
        velocity = (new_centroid[0] - old_centroid[0],
                    new_centroid[1] - old_centroid[1])

        self.objects[track_id].centroid = new_centroid
        self.objects[track_id].velocity = velocity
        self.objects[track_id].bbox = (detection.x1, detection.y1,
                                       detection.x2, detection.y2)
        self.objects[track_id].confidence = detection.confidence
        self.objects[track_id].hits += 1
        self.objects[track_id].time_since_update = 0
        self.objects[track_id].last_seen = time.time()
        self.objects[track_id].trajectory.append(new_centroid)

        # Update Kalman filter
        if self.use_kalman and track_id in self.kalman_filters:
          kf = self.kalman_filters[track_id]
          measurement = np.array([[np.float32(new_centroid[0])],
                                  [np.float32(new_centroid[1])]])
          kf.correct(measurement)
          prediction = kf.predict()
          predicted_centroid = (int(prediction[0]), int(prediction[1]))
          self.objects[track_id].metadata['predicted_position'] = predicted_centroid

        self.disappeared[track_id] = 0
        used_row_indices.add(row)
        used_col_indices.add(col)

      # Handle unmatched detections and objects
      unused_row_indices = set(range(0, D.shape[0])) - used_row_indices
      unused_col_indices = set(range(0, D.shape[1])) - used_col_indices

      if D.shape[0] >= D.shape[1]:
        # More objects than detections
        for row in unused_row_indices:
          track_id = object_ids[row]
          self.disappeared[track_id] += 1
          self.objects[track_id].time_since_update += 1

          if self.disappeared[track_id] > self.max_disappeared:
            self.deregister(track_id)
      else:
        # More detections than objects
        for col in unused_col_indices:
          self.register(input_centroids[col], detections[col])

    return self.objects


class TrackerManager:
  """Main tracker manager with multiple tracking algorithms"""

  def __init__(self):
    self.trackers: Dict[str, Any] = {}
    self.tracker_configs = {}
    self.active_streams = set()

  def create_tracker(self, stream_id: str, tracker_type: TrackerType, **kwargs) -> bool:
    """Create a new tracker for a stream"""
    try:
      if tracker_type == TrackerType.CENTROID:
        tracker = CentroidTracker(
          max_disappeared=kwargs.get('max_disappeared', 30),
          max_distance=kwargs.get('max_distance', 100),
          use_kalman=kwargs.get('use_kalman', True)
        )
      elif tracker_type == TrackerType.KALMAN:
        tracker = CentroidTracker(
          max_disappeared=kwargs.get('max_disappeared', 30),
          max_distance=kwargs.get('max_distance', 100),
          use_kalman=True
        )
      else:
        # Placeholder for other tracker types
        tracker = CentroidTracker(**kwargs)

      self.trackers[stream_id] = tracker
      self.tracker_configs[stream_id] = {
        'type': tracker_type,
        'config': kwargs
      }
      self.active_streams.add(stream_id)
      return True

    except Exception as e:
      print(f"Error creating tracker: {e}")
      return False

  def update_tracker(self, stream_id: str, detections: List[Detection]) -> Dict[str, TrackedObject]:
    """Update tracker with new detections"""
    if stream_id not in self.trackers:
      # Create default tracker if none exists
      self.create_tracker(stream_id, TrackerType.CENTROID)

    return self.trackers[stream_id].update(detections)

  def get_tracked_objects(self, stream_id: str) -> Dict[str, TrackedObject]:
    """Get all tracked objects for a stream"""
    if stream_id in self.trackers:
      return self.trackers[stream_id].objects
    return {}

  def remove_tracker(self, stream_id: str):
    """Remove tracker for a stream"""
    if stream_id in self.trackers:
      del self.trackers[stream_id]
      del self.tracker_configs[stream_id]
      self.active_streams.discard(stream_id)

  def configure_tracker(self, stream_id: str, **kwargs) -> bool:
    """Update tracker configuration"""
    if stream_id in self.trackers:
      tracker_type = self.tracker_configs[stream_id]['type']
      self.remove_tracker(stream_id)
      return self.create_tracker(stream_id, tracker_type, **kwargs)
    return False

  def get_tracker_stats(self, stream_id: str) -> Dict[str, Any]:
    """Get tracker statistics"""
    if stream_id not in self.trackers:
      return {}

    tracker = self.trackers[stream_id]
    objects = tracker.objects

    stats = {
      'total_objects': len(objects),
      'active_tracks': len([obj for obj in objects.values() if obj.time_since_update < 5]),
      'class_distribution': defaultdict(int),
      'average_confidence': 0,
      'tracker_type': self.tracker_configs[stream_id]['type'].value,
      'config': self.tracker_configs[stream_id]['config']
    }

    if objects:
      for obj in objects.values():
        stats['class_distribution'][obj.class_name] += 1
      stats['average_confidence'] = np.mean([obj.confidence for obj in objects.values()])

    return stats


# Global tracker manager instance
tracker_manager = TrackerManager()
