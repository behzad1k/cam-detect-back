from typing import Dict, List, Tuple
from collections import defaultdict, deque
from .tracker_manager import TrackedObject

# app/tracking/analytics.py
class TrackingAnalytics:
  """Advanced analytics for tracked objects"""

  def __init__(self):
    self.zone_definitions = {}
    self.line_crossings = defaultdict(list)

  def define_zone(self, zone_id: str, polygon_points: List[Tuple[int, int]]):
    """Define a zone for zone-based analytics"""
    self.zone_definitions[zone_id] = polygon_points

  def point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
      p2x, p2y = polygon[i % n]
      if y > min(p1y, p2y):
        if y <= max(p1y, p2y):
          if x <= max(p1x, p2x):
            if p1y != p2y:
              xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
              inside = not inside
      p1x, p1y = p2x, p2y

    return inside

  def check_zone_occupancy(self, tracked_objects: Dict[str, TrackedObject]) -> Dict[str, List[str]]:
    """Check which objects are in which zones"""
    zone_occupancy = {zone_id: [] for zone_id in self.zone_definitions}

    for track_id, obj in tracked_objects.items():
      for zone_id, polygon in self.zone_definitions.items():
        if self.point_in_polygon(obj.centroid, polygon):
          zone_occupancy[zone_id].append(track_id)

    return zone_occupancy

  def detect_line_crossing(self, track_id: str, trajectory: deque,
                           line_start: Tuple[int, int], line_end: Tuple[int, int]) -> bool:
    """Detect if an object has crossed a line"""
    if len(trajectory) < 2:
      return False

    # Check if the last two points cross the line
    p1 = trajectory[-2]
    p2 = trajectory[-1]

    return self._lines_intersect(p1, p2, line_start, line_end)

  def _lines_intersect(self, p1, p2, p3, p4):
    """Check if two line segments intersect"""

    def ccw(A, B, C):
      return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

  def calculate_dwell_time(self, tracked_object: TrackedObject, zone_polygon: List[Tuple[int, int]]) -> float:
    """Calculate how long an object has been in a zone"""
    frames_in_zone = 0

    for position in tracked_object.trajectory:
      if self.point_in_polygon(position, zone_polygon):
        frames_in_zone += 1

    # Convert frames to seconds (assuming 30 FPS)
    return frames_in_zone / 30.0

analytics = TrackingAnalytics()