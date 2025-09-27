import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ObjectMeasurements:
  """Object measurements in real-world units"""
  width_meters: float
  height_meters: float
  area_square_meters: float
  distance_from_camera: Optional[float] = None
  center_distance_meters: Optional[float] = None


@dataclass
class DistanceMeasurement:
  """Distance measurement between two objects"""
  object1_id: str
  object2_id: str
  distance_meters: float
  object1_class: str
  object2_class: str
  measurement_type: str  # "center_to_center", "edge_to_edge", "closest_points"


class DistanceSizeCalculator:
  """Calculate distances and sizes of tracked objects"""

  def __init__(self):
    self.calibration_enabled = False
    self.pixels_per_meter = 100.0
    self.camera_height_meters = 3.0  # Default camera height
    self.camera_angle_degrees = 0.0  # Camera tilt angle

    # Known object sizes for validation (meters)
    self.reference_sizes = {
      'person': {'width': 0.6, 'height': 1.7},
      'car': {'width': 1.8, 'height': 1.5},
      'truck': {'width': 2.5, 'height': 3.0},
      'bicycle': {'width': 0.6, 'height': 1.1},
      'motorcycle': {'width': 0.8, 'height': 1.2}
    }

  def set_calibration(self, pixels_per_meter: float, camera_height: float = 3.0, camera_angle: float = 0.0):
    """Set calibration parameters"""
    self.calibration_enabled = True
    self.pixels_per_meter = pixels_per_meter
    self.camera_height_meters = camera_height
    self.camera_angle_degrees = camera_angle
    logger.info(f"Calibration set: {pixels_per_meter:.2f} px/m, height: {camera_height}m, angle: {camera_angle}°")

  def calculate_object_size(self, bbox: List[float], class_name: str = None) -> ObjectMeasurements:
    """Calculate real-world size of an object from its bounding box"""
    if not self.calibration_enabled:
      return ObjectMeasurements(0, 0, 0)

    try:
      x1, y1, x2, y2 = bbox

      # Calculate pixel dimensions
      width_pixels = abs(x2 - x1)
      height_pixels = abs(y2 - y1)

      # Convert to real-world measurements
      width_meters = width_pixels / self.pixels_per_meter
      height_meters = height_pixels / self.pixels_per_meter
      area_square_meters = width_meters * height_meters

      # Estimate distance from camera using perspective (if camera height is known)
      distance_from_camera = None
      if class_name and class_name in self.reference_sizes:
        expected_height = self.reference_sizes[class_name]['height']
        if height_meters > 0:
          # Simple perspective distance estimation
          distance_from_camera = (expected_height * self.pixels_per_meter) / height_pixels

      # Calculate distance from camera center (ground plane)
      center_x = (x1 + x2) / 2
      center_y = (y1 + y2) / 2
      center_distance_meters = self._calculate_distance_from_center(center_x, center_y)

      return ObjectMeasurements(
        width_meters=width_meters,
        height_meters=height_meters,
        area_square_meters=area_square_meters,
        distance_from_camera=distance_from_camera,
        center_distance_meters=center_distance_meters
      )

    except Exception as e:
      logger.error(f"Error calculating object size: {e}")
      return ObjectMeasurements(0, 0, 0)

  def calculate_distance_between_objects(self, obj1: Dict, obj2: Dict,
                                         measurement_type: str = "center_to_center") -> DistanceMeasurement:
    """Calculate distance between two tracked objects"""
    if not self.calibration_enabled:
      return DistanceMeasurement("", "", 0, "", "", measurement_type)

    try:
      obj1_id = str(obj1.get('track_id', ''))
      obj2_id = str(obj2.get('track_id', ''))
      obj1_class = obj1.get('class_name', 'unknown')
      obj2_class = obj2.get('class_name', 'unknown')

      distance_meters = 0

      if measurement_type == "center_to_center":
        distance_meters = self._calculate_center_to_center_distance(obj1, obj2)
      elif measurement_type == "edge_to_edge":
        distance_meters = self._calculate_edge_to_edge_distance(obj1, obj2)
      elif measurement_type == "closest_points":
        distance_meters = self._calculate_closest_points_distance(obj1, obj2)

      return DistanceMeasurement(
        object1_id=obj1_id,
        object2_id=obj2_id,
        distance_meters=distance_meters,
        object1_class=obj1_class,
        object2_class=obj2_class,
        measurement_type=measurement_type
      )

    except Exception as e:
      logger.error(f"Error calculating distance between objects: {e}")
      return DistanceMeasurement("", "", 0, "", "", measurement_type)

  def _calculate_center_to_center_distance(self, obj1: Dict, obj2: Dict) -> float:
    """Calculate distance between object centers"""
    if 'centroid' in obj1 and 'centroid' in obj2:
      x1, y1 = obj1['centroid']
      x2, y2 = obj2['centroid']
    else:
      # Calculate centroids from bboxes
      bbox1 = obj1.get('bbox', [0, 0, 0, 0])
      bbox2 = obj2.get('bbox', [0, 0, 0, 0])
      x1, y1 = (bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2
      x2, y2 = (bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2

    # Calculate pixel distance
    pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Convert to meters
    return pixel_distance / self.pixels_per_meter

  def _calculate_edge_to_edge_distance(self, obj1: Dict, obj2: Dict) -> float:
    """Calculate shortest distance between object edges"""
    bbox1 = obj1.get('bbox', [0, 0, 0, 0])
    bbox2 = obj2.get('bbox', [0, 0, 0, 0])

    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate horizontal and vertical distances
    if x1_max < x2_min:
      dx = x2_min - x1_max
    elif x2_max < x1_min:
      dx = x1_min - x2_max
    else:
      dx = 0  # Overlapping horizontally

    if y1_max < y2_min:
      dy = y2_min - y1_max
    elif y2_max < y1_min:
      dy = y1_min - y2_max
    else:
      dy = 0  # Overlapping vertically

    # Calculate euclidean distance
    pixel_distance = math.sqrt(dx ** 2 + dy ** 2)

    return pixel_distance / self.pixels_per_meter

  def _calculate_closest_points_distance(self, obj1: Dict, obj2: Dict) -> float:
    """Calculate distance between closest points of two objects"""
    bbox1 = obj1.get('bbox', [0, 0, 0, 0])
    bbox2 = obj2.get('bbox', [0, 0, 0, 0])

    # Get all corner points of both bboxes
    points1 = [
      (bbox1[0], bbox1[1]),  # top-left
      (bbox1[2], bbox1[1]),  # top-right
      (bbox1[2], bbox1[3]),  # bottom-right
      (bbox1[0], bbox1[3])  # bottom-left
    ]

    points2 = [
      (bbox2[0], bbox2[1]),
      (bbox2[2], bbox2[1]),
      (bbox2[2], bbox2[3]),
      (bbox2[0], bbox2[3])
    ]

    # Find minimum distance between any two points
    min_distance = float('inf')
    for p1 in points1:
      for p2 in points2:
        distance = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        min_distance = min(min_distance, distance)

    return min_distance / self.pixels_per_meter

  def _calculate_distance_from_center(self, x: float, y: float, frame_width: int = 640,
                                      frame_height: int = 480) -> float:
    """Calculate distance from image center"""
    center_x = frame_width / 2
    center_y = frame_height / 2

    pixel_distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return pixel_distance / self.pixels_per_meter

  def calculate_all_distances(self, tracked_objects: Dict) -> List[DistanceMeasurement]:
    """Calculate distances between all pairs of tracked objects"""
    if not self.calibration_enabled:
      return []

    distances = []
    object_ids = list(tracked_objects.keys())

    for i in range(len(object_ids)):
      for j in range(i + 1, len(object_ids)):
        obj1 = tracked_objects[object_ids[i]]
        obj2 = tracked_objects[object_ids[j]]

        # Calculate center-to-center distance
        distance = self.calculate_distance_between_objects(obj1, obj2, "center_to_center")
        distances.append(distance)

    return distances

  def detect_proximity_alerts(self, tracked_objects: Dict, alert_distance: float = 2.0) -> List[Dict]:
    """Detect objects that are too close to each other"""
    alerts = []

    distances = self.calculate_all_distances(tracked_objects)

    for distance in distances:
      if distance.distance_meters < alert_distance and distance.distance_meters > 0:
        alerts.append({
          'type': 'proximity_alert',
          'object1_id': distance.object1_id,
          'object2_id': distance.object2_id,
          'object1_class': distance.object1_class,
          'object2_class': distance.object2_class,
          'distance_meters': distance.distance_meters,
          'alert_threshold': alert_distance,
          'message': f"{distance.object1_class} and {distance.object2_class} are {distance.distance_meters:.2f}m apart (< {alert_distance}m)"
        })

    return alerts

  def validate_object_size(self, measurements: ObjectMeasurements, class_name: str, tolerance: float = 0.5) -> Dict:
    """Validate if object size is reasonable for its class"""
    if not class_name or class_name not in self.reference_sizes:
      return {'valid': True, 'reason': 'No reference size available'}

    reference = self.reference_sizes[class_name]

    # Check if dimensions are within tolerance
    width_ratio = measurements.width_meters / reference['width']
    height_ratio = measurements.height_meters / reference['height']

    width_valid = (1 - tolerance) <= width_ratio <= (1 + tolerance)
    height_valid = (1 - tolerance) <= height_ratio <= (1 + tolerance)

    if width_valid and height_valid:
      return {'valid': True, 'reason': 'Size within expected range'}
    else:
      return {
        'valid': False,
        'reason': f'Size anomaly: {measurements.width_meters:.2f}x{measurements.height_meters:.2f}m vs expected {reference["width"]}x{reference["height"]}m',
        'width_ratio': width_ratio,
        'height_ratio': height_ratio
      }

  def get_statistics(self, tracked_objects: Dict) -> Dict:
    """Get distance and size statistics for all tracked objects"""
    if not tracked_objects:
      return {}

    distances = self.calculate_all_distances(tracked_objects)

    # Calculate statistics
    if distances:
      distance_values = [d.distance_meters for d in distances if d.distance_meters > 0]
      if distance_values:
        stats = {
          'distance_stats': {
            'min_distance': min(distance_values),
            'max_distance': max(distance_values),
            'avg_distance': sum(distance_values) / len(distance_values),
            'total_pairs': len(distance_values)
          }
        }
      else:
        stats = {'distance_stats': {'total_pairs': 0}}
    else:
      stats = {'distance_stats': {'total_pairs': 0}}

    # Object size statistics
    object_sizes = []
    size_anomalies = []

    for obj_id, obj in tracked_objects.items():
      if 'bbox' in obj:
        measurements = self.calculate_object_size(obj['bbox'], obj.get('class_name'))
        object_sizes.append(measurements)

        # Check for size anomalies
        validation = self.validate_object_size(measurements, obj.get('class_name'))
        if not validation['valid']:
          size_anomalies.append({
            'object_id': obj_id,
            'class_name': obj.get('class_name'),
            'measurements': measurements,
            'validation': validation
          })

    if object_sizes:
      areas = [s.area_square_meters for s in object_sizes if s.area_square_meters > 0]
      if areas:
        stats['size_stats'] = {
          'min_area': min(areas),
          'max_area': max(areas),
          'avg_area': sum(areas) / len(areas),
          'total_objects': len(areas),
          'size_anomalies': len(size_anomalies)
        }
      else:
        stats['size_stats'] = {'total_objects': 0}

    stats['anomalies'] = size_anomalies

    return stats
distance_calculator = DistanceSizeCalculator()