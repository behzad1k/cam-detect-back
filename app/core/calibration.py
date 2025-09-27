import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass, asdict
from enum import Enum


class CalibrationMode(Enum):
  PERSPECTIVE_TRANSFORM = "perspective_transform"
  REFERENCE_OBJECT = "reference_object"
  VANISHING_POINT = "vanishing_point"


@dataclass
class CalibrationPoint:
  pixel_x: float
  pixel_y: float
  real_x: float  # in meters
  real_y: float  # in meters


@dataclass
class CalibrationData:
  mode: CalibrationMode
  points: List[CalibrationPoint]
  perspective_matrix: Optional[List[List[float]]] = None
  meters_per_pixel: Optional[float] = None
  reference_width_meters: Optional[float] = None
  reference_height_meters: Optional[float] = None
  vanishing_point: Optional[Tuple[float, float]] = None
  horizon_line: Optional[float] = None
  calibration_timestamp: float = 0.0
  frame_width: int = 640
  frame_height: int = 480


class CameraCalibration:
  def __init__(self):
    self.calibration_data: Optional[CalibrationData] = None
    self.is_calibrated = False

  def set_perspective_transform_calibration(self, points: List[CalibrationPoint], frame_width: int,
                                            frame_height: int) -> bool:
    """
    Calibrate using 4 corner points of a known rectangle in real world
    """
    if len(points) != 4:
      raise ValueError("Perspective transform requires exactly 4 points")

    # Source points (pixel coordinates)
    src_points = np.float32([[p.pixel_x, p.pixel_y] for p in points])

    # Destination points (real world coordinates)
    dst_points = np.float32([[p.real_x, p.real_y] for p in points])

    # Calculate perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    self.calibration_data = CalibrationData(
      mode=CalibrationMode.PERSPECTIVE_TRANSFORM,
      points=points,
      perspective_matrix=perspective_matrix.tolist(),
      calibration_timestamp=time.time(),
      frame_width=frame_width,
      frame_height=frame_height
    )
    self.is_calibrated = True
    return True

  def set_reference_object_calibration(self, points: List[CalibrationPoint], reference_width_meters: float,
                                       reference_height_meters: float, frame_width: int, frame_height: int) -> bool:
    """
    Calibrate using a known reference object (e.g., a 1m x 1m square)
    """
    if len(points) < 2:
      raise ValueError("Reference object calibration requires at least 2 points")

    # Calculate pixels per meter based on reference object
    point1, point2 = points[0], points[1]
    pixel_distance = np.sqrt((point2.pixel_x - point1.pixel_x) ** 2 + (point2.pixel_y - point1.pixel_y) ** 2)
    real_distance = np.sqrt((point2.real_x - point1.real_x) ** 2 + (point2.real_y - point1.real_y) ** 2)

    meters_per_pixel = real_distance / pixel_distance if pixel_distance > 0 else 0

    self.calibration_data = CalibrationData(
      mode=CalibrationMode.REFERENCE_OBJECT,
      points=points,
      meters_per_pixel=meters_per_pixel,
      reference_width_meters=reference_width_meters,
      reference_height_meters=reference_height_meters,
      calibration_timestamp=time.time(),
      frame_width=frame_width,
      frame_height=frame_height
    )
    self.is_calibrated = True
    return True

  def set_vanishing_point_calibration(self, vanishing_point: Tuple[float, float], horizon_line: float,
                                      reference_distance_meters: float, reference_pixel_height: float, frame_width: int,
                                      frame_height: int) -> bool:
    """
    Calibrate using vanishing point for perspective depth estimation
    """
    self.calibration_data = CalibrationData(
      mode=CalibrationMode.VANISHING_POINT,
      points=[],
      vanishing_point=vanishing_point,
      horizon_line=horizon_line,
      meters_per_pixel=reference_distance_meters / reference_pixel_height,
      calibration_timestamp=time.time(),
      frame_width=frame_width,
      frame_height=frame_height
    )
    self.is_calibrated = True
    return True

  def pixel_to_meters(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
    """Convert pixel coordinates to real-world meters"""
    if not self.is_calibrated:
      return pixel_x, pixel_y

    if self.calibration_data.mode == CalibrationMode.PERSPECTIVE_TRANSFORM:
      # Use perspective transformation
      perspective_matrix = np.array(self.calibration_data.perspective_matrix)
      pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
      real_point = cv2.perspectiveTransform(pixel_point.reshape(-1, 1, 2), perspective_matrix)
      return float(real_point[0, 0, 0]), float(real_point[0, 0, 1])

    elif self.calibration_data.mode == CalibrationMode.REFERENCE_OBJECT:
      # Simple scaling based on reference object
      return (pixel_x * self.calibration_data.meters_per_pixel,
              pixel_y * self.calibration_data.meters_per_pixel)

    elif self.calibration_data.mode == CalibrationMode.VANISHING_POINT:
      # Perspective depth estimation using vanishing point
      vp_x, vp_y = self.calibration_data.vanishing_point
      horizon_y = self.calibration_data.horizon_line

      # Calculate perspective scaling factor
      if abs(pixel_y - horizon_y) > 0:
        depth_factor = abs(vp_y - horizon_y) / abs(pixel_y - horizon_y)
        scale_factor = self.calibration_data.meters_per_pixel * depth_factor
        return pixel_x * scale_factor, pixel_y * scale_factor

    return pixel_x, pixel_y

  def calculate_distance(self, point1_pixel: Tuple[float, float], point2_pixel: Tuple[float, float]) -> float:
    """Calculate real-world distance between two pixel points"""
    real_point1 = self.pixel_to_meters(*point1_pixel)
    real_point2 = self.pixel_to_meters(*point2_pixel)

    distance = np.sqrt((real_point2[0] - real_point1[0]) ** 2 + (real_point2[1] - real_point1[1]) ** 2)
    return distance

  def get_calibration_info(self) -> Dict:
    """Get current calibration information"""
    if not self.is_calibrated:
      return {"calibrated": False}

    return {
      "calibrated": True,
      "mode": self.calibration_data.mode.value,
      "timestamp": self.calibration_data.calibration_timestamp,
      "frame_size": [self.calibration_data.frame_width, self.calibration_data.frame_height],
      "meters_per_pixel": self.calibration_data.meters_per_pixel
    }

  def save_calibration(self, filepath: str) -> bool:
    """Save calibration to file"""
    if not self.is_calibrated:
      return False

    try:
      with open(filepath, 'w') as f:
        json.dump(asdict(self.calibration_data), f, indent=2)
      return True
    except Exception as e:
      print(f"Error saving calibration: {e}")
      return False

  def load_calibration(self, filepath: str) -> bool:
    """Load calibration from file"""
    try:
      with open(filepath, 'r') as f:
        data = json.load(f)

      # Convert back to CalibrationData
      points = [CalibrationPoint(**p) for p in data.get('points', [])]
      data['points'] = points
      data['mode'] = CalibrationMode(data['mode'])

      self.calibration_data = CalibrationData(**data)
      self.is_calibrated = True
      return True
    except Exception as e:
      print(f"Error loading calibration: {e}")
      return False
