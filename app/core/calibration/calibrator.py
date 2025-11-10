import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CameraCalibrator:
  """Camera calibration for pixel-to-meter conversion"""

  @staticmethod
  def calibrate_reference_object(points: List[Dict[str, float]]) -> Optional[float]:
    """
    Calibrate using reference object method
    Auto-detects if input is in centimeters and converts
    """
    if len(points) < 2:
      logger.error("Need at least 2 points for reference object calibration")
      return None

    try:
      p1, p2 = points[0], points[1]

      # Calculate pixel distance
      pixel_dist = np.sqrt(
        (p2["pixel_x"] - p1["pixel_x"]) ** 2 +
        (p2["pixel_y"] - p1["pixel_y"]) ** 2
      )

      # Calculate real-world distance
      real_dist = np.sqrt(
        (p2["real_x"] - p1["real_x"]) ** 2 +
        (p2["real_y"] - p1["real_y"]) ** 2
      )

      if pixel_dist == 0 or real_dist == 0:
        logger.error("Invalid calibration points (zero distance)")
        return None

      pixels_per_unit = pixel_dist / real_dist

      # SMART DETECTION: If ratio is very small, input was likely in cm
      # Typical pixels_per_meter: 50-500 (for normal camera views)
      # If we get < 10, the input was probably in cm
      if pixels_per_unit < 10:
        logger.warning(f"⚠️ Detected centimeter input! Converting to meters...")
        logger.warning(f"   Original ratio: {pixels_per_unit:.4f} (likely pixels/cm)")
        pixels_per_meter = pixels_per_unit * 100  # Convert cm to m
        logger.info(f"✅ Calibrated: {pixels_per_meter:.2f} pixels/meter")
      else:
        pixels_per_meter = pixels_per_unit
        logger.info(f"✅ Calibrated: {pixels_per_meter:.2f} pixels/meter")

      return pixels_per_meter

    except Exception as e:
      logger.error(f"Calibration failed: {e}")
      return None

  @staticmethod
  def pixel_to_meters(
    pixel_x: float,
    pixel_y: float,
    pixels_per_meter: float
  ) -> tuple:
    """Convert pixel coordinates to meters"""
    real_x = pixel_x / pixels_per_meter
    real_y = pixel_y / pixels_per_meter
    return real_x, real_y

  @staticmethod
  def calculate_distance(
    point1: tuple,
    point2: tuple,
    pixels_per_meter: float
  ) -> float:
    """Calculate real-world distance between two pixel points"""
    real_point1 = CameraCalibrator.pixel_to_meters(
      point1[0], point1[1], pixels_per_meter
    )
    real_point2 = CameraCalibrator.pixel_to_meters(
      point2[0], point2[1], pixels_per_meter
    )

    distance = np.sqrt(
      (real_point2[0] - real_point1[0]) ** 2 +
      (real_point2[1] - real_point1[1]) ** 2
    )
    return distance


# Global calibrator instance
calibrator = CameraCalibrator()