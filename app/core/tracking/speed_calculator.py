import numpy as np
from typing import Dict, Optional


class SpeedCalculator:
  """Calculate object speed from tracking data"""

  def __init__(self, fps: int = 30):
    self.fps = fps

  def calculate_speed(
    self,
    velocity: tuple,
    pixels_per_meter: Optional[float] = None
  ) -> Dict[str, float]:
    """Calculate speed from velocity"""
    vx, vy = velocity

    # Speed in pixels per second
    speed_px = np.sqrt(vx ** 2 + vy ** 2) * self.fps

    result = {
      "speed_px_per_sec": speed_px,
      "velocity_x": vx * self.fps,
      "velocity_y": vy * self.fps
    }

    # Convert to real-world speed if calibrated
    if pixels_per_meter:
      speed_m_per_sec = speed_px / pixels_per_meter
      result["speed_m_per_sec"] = speed_m_per_sec
      result["speed_kmh"] = speed_m_per_sec * 3.6

    return result


# Global calculator instance
speed_calculator = SpeedCalculator()