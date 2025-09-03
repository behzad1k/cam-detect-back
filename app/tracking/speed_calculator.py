import numpy as np
from typing import Dict
from .tracker_manager import TrackedObject

class SpeedCalculator:
  """Calculate object speed and movement patterns"""

  def __init__(self, fps=30, pixel_to_meter_ratio=None):
    self.fps = fps
    self.pixel_to_meter_ratio = pixel_to_meter_ratio or 1.0  # Default: 1 pixel = 1 meter

  def calculate_speed(self, tracked_object: TrackedObject) -> Dict[str, float]:
    """Calculate speed in pixels/second and optionally meters/second"""
    if len(tracked_object.trajectory) < 2:
      return {'speed_px_per_sec': 0, 'speed_m_per_sec': 0, 'direction': 0}

    # Get last two positions
    current_pos = tracked_object.trajectory[-1]
    prev_pos = tracked_object.trajectory[-2]

    # Calculate distance
    dx = current_pos[0] - prev_pos[0]
    dy = current_pos[1] - prev_pos[1]
    distance_px = np.sqrt(dx ** 2 + dy ** 2)

    # Calculate speed (assuming 1 frame time difference)
    speed_px_per_sec = distance_px * self.fps
    speed_m_per_sec = speed_px_per_sec * self.pixel_to_meter_ratio

    # Calculate direction (angle in degrees)
    direction = np.degrees(np.arctan2(dy, dx))

    return {
      'speed_px_per_sec': speed_px_per_sec,
      'speed_m_per_sec': speed_m_per_sec,
      'direction': direction,
      'velocity_x': dx * self.fps,
      'velocity_y': dy * self.fps
    }

  def calculate_average_speed(self, tracked_object: TrackedObject, window_size=10) -> Dict[str, float]:
    """Calculate average speed over a window of frames"""
    if len(tracked_object.trajectory) < 2:
      return {'avg_speed_px_per_sec': 0, 'avg_speed_m_per_sec': 0}

    # Get recent positions
    recent_positions = list(tracked_object.trajectory)[-window_size:]

    if len(recent_positions) < 2:
      return self.calculate_speed(tracked_object)

    total_distance = 0
    for i in range(1, len(recent_positions)):
      dx = recent_positions[i][0] - recent_positions[i - 1][0]
      dy = recent_positions[i][1] - recent_positions[i - 1][1]
      total_distance += np.sqrt(dx ** 2 + dy ** 2)

    time_frames = len(recent_positions) - 1
    avg_speed_px_per_sec = (total_distance / time_frames) * self.fps
    avg_speed_m_per_sec = avg_speed_px_per_sec * self.pixel_to_meter_ratio

    return {
      'avg_speed_px_per_sec': avg_speed_px_per_sec,
      'avg_speed_m_per_sec': avg_speed_m_per_sec
    }

speed_calculator = SpeedCalculator()
