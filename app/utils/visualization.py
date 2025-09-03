import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..tracking.tracker_manager import TrackedObject
import colorsys


class TrackingVisualizer:
  """Visualization utilities for tracking results"""

  def __init__(self):
    self.colors = self._generate_colors(100)
    self.track_colors = {}

  def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for tracking visualization"""
    colors = []
    for i in range(num_colors):
      hue = i / num_colors
      saturation = 0.8
      value = 0.9
      rgb = colorsys.hsv_to_rgb(hue, saturation, value)
      colors.append(tuple(int(c * 255) for c in rgb))
    return colors

  def get_track_color(self, track_id: str) -> Tuple[int, int, int]:
    """Get consistent color for a track ID"""
    if track_id not in self.track_colors:
      color_index = len(self.track_colors) % len(self.colors)
      self.track_colors[track_id] = self.colors[color_index]
    return self.track_colors[track_id]

  def draw_tracked_objects(self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject],
                           show_trajectory: bool = True, show_velocity: bool = True,
                           show_info: bool = True) -> np.ndarray:
    """Draw tracked objects on image"""
    img = image.copy()

    for track_id, obj in tracked_objects.items():
      if obj.time_since_update > 5:  # Skip stale objects
        continue

      color = self.get_track_color(track_id)

      # Draw bounding box
      x1, y1, x2, y2 = [int(x) for x in obj.bbox]
      cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

      # Draw centroid
      cx, cy = [int(x) for x in obj.centroid]
      cv2.circle(img, (cx, cy), 5, color, -1)

      # Draw trajectory
      if show_trajectory and len(obj.trajectory) > 1:
        trajectory_points = [(int(x), int(y)) for x, y in obj.trajectory]
        for i in range(1, len(trajectory_points)):
          cv2.line(img, trajectory_points[i - 1], trajectory_points[i], color, 2)

      # Draw velocity vector
      if show_velocity and obj.velocity != (0, 0):
        vx, vy = obj.velocity
        end_x = int(cx + vx * 5)  # Scale velocity for visibility
        end_y = int(cy + vy * 5)
        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), color, 2)

      # Draw info text
      if show_info:
        info_text = f"ID: {track_id[:6]}"
        info_text += f"\nClass: {obj.class_name}"
        info_text += f"\nConf: {obj.confidence:.2f}"
        info_text += f"\nAge: {obj.age}"

        lines = info_text.split('\n')
        for i, line in enumerate(lines):
          y_offset = y1 - 10 - (len(lines) - i - 1) * 20
          cv2.putText(img, line, (x1, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

  def draw_zones(self, image: np.ndarray, zones: Dict[str, List[Tuple[int, int]]],
                 zone_occupancy: Optional[Dict[str, List[str]]] = None) -> np.ndarray:
    """Draw defined zones on image"""
    img = image.copy()

    for zone_id, polygon in zones.items():
      # Determine zone color based on occupancy
      if zone_occupancy and zone_id in zone_occupancy:
        occupied = len(zone_occupancy[zone_id]) > 0
        color = (0, 255, 0) if not occupied else (0, 0, 255)  # Green if empty, red if occupied
      else:
        color = (255, 255, 0)  # Yellow for undefined occupancy

      # Draw polygon
      points = np.array(polygon, np.int32)
      cv2.polylines(img, [points], True, color, 2)

      # Fill with transparent color
      overlay = img.copy()
      cv2.fillPoly(overlay, [points], color)
      img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)

      # Draw zone label
      center_x = int(np.mean([p[0] for p in polygon]))
      center_y = int(np.mean([p[1] for p in polygon]))
      cv2.putText(img, zone_id, (center_x, center_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img

  def draw_speed_info(self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject],
                      speed_data: Dict[str, Dict]) -> np.ndarray:
    """Draw speed information on tracked objects"""
    img = image.copy()

    for track_id, obj in tracked_objects.items():
      if track_id not in speed_data:
        continue

      speed_info = speed_data[track_id]
      speed_px = speed_info.get('speed_px_per_sec', 0)
      speed_m = speed_info.get('speed_m_per_sec', 0)

      # Position for speed text
      x1, y1, x2, y2 = [int(x) for x in obj.bbox]

      # Draw speed info
      speed_text = f"Speed: {speed_px:.1f} px/s"
      if speed_m != speed_px:
        speed_text += f" ({speed_m:.1f} m/s)"

      cv2.putText(img, speed_text, (x1, y2 + 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

  def create_tracking_overlay(self, image: np.ndarray,
                              tracked_objects: Dict[str, TrackedObject],
                              zones: Optional[Dict[str, List[Tuple[int, int]]]] = None,
                              zone_occupancy: Optional[Dict[str, List[str]]] = None,
                              speed_data: Optional[Dict[str, Dict]] = None,
                              show_trajectory: bool = True,
                              show_velocity: bool = True,
                              show_zones: bool = True,
                              show_speed: bool = True) -> np.ndarray:
    """Create comprehensive tracking overlay"""
    img = image.copy()

    # Draw zones first (background)
    if show_zones and zones:
      img = self.draw_zones(img, zones, zone_occupancy)

    # Draw tracked objects
    img = self.draw_tracked_objects(img, tracked_objects, show_trajectory, show_velocity, True)

    # Draw speed information
    if show_speed and speed_data:
      img = self.draw_speed_info(img, tracked_objects, speed_data)

    # Draw summary info
    active_count = len([obj for obj in tracked_objects.values() if obj.time_since_update < 5])
    total_count = len(tracked_objects)

    summary_text = f"Active: {active_count} | Total: {total_count}"
    cv2.putText(img, summary_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


# Global visualizer instance
visualizer = TrackingVisualizer()