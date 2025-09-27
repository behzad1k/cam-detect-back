import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict


class CalibrationHelpers:
  @staticmethod
  def detect_checkerboard(image: np.ndarray, pattern_size: Tuple[int, int] = (9, 6)) -> Optional[
    List[Tuple[float, float]]]:
    """
    Detect checkerboard pattern for automatic calibration point detection
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
      # Refine corner positions
      criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
      corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

      # Convert to list of tuples
      corner_points = [(float(corner[0][0]), float(corner[0][1])) for corner in corners2]
      return corner_points

    return None

  @staticmethod
  def detect_aruco_markers(image: np.ndarray, dictionary_type: int = cv2.aruco.DICT_6X6_250) -> List[Dict]:
    """
    Detect ArUco markers for automatic calibration
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(dictionary_type)
    parameters = cv2.aruco.DetectorParameters_create()

    # Detect markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    markers = []
    if ids is not None:
      for i, marker_id in enumerate(ids.flatten()):
        # Calculate center of marker
        center_x = np.mean(corners[i][0][:, 0])
        center_y = np.mean(corners[i][0][:, 1])

        markers.append({
          "id": int(marker_id),
          "center": (float(center_x), float(center_y)),
          "corners": corners[i][0].tolist()
        })

    return markers

  @staticmethod
  def estimate_vanishing_point(lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Optional[
    Tuple[float, float]]:
    """
    Estimate vanishing point from parallel lines in the scene
    """
    if len(lines) < 2:
      return None

    # Convert lines to homogeneous coordinates and find intersection
    intersections = []

    for i in range(len(lines)):
      for j in range(i + 1, len(lines)):
        line1 = lines[i]
        line2 = lines[j]

        # Calculate intersection point
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) > 1e-10:  # Avoid division by zero
          px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
          py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
          intersections.append((px, py))

    if not intersections:
      return None

    # Use median of intersections as vanishing point estimate
    intersections = np.array(intersections)
    vp_x = np.median(intersections[:, 0])
    vp_y = np.median(intersections[:, 1])

    return (float(vp_x), float(vp_y))

  @staticmethod
  def draw_calibration_overlay(image: np.ndarray, calibration_points: List[Tuple[float, float]],
                               mode: str = "perspective_transform") -> np.ndarray:
    """
    Draw calibration overlay on image for visualization
    """
    overlay = image.copy()

    # Draw calibration points
    for i, (x, y) in enumerate(calibration_points):
      cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 0), -1)
      cv2.putText(overlay, f"P{i + 1}", (int(x) + 10, int(y) - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw connections based on mode
    if mode == "perspective_transform" and len(calibration_points) == 4:
      # Draw rectangle
      points = np.array(calibration_points, dtype=np.int32)
      cv2.polylines(overlay, [points], True, (255, 0, 0), 2)
    elif mode == "reference_object" and len(calibration_points) >= 2:
      # Draw line between reference points
      p1 = (int(calibration_points[0][0]), int(calibration_points[0][1]))
      p2 = (int(calibration_points[1][0]), int(calibration_points[1][1]))
      cv2.line(overlay, p1, p2, (255, 0, 0), 2)

    return overlay

