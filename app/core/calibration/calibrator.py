# app/core/calibration/advanced_calibrator.py
"""
Advanced Camera Calibration System with 3D depth estimation
Supports multiple calibration methods for accurate real-world measurements
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Supported calibration methods"""

    REFERENCE_OBJECT = "reference_object"  # Simple 2D reference
    PERSPECTIVE_TRANSFORM = "perspective_transform"  # 3D perspective
    VANISHING_POINT = "vanishing_point"  # Depth from vanishing points
    GROUND_PLANE = "ground_plane"  # Ground plane homography
    CHECKERBOARD = "checkerboard"  # Camera intrinsics


@dataclass
class CalibrationPoint:
    """Calibration point with pixel and real-world coordinates"""

    pixel_x: float
    pixel_y: float
    real_x: float  # meters
    real_y: float  # meters
    real_z: float = 0.0  # meters (height/depth)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""

    focal_length_x: float  # pixels
    focal_length_y: float  # pixels
    principal_point_x: float  # pixels
    principal_point_y: float  # pixels
    distortion_coeffs: np.ndarray = None

    def to_matrix(self) -> np.ndarray:
        """Convert to camera matrix"""
        return np.array(
            [
                [self.focal_length_x, 0, self.principal_point_x],
                [0, self.focal_length_y, self.principal_point_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )


@dataclass
class CalibrationResult:
    """Complete calibration result"""

    method: CalibrationMethod
    success: bool

    # Basic metrics
    pixels_per_meter: Optional[float] = None

    # Advanced calibration
    homography_matrix: Optional[np.ndarray] = None
    perspective_transform: Optional[np.ndarray] = None
    camera_intrinsics: Optional[CameraIntrinsics] = None

    # Vanishing point data
    vanishing_points: Optional[List[Tuple[float, float]]] = None
    horizon_line: Optional[Tuple[float, float, float]] = None  # ax + by + c = 0

    # Quality metrics
    reprojection_error: Optional[float] = None
    calibration_points: Optional[List[CalibrationPoint]] = None

    # Camera pose (for 3D)
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    camera_height: Optional[float] = None  # meters above ground


class CameraCalibrator:
    """
    Advanced calibration system supporting multiple methods:

    1. Reference Object: Simple 2D distance-based calibration
    2. Perspective Transform: 4-point homography for ground plane
    3. Vanishing Point: Depth estimation from perspective lines
    4. Ground Plane: Full 3D calibration with camera pose
    5. Checkerboard: Camera intrinsics calibration
    """

    def __init__(self):
        self.current_calibration: Optional[CalibrationResult] = None

    # ==================== METHOD 1: REFERENCE OBJECT ====================

    def calibrate_reference_object(
        self, points: List[Dict], known_distance: Optional[float] = None
    ) -> CalibrationResult:
        """
        Simple 2D calibration using reference object

        Args:
            points: List of {pixel_x, pixel_y, real_x, real_y}
            known_distance: Known distance in meters (optional)

        Returns:
            CalibrationResult with pixels_per_meter
        """
        if len(points) < 2:
            return CalibrationResult(
                method=CalibrationMethod.REFERENCE_OBJECT, success=False
            )

        try:
            cal_points = [
                CalibrationPoint(
                    pixel_x=p["pixel_x"],
                    pixel_y=p["pixel_y"],
                    real_x=p.get("real_x", 0),
                    real_y=p.get("real_y", 0),
                )
                for p in points
            ]

            # Calculate pixel distance
            p1, p2 = cal_points[0], cal_points[1]
            pixel_dist = np.sqrt(
                (p2.pixel_x - p1.pixel_x) ** 2 + (p2.pixel_y - p1.pixel_y) ** 2
            )

            # Calculate real distance
            if known_distance is not None:
                real_dist = known_distance
            else:
                real_dist = np.sqrt(
                    (p2.real_x - p1.real_x) ** 2 + (p2.real_y - p1.real_y) ** 2
                )

            if pixel_dist == 0 or real_dist == 0:
                return CalibrationResult(
                    method=CalibrationMethod.REFERENCE_OBJECT, success=False
                )

            pixels_per_meter = pixel_dist / real_dist

            # Auto-detect centimeter input
            if pixels_per_meter < 10:
                logger.warning("Detected centimeter input, converting to meters")
                pixels_per_meter *= 100

            logger.info(f"✅ Reference calibration: {pixels_per_meter:.2f} px/m")

            return CalibrationResult(
                method=CalibrationMethod.REFERENCE_OBJECT,
                success=True,
                pixels_per_meter=pixels_per_meter,
                calibration_points=cal_points,
            )

        except Exception as e:
            logger.error(f"Reference calibration failed: {e}")
            return CalibrationResult(
                method=CalibrationMethod.REFERENCE_OBJECT, success=False
            )

    # ==================== METHOD 2: PERSPECTIVE TRANSFORM ====================

    def calibrate_perspective_transform(
        self,
        points: List[Dict],
        real_world_dimensions: Tuple[float, float],  # (width_m, height_m)
    ) -> CalibrationResult:
        """
        Calibrate using 4-point perspective transform

        This creates a "bird's eye view" transformation that accounts for
        perspective distortion. Perfect for overhead monitoring.

        Args:
            points: 4 points defining a rectangle in pixel space (order: TL, TR, BR, BL)
            real_world_dimensions: Real dimensions of the rectangle (width, height) in meters

        Returns:
            CalibrationResult with homography matrix
        """
        if len(points) != 4:
            logger.error("Perspective transform requires exactly 4 points")
            return CalibrationResult(
                method=CalibrationMethod.PERSPECTIVE_TRANSFORM, success=False
            )

        try:
            # Extract pixel coordinates
            src_points = np.array(
                [[p["pixel_x"], p["pixel_y"]] for p in points], dtype=np.float32
            )

            # Define destination rectangle in "world" coordinates (pixels)
            # We'll use 100 pixels per meter as reference scale
            scale = 100  # pixels per meter in transformed space
            width_px = real_world_dimensions[0] * scale
            height_px = real_world_dimensions[1] * scale

            dst_points = np.array(
                [[0, 0], [width_px, 0], [width_px, height_px], [0, height_px]],
                dtype=np.float32,
            )

            # Compute homography matrix
            homography_matrix, status = cv2.findHomography(src_points, dst_points)

            if homography_matrix is None:
                raise ValueError("Failed to compute homography")

            # Calculate average pixels per meter
            # Use the scale we defined in destination space
            pixels_per_meter = scale

            logger.info(f"✅ Perspective calibration: {pixels_per_meter:.2f} px/m")
            logger.info(
                f"   Area: {real_world_dimensions[0]:.2f}m x {real_world_dimensions[1]:.2f}m"
            )

            return CalibrationResult(
                method=CalibrationMethod.PERSPECTIVE_TRANSFORM,
                success=True,
                pixels_per_meter=pixels_per_meter,
                homography_matrix=homography_matrix,
                perspective_transform=homography_matrix,
                calibration_points=[
                    CalibrationPoint(p["pixel_x"], p["pixel_y"], 0, 0) for p in points
                ],
            )

        except Exception as e:
            logger.error(f"Perspective calibration failed: {e}")
            return CalibrationResult(
                method=CalibrationMethod.PERSPECTIVE_TRANSFORM, success=False
            )

    # ==================== METHOD 3: VANISHING POINT ====================

    def calibrate_vanishing_point(
        self,
        parallel_lines: List[List[Tuple[float, float]]],
        reference_points: List[Dict],
        camera_height: float,
    ) -> CalibrationResult:
        """
        Calibrate using vanishing point detection for depth estimation

        This method uses perspective geometry to estimate depth.
        Perfect for hallways, roads, or any scene with parallel lines.

        Args:
            parallel_lines: List of line segments (each line is [(x1,y1), (x2,y2)])
            reference_points: At least 2 points with known real-world distance
            camera_height: Camera height above ground in meters

        Returns:
            CalibrationResult with vanishing points and depth estimation
        """
        try:
            # Find vanishing points from parallel lines
            vanishing_points = self._find_vanishing_points(parallel_lines)

            if len(vanishing_points) < 1:
                raise ValueError("Could not find vanishing point")

            # Find horizon line (line connecting vanishing points)
            if len(vanishing_points) >= 2:
                vp1, vp2 = vanishing_points[0], vanishing_points[1]
                # Line equation: ax + by + c = 0
                horizon_line = self._compute_line_equation(vp1, vp2)
            else:
                horizon_line = None

            # Calibrate scale using reference points
            ref_result = self.calibrate_reference_object(reference_points)

            if not ref_result.success:
                raise ValueError("Reference point calibration failed")

            logger.info(f"✅ Vanishing point calibration")
            logger.info(f"   Vanishing points: {len(vanishing_points)}")
            logger.info(f"   Scale: {ref_result.pixels_per_meter:.2f} px/m")

            return CalibrationResult(
                method=CalibrationMethod.VANISHING_POINT,
                success=True,
                pixels_per_meter=ref_result.pixels_per_meter,
                vanishing_points=vanishing_points,
                horizon_line=horizon_line,
                camera_height=camera_height,
            )

        except Exception as e:
            logger.error(f"Vanishing point calibration failed: {e}")
            return CalibrationResult(
                method=CalibrationMethod.VANISHING_POINT, success=False
            )

    def _find_vanishing_points(
        self, line_segments: List[List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """Find vanishing points from parallel line segments"""
        if len(line_segments) < 2:
            return []

        vanishing_points = []

        # Find intersection of line pairs
        for i in range(len(line_segments)):
            for j in range(i + 1, len(line_segments)):
                line1 = line_segments[i]
                line2 = line_segments[j]

                intersection = self._line_intersection(
                    line1[0], line1[1], line2[0], line2[1]
                )

                if intersection is not None:
                    vanishing_points.append(intersection)

        # Cluster vanishing points (should converge to same point)
        if len(vanishing_points) > 0:
            # Simple average for now (could use RANSAC for robustness)
            avg_vp = (
                np.mean([vp[0] for vp in vanishing_points]),
                np.mean([vp[1] for vp in vanishing_points]),
            )
            return [avg_vp]

        return []

    def _line_intersection(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """Find intersection of two lines defined by point pairs"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-6:
            return None  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def _compute_line_equation(
        self, p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> Tuple[float, float, float]:
        """Compute line equation ax + by + c = 0"""
        x1, y1 = p1
        x2, y2 = p2

        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        # Normalize
        norm = np.sqrt(a**2 + b**2)
        if norm > 0:
            a, b, c = a / norm, b / norm, c / norm

        return (a, b, c)

    # ==================== METHOD 4: GROUND PLANE HOMOGRAPHY ====================

    def calibrate_ground_plane(
        self,
        ground_points: List[Dict],  # Points on the ground plane with 3D coords
        camera_height: float,
        image_size: Tuple[int, int],  # (width, height)
    ) -> CalibrationResult:
        """
        Full 3D calibration using ground plane homography

        This computes the camera pose relative to the ground plane and
        allows accurate 3D->2D projection and distance estimation.

        Args:
            ground_points: At least 4 points with pixel_x, pixel_y, real_x, real_y, real_z
            camera_height: Camera height above ground in meters
            image_size: Image dimensions (width, height)

        Returns:
            CalibrationResult with full 3D calibration
        """
        if len(ground_points) < 4:
            logger.error("Ground plane calibration requires at least 4 points")
            return CalibrationResult(
                method=CalibrationMethod.GROUND_PLANE, success=False
            )

        try:
            # Extract points
            image_points = np.array(
                [[p["pixel_x"], p["pixel_y"]] for p in ground_points], dtype=np.float32
            )

            object_points = np.array(
                [
                    [p.get("real_x", 0), p.get("real_y", 0), p.get("real_z", 0)]
                    for p in ground_points
                ],
                dtype=np.float32,
            )

            # Estimate camera intrinsics (assuming no distortion)
            # Focal length estimate: approximately image width
            focal_length = image_size[0]
            principal_point = (image_size[0] / 2, image_size[1] / 2)

            camera_matrix = np.array(
                [
                    [focal_length, 0, principal_point[0]],
                    [0, focal_length, principal_point[1]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            dist_coeffs = np.zeros(5)

            # Solve PnP (Perspective-n-Point)
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                raise ValueError("PnP solver failed")

            # Compute ground plane homography
            # We want to map ground plane (z=0) to image
            ground_points_2d = object_points[:, :2]  # Just x, y
            homography, _ = cv2.findHomography(ground_points_2d, image_points)

            # Estimate pixels per meter at ground level
            # Sample a few points and average
            test_points_world = np.array(
                [[0, 0], [1, 0], [0, 1]], dtype=np.float32
            ).reshape(-1, 1, 2)

            test_points_image = cv2.perspectiveTransform(test_points_world, homography)

            dist_01 = np.linalg.norm(test_points_image[1] - test_points_image[0])
            dist_02 = np.linalg.norm(test_points_image[2] - test_points_image[0])

            pixels_per_meter = (dist_01 + dist_02) / 2.0

            # Calculate reprojection error
            projected_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            reprojection_error = np.mean(
                np.linalg.norm(projected_points.reshape(-1, 2) - image_points, axis=1)
            )

            intrinsics = CameraIntrinsics(
                focal_length_x=focal_length,
                focal_length_y=focal_length,
                principal_point_x=principal_point[0],
                principal_point_y=principal_point[1],
                distortion_coeffs=dist_coeffs,
            )

            logger.info(f"✅ Ground plane calibration")
            logger.info(f"   Pixels per meter: {pixels_per_meter:.2f}")
            logger.info(f"   Camera height: {camera_height:.2f}m")
            logger.info(f"   Reprojection error: {reprojection_error:.2f}px")

            return CalibrationResult(
                method=CalibrationMethod.GROUND_PLANE,
                success=True,
                pixels_per_meter=pixels_per_meter,
                homography_matrix=homography,
                camera_intrinsics=intrinsics,
                rotation_vector=rvec,
                translation_vector=tvec,
                camera_height=camera_height,
                reprojection_error=reprojection_error,
                calibration_points=[
                    CalibrationPoint(
                        p["pixel_x"],
                        p["pixel_y"],
                        p.get("real_x", 0),
                        p.get("real_y", 0),
                        p.get("real_z", 0),
                    )
                    for p in ground_points
                ],
            )

        except Exception as e:
            logger.error(f"Ground plane calibration failed: {e}")
            return CalibrationResult(
                method=CalibrationMethod.GROUND_PLANE, success=False
            )

    # ==================== METHOD 5: CHECKERBOARD ====================

    def calibrate_checkerboard(
        self,
        images: List[np.ndarray],
        checkerboard_size: Tuple[int, int],  # (rows, cols) of inner corners
        square_size: float,  # Size of each square in meters
    ) -> CalibrationResult:
        """
        Calibrate camera intrinsics using checkerboard pattern

        This computes accurate camera matrix and distortion coefficients.

        Args:
            images: List of images with checkerboard visible
            checkerboard_size: Number of inner corners (rows, cols)
            square_size: Physical size of each square in meters

        Returns:
            CalibrationResult with camera intrinsics
        """
        try:
            # Prepare object points
            objp = np.zeros(
                (checkerboard_size[0] * checkerboard_size[1], 3), np.float32
            )
            objp[:, :2] = np.mgrid[
                0 : checkerboard_size[0], 0 : checkerboard_size[1]
            ].T.reshape(-1, 2)
            objp *= square_size

            object_points = []
            image_points = []

            # Find checkerboard corners in each image
            for img in images:
                gray = (
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if len(img.shape) == 3
                    else img
                )

                ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

                if ret:
                    object_points.append(objp)

                    # Refine corner locations
                    corners_refined = cv2.cornerSubPix(
                        gray,
                        corners,
                        (11, 11),
                        (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                    )
                    image_points.append(corners_refined)

            if len(object_points) < 3:
                raise ValueError(
                    f"Need at least 3 valid images, found {len(object_points)}"
                )

            # Calibrate camera
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, gray.shape[::-1], None, None
            )

            if not ret:
                raise ValueError("Camera calibration failed")

            # Calculate reprojection error
            total_error = 0
            for i in range(len(object_points)):
                projected, _ = cv2.projectPoints(
                    object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(image_points[i], projected, cv2.NORM_L2) / len(
                    projected
                )
                total_error += error

            mean_error = total_error / len(object_points)

            intrinsics = CameraIntrinsics(
                focal_length_x=camera_matrix[0, 0],
                focal_length_y=camera_matrix[1, 1],
                principal_point_x=camera_matrix[0, 2],
                principal_point_y=camera_matrix[1, 2],
                distortion_coeffs=dist_coeffs,
            )

            logger.info(f"✅ Checkerboard calibration")
            logger.info(f"   Images used: {len(object_points)}")
            logger.info(f"   Focal length: {intrinsics.focal_length_x:.2f}px")
            logger.info(f"   Reprojection error: {mean_error:.4f}px")

            return CalibrationResult(
                method=CalibrationMethod.CHECKERBOARD,
                success=True,
                camera_intrinsics=intrinsics,
                reprojection_error=mean_error,
            )

        except Exception as e:
            logger.error(f"Checkerboard calibration failed: {e}")
            return CalibrationResult(
                method=CalibrationMethod.CHECKERBOARD, success=False
            )

    # ==================== MEASUREMENT METHODS ====================

    def pixel_to_world(
        self,
        pixel_point: Tuple[float, float],
        calibration: CalibrationResult,
        z_height: float = 0.0,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Convert pixel coordinates to real-world coordinates

        Args:
            pixel_point: (x, y) in pixels
            calibration: CalibrationResult from calibration
            z_height: Height above ground plane in meters

        Returns:
            (x, y, z) in meters, or None if conversion fails
        """
        if calibration.method == CalibrationMethod.PERSPECTIVE_TRANSFORM:
            if calibration.homography_matrix is None:
                return None

            # Apply inverse homography
            pt = np.array([[pixel_point]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(
                pt, np.linalg.inv(calibration.homography_matrix)
            )

            # Convert from pixel space to meters (we used 100 px/m scale)
            scale = 100.0
            x = world_pt[0, 0, 0] / scale
            y = world_pt[0, 0, 1] / scale

            return (x, y, z_height)

        elif calibration.method == CalibrationMethod.REFERENCE_OBJECT:
            # Simple 2D conversion
            if calibration.pixels_per_meter is None:
                return None

            x = pixel_point[0] / calibration.pixels_per_meter
            y = pixel_point[1] / calibration.pixels_per_meter

            return (x, y, z_height)

        return None

    def compute_distance_3d(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        calibration: CalibrationResult,
        z1: float = 0.0,
        z2: float = 0.0,
    ) -> Optional[float]:
        """
        Compute 3D distance between two pixel points

        Args:
            point1: First point in pixels
            point2: Second point in pixels
            calibration: CalibrationResult
            z1, z2: Heights above ground for each point

        Returns:
            Distance in meters, or None if conversion fails
        """
        world1 = self.pixel_to_world(point1, calibration, z1)
        world2 = self.pixel_to_world(point2, calibration, z2)

        if world1 is None or world2 is None:
            return None

        distance = np.sqrt(
            (world2[0] - world1[0]) ** 2
            + (world2[1] - world1[1]) ** 2
            + (world2[2] - world1[2]) ** 2
        )

        return distance


# Global instance
calibrator = CameraCalibrator()
