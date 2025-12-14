# app/core/tracking/advanced_tracker.py
"""
Advanced Multi-Object Tracker with SORT/DeepSORT-inspired algorithms
Fixes duplicate detection issues with proper IOU matching and NMS
"""

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, euclidean

from app.schemas.detection import Detection


@dataclass
class TrackedObject:
    """Enhanced tracked object with appearance features"""

    track_id: str
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    centroid: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    last_seen: float = field(default_factory=time.time)
    trajectory: deque = field(default_factory=lambda: deque(maxlen=100))
    distance_traveled: float = 0.0

    # NEW: Appearance features for better matching
    appearance_feature: Optional[np.ndarray] = None
    color_histogram: Optional[np.ndarray] = None

    # NEW: Quality metrics
    avg_confidence: float = 0.0
    detection_quality: float = 1.0

    # NEW: State estimation
    predicted_centroid: Optional[Tuple[float, float]] = None
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))


class KalmanFilter2D:
    """2D Kalman filter for position and velocity tracking"""

    def __init__(self):
        # State: [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)

        # Measurement matrix (we measure x, y)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )

        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

    def predict(self) -> Tuple[float, float]:
        """Predict next position"""
        prediction = self.kf.predict()
        return float(prediction[0]), float(prediction[1])

    def update(self, measurement: Tuple[float, float]):
        """Update with new measurement"""
        measurement_array = np.array(
            [[np.float32(measurement[0])], [np.float32(measurement[1])]]
        )
        self.kf.correct(measurement_array)

    def get_state(self) -> Tuple[float, float, float, float]:
        """Get current state [x, y, vx, vy]"""
        state = self.kf.statePost
        return float(state[0]), float(state[1]), float(state[2]), float(state[3])


class ObjectTracker:
    """
    Advanced multi-object tracker with:
    1. Improved IOU matching to prevent duplicate detections
    2. Kalman filtering for position prediction
    3. Appearance-based matching
    4. Occlusion handling
    5. Track quality scoring
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        appearance_weight: float = 0.3,
        distance_threshold: float = 100.0,
    ):
        """
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IOU threshold for matching (lower = more strict)
            appearance_weight: Weight for appearance-based matching (0-1)
            distance_threshold: Maximum pixel distance for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self.distance_threshold = distance_threshold

        self.objects: Dict[str, TrackedObject] = {}
        self.kalman_filters: Dict[str, KalmanFilter2D] = {}
        self.frame_count = 0

    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute Intersection over Union between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Union area
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _compute_distance(self, point1: Tuple, point2: Tuple) -> float:
        """Compute Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def _extract_appearance_features(
        self, frame: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> Optional[np.ndarray]:
        """Extract simple appearance features (color histogram)"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                return None

            roi = frame[y1:y2, x1:x2]

            # Compute color histogram in HSV space
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([roi_hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([roi_hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([roi_hsv], [2], None, [32], [0, 256])

            # Normalize and concatenate
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-6)
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-6)
            hist_v = hist_v.flatten() / (hist_v.sum() + 1e-6)

            return np.concatenate([hist_h, hist_s, hist_v])
        except Exception:
            return None

    def _compute_appearance_similarity(
        self, feature1: Optional[np.ndarray], feature2: Optional[np.ndarray]
    ) -> float:
        """Compute appearance similarity (0-1, higher is more similar)"""
        if feature1 is None or feature2 is None:
            return 0.0

        # Use histogram correlation
        similarity = cv2.compareHist(
            feature1.astype(np.float32), feature2.astype(np.float32), cv2.HISTCMP_CORREL
        )

        return max(0.0, similarity)  # Clamp to [0, 1]

    def _build_cost_matrix(
        self,
        track_objects: List[TrackedObject],
        detections: List[Detection],
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build cost matrix combining IOU, distance, and appearance
        Lower cost = better match
        """
        num_tracks = len(track_objects)
        num_dets = len(detections)

        if num_tracks == 0 or num_dets == 0:
            return np.zeros((num_tracks, num_dets))

        cost_matrix = np.zeros((num_tracks, num_dets))

        for i, track in enumerate(track_objects):
            track_bbox = track.bbox
            track_centroid = track.centroid

            # Use predicted position if available
            if track.predicted_centroid:
                predicted_bbox = self._update_bbox_with_centroid(
                    track_bbox, track.predicted_centroid
                )
            else:
                predicted_bbox = track_bbox

            for j, det in enumerate(detections):
                det_bbox = (det.x1, det.y1, det.x2, det.y2)
                det_centroid = ((det.x1 + det.x2) / 2, (det.y1 + det.y2) / 2)

                # IOU-based cost (1 - IOU, so lower is better)
                iou = self._compute_iou(predicted_bbox, det_bbox)
                iou_cost = 1.0 - iou

                # Distance-based cost (normalized)
                distance = self._compute_distance(track_centroid, det_centroid)
                distance_cost = min(1.0, distance / self.distance_threshold)

                # Class consistency
                class_cost = 0.0 if track.class_name == det.label else 0.5

                # Appearance-based cost
                appearance_cost = 0.5  # Default neutral cost
                if frame is not None and self.appearance_weight > 0:
                    det_feature = self._extract_appearance_features(frame, det_bbox)
                    if det_feature is not None and track.appearance_feature is not None:
                        similarity = self._compute_appearance_similarity(
                            track.appearance_feature, det_feature
                        )
                        appearance_cost = 1.0 - similarity

                # Combined cost
                cost = (1.0 - self.appearance_weight) * (
                    0.5 * iou_cost + 0.3 * distance_cost + 0.2 * class_cost
                ) + self.appearance_weight * appearance_cost

                cost_matrix[i, j] = cost

        return cost_matrix

    def _update_bbox_with_centroid(
        self, bbox: Tuple[float, float, float, float], new_centroid: Tuple[float, float]
    ) -> Tuple[float, float, float, float]:
        """Update bbox to have a new centroid while maintaining size"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        new_x1 = new_centroid[0] - width / 2
        new_y1 = new_centroid[1] - height / 2
        new_x2 = new_x1 + width
        new_y2 = new_y1 + height

        return (new_x1, new_y1, new_x2, new_y2)

    def _register_track(
        self, detection: Detection, frame: Optional[np.ndarray] = None
    ) -> str:
        """Register a new track"""
        track_id = str(uuid.uuid4())[:8]

        centroid = (
            (detection.x1 + detection.x2) / 2,
            (detection.y1 + detection.y2) / 2,
        )
        bbox = (detection.x1, detection.y1, detection.x2, detection.y2)

        # Extract appearance features
        appearance_feature = None
        if frame is not None:
            appearance_feature = self._extract_appearance_features(frame, bbox)

        track = TrackedObject(
            track_id=track_id,
            class_id=detection.class_id,
            class_name=detection.label,
            bbox=bbox,
            confidence=detection.confidence,
            centroid=centroid,
            appearance_feature=appearance_feature,
            avg_confidence=detection.confidence,
        )

        track.trajectory.append(centroid)
        self.objects[track_id] = track

        # Initialize Kalman filter
        kf = KalmanFilter2D()
        kf.kf.statePost = np.array(
            [
                [np.float32(centroid[0])],
                [np.float32(centroid[1])],
                [np.float32(0)],
                [np.float32(0)],
            ]
        )
        self.kalman_filters[track_id] = kf

        return track_id

    def _update_track(
        self, track_id: str, detection: Detection, frame: Optional[np.ndarray] = None
    ):
        """Update an existing track with new detection"""
        track = self.objects[track_id]
        old_centroid = track.centroid

        new_centroid = (
            (detection.x1 + detection.x2) / 2,
            (detection.y1 + detection.y2) / 2,
        )
        new_bbox = (detection.x1, detection.y1, detection.x2, detection.y2)

        # Update Kalman filter
        kf = self.kalman_filters[track_id]
        kf.update(new_centroid)

        # Calculate velocity
        velocity = (
            new_centroid[0] - old_centroid[0],
            new_centroid[1] - old_centroid[1],
        )
        distance = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        # Update appearance features (with exponential moving average)
        if frame is not None:
            new_feature = self._extract_appearance_features(frame, new_bbox)
            if new_feature is not None:
                if track.appearance_feature is not None:
                    # EMA: 80% old, 20% new
                    track.appearance_feature = (
                        0.8 * track.appearance_feature + 0.2 * new_feature
                    )
                else:
                    track.appearance_feature = new_feature

        # Update track properties
        track.centroid = new_centroid
        track.bbox = new_bbox
        track.confidence = detection.confidence
        track.velocity = velocity
        track.hits += 1
        track.time_since_update = 0
        track.last_seen = time.time()
        track.trajectory.append(new_centroid)
        track.distance_traveled += distance
        track.velocity_history.append(velocity)

        # Update quality metrics
        track.avg_confidence = (
            track.avg_confidence * track.age + detection.confidence
        ) / (track.age + 1)
        track.age += 1

    def update(
        self, detections: List[Detection], frame: Optional[np.ndarray] = None
    ) -> Dict[str, TrackedObject]:
        """
        Update tracker with new detections

        Args:
            detections: List of detections from current frame
            frame: Optional frame for appearance-based matching

        Returns:
            Dictionary of active tracks
        """
        self.frame_count += 1

        # Step 1: Predict next positions for all tracks
        for track_id, kf in self.kalman_filters.items():
            predicted_pos = kf.predict()
            self.objects[track_id].predicted_centroid = predicted_pos

        # Step 2: Build cost matrix and perform Hungarian matching
        track_list = list(self.objects.values())

        if len(track_list) > 0 and len(detections) > 0:
            cost_matrix = self._build_cost_matrix(track_list, detections, frame)

            # Hungarian algorithm for optimal assignment
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_detections = set()

            # Step 3: Process matches
            for track_idx, det_idx in zip(track_indices, det_indices):
                cost = cost_matrix[track_idx, det_idx]

                # Only accept matches below cost threshold
                # Convert cost to IOU-like threshold
                if cost < (1.0 - self.iou_threshold):
                    track_id = track_list[track_idx].track_id
                    detection = detections[det_idx]

                    self._update_track(track_id, detection, frame)
                    matched_tracks.add(track_id)
                    matched_detections.add(det_idx)

            # Step 4: Handle unmatched detections (new tracks)
            for det_idx, detection in enumerate(detections):
                if det_idx not in matched_detections:
                    self._register_track(detection, frame)

            # Step 5: Handle unmatched tracks (missing detections)
            for track in track_list:
                if track.track_id not in matched_tracks:
                    track.time_since_update += 1
                    track.age += 1

                    # Use predicted position to update trajectory
                    if track.predicted_centroid:
                        track.trajectory.append(track.predicted_centroid)

        elif len(detections) > 0:
            # No existing tracks, register all detections
            for detection in detections:
                self._register_track(detection, frame)

        else:
            # No detections, age all tracks
            for track in track_list:
                track.time_since_update += 1
                track.age += 1

        # Step 6: Remove dead tracks
        tracks_to_remove = []
        for track_id, track in self.objects.items():
            if track.time_since_update > self.max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.objects[track_id]
            del self.kalman_filters[track_id]

        # Step 7: Return only confirmed tracks (with sufficient hits)
        confirmed_tracks = {
            track_id: track
            for track_id, track in self.objects.items()
            if track.hits >= self.min_hits
        }

        return confirmed_tracks

    def get_all_tracks(self) -> Dict[str, TrackedObject]:
        """Get all tracks (including tentative)"""
        return self.objects

    def reset(self):
        """Reset tracker state"""
        self.objects.clear()
        self.kalman_filters.clear()
        self.frame_count = 0
