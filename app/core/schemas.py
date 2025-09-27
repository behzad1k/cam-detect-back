from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel

class Detection(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    label: str

class ModelResult(BaseModel):
    detections: List[Detection]
    count: int
    model: str
    error: Optional[str] = None

class ProcessingRequest(BaseModel):
    model_names: List[str]
    class_filters: Optional[Dict[str, List[str]]] = None  # model_name -> list of desired class names
    confidence_threshold: Optional[float] = None

class WebSocketResponse(BaseModel):
    type: str
    results: Optional[Dict[str, ModelResult]] = None
    message: Optional[str] = None
    timestamp: int

class ModelInfo(BaseModel):
    name: str
    loaded: bool
    available_classes: Optional[List[str]] = None
    path: str

class HealthResponse(BaseModel):
    status: str
    device: str
    models_available: List[str]
    models_loaded: List[str]

# app/core/schemas.py (Add tracking-related schemas)
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

class TrackerTypeEnum(str, Enum):
    CENTROID = "centroid"
    KALMAN = "kalman"
    DEEP_SORT = "deep_sort"
    BYTE_TRACK = "byte_track"

class TrackingConfig(BaseModel):
    tracker_type: TrackerTypeEnum = TrackerTypeEnum.CENTROID
    tracker_params: Dict[str, Any] = {}
    speed_config: Optional[Dict[str, Any]] = None
    enable_zones: bool = False
    enable_line_crossing: bool = False

class ZoneDefinition(BaseModel):
    zone_id: str
    polygon_points: List[Tuple[int, int]]
    zone_type: str = "detection"  # detection, counting, restricted

class LineDefinition(BaseModel):
    line_id: str
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    direction: Optional[str] = None  # "both", "left_to_right", "right_to_left"

class TrackingMessage(BaseModel):
    type: str
    stream_id: Optional[str] = None
    config: Optional[TrackingConfig] = None
    zone: Optional[ZoneDefinition] = None
    line: Optional[LineDefinition] = None

class TrackedObjectResponse(BaseModel):
    track_id: str
    class_name: str
    class_id: int
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    confidence: float
    age: int
    hits: int
    time_since_update: int
    velocity: Tuple[float, float]
    speed_info: Dict[str, float]
    trajectory_length: int

class TrackingResponse(BaseModel):
    type: str
    stream_id: str
    timestamp: float
    tracked_objects: Dict[str, TrackedObjectResponse]
    zone_occupancy: Dict[str, List[str]]
    summary: Dict[str, Any]

class CalibrationModeEnum(str, Enum):
    perspective_transform = "perspective_transform"
    reference_object = "reference_object"
    vanishing_point = "vanishing_point"

class CalibrationPointModel(BaseModel):
    pixel_x: float
    pixel_y: float
    real_x: float
    real_y: float

class CalibrationRequest(BaseModel):
    mode: CalibrationModeEnum
    points: List[CalibrationPointModel]
    frame_width: int
    frame_height: int
    reference_width_meters: Optional[float] = None
    reference_height_meters: Optional[float] = None
    vanishing_point: Optional[List[float]] = None
    horizon_line: Optional[float] = None
    reference_distance_meters: Optional[float] = None
    reference_pixel_height: Optional[float] = None

class WebSocketCommand(BaseModel):
    command: str
    data: Optional[Dict[str, Any]] = None

class TrackingSettings(BaseModel):
    enabled: bool = True
    max_disappeared: int = 10
    max_distance: float = 100.0
    show_trails: bool = True
    show_speed: bool = True

class ProcessingSettings(BaseModel):
    calibration_enabled: bool = False
    tracking_enabled: bool = False
    speed_detection_enabled: bool = False
    distance_measurement_enabled: bool = False

# Global handler instance
