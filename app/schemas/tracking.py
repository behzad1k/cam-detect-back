from pydantic import BaseModel
from typing import Dict, Tuple, List, Optional, Any


class TrackedObject(BaseModel):
    track_id: str
    class_name: str
    class_id: int
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    confidence: float
    age: int
    velocity: Tuple[float, float]
    speed_px_per_sec: float = 0.0
    speed_m_per_sec: Optional[float] = None
    distance_traveled: float = 0.0


class TrackingResponse(BaseModel):
    camera_id: str
    timestamp: int
    tracked_objects: Dict[str, TrackedObject]
    summary: Dict[str, Any]
    type: str = "tracking"


class ZoneDefinition(BaseModel):
    zone_id: str
    polygon_points: List[Tuple[int, int]]


class TrackingConfig(BaseModel):
    max_disappeared: int = 30
    max_distance: float = 100.0
    enable_speed: bool = True
    enable_zones: bool = False