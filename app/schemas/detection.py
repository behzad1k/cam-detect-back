from pydantic import BaseModel
from typing import List, Optional, Dict


class Detection(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str


class ModelResult(BaseModel):
    detections: List[Detection]
    count: int
    model: str
    error: Optional[str] = None


class DetectionResponse(BaseModel):
    camera_id: str
    timestamp: int
    results: Dict[str, ModelResult]
    type: str = "detections"