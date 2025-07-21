from typing import List, Dict, Optional, Any
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
