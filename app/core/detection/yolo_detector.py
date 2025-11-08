import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Dict
from pathlib import Path
import logging

from app.config import settings
from app.schemas.detection import Detection, ModelResult

logger = logging.getLogger(__name__)


class YOLODetector:
  """YOLO model wrapper for object detection"""

  def __init__(self):
    self.models: Dict[str, YOLO] = {}
    self.model_classes: Dict[str, Dict[int, str]] = {}
    self.device = settings.DEVICE

  def load_model(self, model_name: str) -> bool:
    """Load a YOLO model"""
    if model_name in self.models:
      logger.info(f"Model {model_name} already loaded")
      return True

    if model_name not in settings.AVAILABLE_MODELS:
      logger.error(f"Unknown model: {model_name}")
      return False

    try:
      model_file = settings.AVAILABLE_MODELS[model_name]
      model_path = settings.MODEL_DIR / model_file

      if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False

      model = YOLO(str(model_path))
      model.to(self.device)

      self.models[model_name] = model
      self.model_classes[model_name] = model.names

      logger.info(f"✅ Loaded model: {model_name}")
      return True

    except Exception as e:
      logger.error(f"❌ Failed to load {model_name}: {e}")
      return False

  def detect(
    self,
    image: np.ndarray,
    model_name: str,
    class_filter: Optional[List[str]] = None
  ) -> ModelResult:
    """Run detection on an image"""
    try:
      # Load model if not loaded
      if model_name not in self.models:
        if not self.load_model(model_name):
          return ModelResult(
            detections=[],
            count=0,
            model=model_name,
            error="Failed to load model"
          )

      model = self.models[model_name]

      # Run inference
      results = model.predict(
        image,
        conf=settings.CONFIDENCE_THRESHOLD,
        iou=settings.IOU_THRESHOLD,
        max_det=settings.MAX_DETECTIONS,
        verbose=False
      )[0]

      # Parse results
      detections = []
      boxes = results.boxes

      if boxes is not None and len(boxes) > 0:
        for box in boxes:
          x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
          conf = float(box.conf[0])
          cls_id = int(box.cls[0])
          label = self.model_classes[model_name].get(cls_id, f"class_{cls_id}")

          # Apply class filter
          if class_filter and label not in class_filter:
            continue

          detections.append(Detection(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            confidence=conf,
            class_id=cls_id,
            label=label
          ))

      return ModelResult(
        detections=detections,
        count=len(detections),
        model=model_name,
        error=None
      )

    except Exception as e:
      logger.error(f"Detection error for {model_name}: {e}")
      return ModelResult(
        detections=[],
        count=0,
        model=model_name,
        error=str(e)
      )

  def get_available_classes(self, model_name: str) -> Optional[Dict[int, str]]:
    """Get available classes for a model"""
    if model_name not in self.models:
      self.load_model(model_name)
    return self.model_classes.get(model_name)


# Global detector instance
detector = YOLODetector()