import torch
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path
from ultralytics import YOLO

from app.core.config import settings
from app.core.schemas import Detection, ModelResult

logger = logging.getLogger(__name__)


class ModelManager:
  def __init__(self):
    self.models: Dict[str, YOLO] = {}
    self.model_classes: Dict[str, Dict[int, str]] = {}  # model_name -> {class_id: class_name}

  def load_model(self, model_name: str) -> bool:
    """Load a model by name"""
    try:
      if model_name in self.models:
        logger.info(f"Model {model_name} already loaded")
        return True

      model_path = settings.MODEL_PATHS.get(model_name)
      if not model_path or not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return False

      logger.info(f"Loading model {model_name} from {model_path}")

      model = YOLO(model_path)
      model.conf = settings.CONFIDENCE_THRESHOLD
      model.iou = settings.IOU_THRESHOLD
      model.max_det = settings.MAX_DETECTIONS

      self.models[model_name] = model

      # Store class names for this model
      if hasattr(model, 'names') and model.names:
        self.model_classes[model_name] = model.names

      logger.info(f"Model {model_name} loaded successfully on {settings.DEVICE}")
      return True

    except ImportError as e:
      logger.error(f"ultralytics not installed: {e}")
      return False
    except Exception as e:
      logger.error(f"Error loading model {model_name}: {e}")
      return False

  def get_model_classes(self, model_name: str) -> Optional[Dict[int, str]]:
    """Get available class names for a model"""
    if model_name in self.model_classes:
      return self.model_classes[model_name]

    if model_name in self.models:
      model = self.models[model_name]
      if hasattr(model, 'names') and model.names:
        self.model_classes[model_name] = model.names
        return model.names

    return None

  def filter_detections_by_class(
      self,
      detections: List[Detection],
      allowed_classes: Set[str]
  ) -> List[Detection]:
    """Filter detections to only include specified class names"""
    if not allowed_classes:
      return detections

    return [
      detection for detection in detections
      if detection.label in allowed_classes
    ]

  def process_detections(
      self,
      results,
      model_name: str,
      class_filter: Optional[List[str]] = None
  ) -> ModelResult:
    """Process YOLO detection results with optional class filtering"""
    try:
      detections = []

      # Convert class filter to set for faster lookup
      allowed_classes = set(class_filter) if class_filter else None

      if isinstance(results, list):
        for result in results:
          detections.extend(self._extract_detections_from_result(result, model_name))
      elif hasattr(results, 'boxes'):
        detections.extend(self._extract_detections_from_result(results, model_name))

      # Apply class filtering if specified
      if allowed_classes:
        detections = self.filter_detections_by_class(detections, allowed_classes)

      logger.info(f"Processed {len(detections)} detections for {model_name}")

      return ModelResult(
        detections=detections,
        count=len(detections),
        model=model_name,
        error=None
      )

    except Exception as e:
      logger.error(f"Error processing detections for {model_name}: {e}")
      return ModelResult(
        detections=[],
        count=0,
        model=model_name,
        error=str(e)
      )

  def _extract_detections_from_result(self, result, model_name: str) -> List[Detection]:
    """Extract detections from a single YOLO result"""
    detections = []

    if not (hasattr(result, 'boxes') and result.boxes is not None):
      return detections

    boxes = result.boxes
    if len(boxes) == 0:
      return detections

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    for i in range(len(boxes)):
      confidence = float(conf[i])
      class_id = int(cls[i])

      if confidence >= settings.CONFIDENCE_THRESHOLD:
        x1, y1, x2, y2 = xyxy[i]

        # Get class name
        class_name = "unknown"
        if hasattr(result, 'names') and result.names:
          class_name = result.names.get(class_id, f"class_{class_id}")

        detections.append(Detection(
          x1=int(x1),
          y1=int(y1),
          x2=int(x2),
          y2=int(y2),
          confidence=confidence,
          class_id=class_id,
          label=class_name
        ))

    return detections

  def run_inference(
      self,
      frame,
      model_name: str,
      class_filter: Optional[List[str]] = None
  ) -> ModelResult:
    """Run inference on a frame with a specific model"""
    if model_name not in self.models:
      if not self.load_model(model_name):
        return ModelResult(
          detections=[],
          count=0,
          model=model_name,
          error=f"Model {model_name} could not be loaded"
        )

    try:
      model_results = self.models[model_name](frame)
      return self.process_detections(model_results, model_name, class_filter)
    except Exception as e:
      logger.error(f"Error running inference with {model_name}: {e}")
      return ModelResult(
        detections=[],
        count=0,
        model=model_name,
        error=str(e)
      )

  def get_loaded_models(self) -> List[str]:
    """Get list of loaded model names"""
    return list(self.models.keys())

  def get_available_models(self) -> List[str]:
    """Get list of available model names"""
    return list(settings.MODEL_PATHS.keys())


# Global model manager instance
model_manager = ModelManager()
