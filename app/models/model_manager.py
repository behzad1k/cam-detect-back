import torch
import logging
from typing import Dict, List, Optional, Set
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2
from functools import lru_cache

from app.core.config import settings
from app.core.schemas import Detection, ModelResult

logger = logging.getLogger(__name__)


class ModelManager:
  def __init__(self):
    self.models: Dict[str, YOLO] = {}
    self.model_classes: Dict[str, Dict[int, str]] = {}
    # New performance optimization attributes
    self.device = self._get_optimal_device()
    self.model_configs: Dict[str, Dict] = {}

  def _get_optimal_device(self) -> str:
    """Determine the best device for inference"""
    if hasattr(settings, 'DEVICE'):
      return settings.DEVICE

    if torch.cuda.is_available():
      gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
      logger.info(f"GPU memory available: {gpu_memory:.1f}GB")
      return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
      return 'mps'
    else:
      return 'cpu'

  def _apply_performance_optimizations(self, model: YOLO, model_name: str) -> None:
    """Apply performance optimizations to loaded model"""
    try:
      # Move model to optimal device
      model.to(self.device)

      # Apply GPU-specific optimizations
      if self.device == 'cuda':
        # Enable mixed precision for faster inference
        if hasattr(model.model, 'half'):
          model.model.half()
        torch.backends.cudnn.benchmark = True

      # Model-specific optimizations for large models
      if any(keyword in model_name.lower() for keyword in ['80', 'coco', 'large']):
        # For models with many classes, increase confidence to reduce noise
        model.conf = max(settings.CONFIDENCE_THRESHOLD, 0.35)
        # Reduce max detections for speed
        model.max_det = min(settings.MAX_DETECTIONS, 100)
        logger.info(f"Applied large model optimizations to {model_name}")

      # Warm up the model with a dummy inference
      dummy_img = torch.zeros((640, 640, 3), dtype=torch.uint8).numpy()
      with torch.no_grad():
        model(dummy_img, verbose=False)

    except Exception as e:
      logger.warning(f"Could not apply all optimizations to {model_name}: {e}")

  def load_model(self, model_name: str) -> bool:
    """Load a model by name with performance optimizations"""
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

      # Apply performance optimizations
      self._apply_performance_optimizations(model, model_name)

      self.models[model_name] = model

      # Store class names for this model
      if hasattr(model, 'names') and model.names:
        self.model_classes[model_name] = model.names

      # Store model configuration for optimization reference
      self.model_configs[model_name] = {
        'classes': len(model.names) if model.names else 0,
        'device': str(self.device)
      }

      logger.info(f"Model {model_name} loaded successfully on {self.device}")
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

  @lru_cache(maxsize=128)
  def _get_class_indices_for_filter(self, model_name: str, class_filter_tuple: tuple) -> Set[int]:
    """Cache class name to index mapping for performance"""
    if model_name not in self.model_classes:
      return set()

    model_classes = self.model_classes[model_name]
    allowed_indices = set()

    for class_name in class_filter_tuple:
      for class_id, name in model_classes.items():
        if name.lower() == class_name.lower():
          allowed_indices.add(class_id)
          break

    return allowed_indices

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
      for result in results:
        detections.extend(self._extract_detections_from_result(result, model_name))
      if hasattr(results, 'boxes'):
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
    """Extract detections from a single YOLO result with optimizations"""
    detections = []
    if not (hasattr(result, 'boxes') and result.boxes is not None):
      return detections

    boxes = result.boxes
    if len(boxes) == 0:
      return detections

    # Performance optimization: filter on GPU before moving to CPU
    try:
      # Apply confidence filtering on GPU if possible
      device = boxes.xyxy.device
      conf_mask = boxes.conf >= settings.CONFIDENCE_THRESHOLD

      if not conf_mask.any():
        return detections

      # Filter all tensors at once
      filtered_xyxy = boxes.xyxy[conf_mask]
      filtered_conf = boxes.conf[conf_mask]
      filtered_cls = boxes.cls[conf_mask]

      # Move to CPU only once
      xyxy = filtered_xyxy.cpu().numpy()
      conf = filtered_conf.cpu().numpy()
      cls = filtered_cls.cpu().numpy()

    except Exception:
      # Fallback to original method if GPU optimization fails
      xyxy = boxes.xyxy.cpu().numpy()
      conf = boxes.conf.cpu().numpy()
      cls = boxes.cls.cpu().numpy()

      # Apply confidence filtering on CPU
      valid_indices = conf >= settings.CONFIDENCE_THRESHOLD
      xyxy = xyxy[valid_indices]
      conf = conf[valid_indices]
      cls = cls[valid_indices]

    for i in range(len(xyxy)):
      confidence = float(conf[i])
      class_id = int(cls[i])
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
      # Performance optimization: use torch.no_grad() for inference
      with torch.no_grad():
        model_results = self.models[model_name](frame, verbose=False)
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

  def load_all_models(self) -> Dict[str, bool]:
    """Load all models defined in settings.MODEL_PATHS"""
    results = {}
    for model_name in settings.MODEL_PATHS:
      try:
        success = self.load_model(model_name)
        results[model_name] = success
        if success:
          logger.info(f"Successfully loaded model {model_name}")
        else:
          logger.error(f"Failed to load model {model_name}")
      except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}", exc_info=True)
        results[model_name] = False
    return results

  # New optimization methods that can be called optionally
  def optimize_for_speed(self, model_name: str) -> bool:
    """Apply additional speed optimizations to a specific model"""
    if model_name not in self.models:
      return False

    try:
      model = self.models[model_name]
      # Increase confidence threshold to reduce processing
      model.conf = max(model.conf, 0.4)
      # Reduce max detections
      model.max_det = min(model.max_det, 50)
      # Reduce image size for faster processing
      if hasattr(model, 'imgsz'):
        model.imgsz = 416  # Smaller input size

      logger.info(f"Applied speed optimizations to {model_name}")
      return True
    except Exception as e:
      logger.error(f"Error optimizing {model_name}: {e}")
      return False

  def get_model_performance_stats(self) -> Dict[str, Dict]:
    """Get performance statistics for loaded models"""
    stats = {}
    for model_name, model in self.models.items():
      config = self.model_configs.get(model_name, {})
      stats[model_name] = {
        'device': config.get('device', 'unknown'),
        'classes': config.get('classes', 0),
        'confidence_threshold': model.conf,
        'max_detections': model.max_det,
        'iou_threshold': model.iou
      }
    return stats

  def cleanup_gpu_memory(self):
    """Clean up GPU memory if using CUDA"""
    if self.device == 'cuda':
      torch.cuda.empty_cache()
      logger.info("GPU memory cache cleared")


# Global model manager instance
model_manager = ModelManager()