import cv2
import torch
import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Set
from pathlib import Path
from ultralytics import YOLO

from app.core.config import settings
from app.core.schemas import Detection, ModelResult

logger = logging.getLogger(__name__)


class ModelManager:
  def __init__(self):
    self.models: Dict[str, ort.InferenceSession] = {}
    self.model_classes: Dict[str, Dict[int, str]] = {}  # model_name -> {class_id: class_name}
    self.model_input_shapes: Dict[str, tuple] = {}  # Store input shapes for each model
    self.model_input_names: Dict[str, str] = {}  # Store input names for each model

  def load_model(self, model_name: str) -> bool:
    """Load a model by name, converting to ONNX if needed"""
    try:
      if model_name in self.models:
        logger.info(f"Model {model_name} already loaded")
        return True

      model_path = settings.MODEL_PATHS.get(model_name)
      if not model_path or not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return False

      logger.info(f"Loading model {model_name} from {model_path}")

      # Create ONNX Runtime session
      so = ort.SessionOptions()
      so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

      session = ort.InferenceSession(str(model_path), so, providers=['CPUExecutionProvider'])

      # Store model metadata - ensure we get integers for height/width
      input_shape = session.get_inputs()[0].shape
      self.models[model_name] = session
      self.model_input_shapes[model_name] = tuple(int(dim) if isinstance(dim, str) else dim for dim in input_shape)
      self.model_input_names[model_name] = session.get_inputs()[0].name

      # Load class names from a separate file or configuration
      # This assumes you have a way to get class names without loading the original model

      if model_name in settings.MODEL_CLASSES:
        self.model_classes[model_name] = settings.MODEL_CLASSES[model_name]
      else:
        logger.warning(f"No class names found for model {model_name}")
        self.model_classes[model_name] = {}

        logger.info(f"Model {model_name} loaded successfully in ONNX format. Input shape: {input_shape}")
        return True

    except Exception as e:
      logger.error(f"Error loading model {model_name}: {e}")
      return False
  def get_model_classes(self, model_name: str) -> Optional[Dict[int, str]]:
    """Get available class names for a model"""
    if model_name in self.model_classes:
      return self.model_classes[model_name]
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
      results: np.ndarray,
      model_name: str,
      class_filter: Optional[List[str]] = None
  ) -> ModelResult:
    """Process detection results with optional class filtering"""
    try:
      detections = []
      allowed_classes = set(class_filter) if class_filter else None

      # Check if we have any detections
      if results.shape[1] == 0:  # No detections
        return ModelResult(
          detections=[],
          count=0,
          model=model_name,
          error=None
        )

      # Handle different output formats
      for detection in results[0]:
        # Different YOLO versions/models may have different output formats
        if detection.shape[0] >= 6:  # Standard format [x1, y1, x2, y2, conf, class, ...]
          x1, y1, x2, y2, confidence, class_id = detection[:6]
        elif detection.shape[0] == 5:  # Some models output [x1, y1, x2, y2, conf]
          x1, y1, x2, y2, confidence = detection
          class_id = 0  # Default class if not provided
        else:
          logger.warning(f"Unexpected detection format: {detection.shape}")
          continue

        confidence = float(confidence)
        class_id = int(class_id)

        if confidence >= settings.CONFIDENCE_THRESHOLD:
          # Get class name
          class_name = "unknown"
          if model_name in self.model_classes:
            class_name = self.model_classes[model_name].get(class_id, f"class_{class_id}")

          detections.append(Detection(
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            confidence=confidence,
            class_id=class_id,
            label=class_name
          ))

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

  def run_inference(
      self,
      frame,
      model_name: str,
      class_filter: Optional[List[str]] = None
  ) -> ModelResult:
    """Run inference on a frame with a specific model using ONNX Runtime"""
    if model_name not in self.models:
      if not self.load_model(model_name):
        return ModelResult(
          detections=[],
          count=0,
          model=model_name,
          error=f"Model {model_name} could not be loaded"
        )

    try:
      # Get model input requirements
      session = self.models[model_name]
      input_name = self.model_input_names[model_name]
      input_shape = self.model_input_shapes[model_name]

      # Ensure we have proper dimensions
      # if len(input_shape) != 4:
      #   raise ValueError(f"Unexpected input shape: {input_shape}. Expected NCHW format")
      #
      _, _, model_height, model_width = input_shape
      # model_height = int(model_height)
      # model_width = int(model_width)

      logger.debug(f"Resizing to: {model_width}x{model_height}")

      # Convert frame to numpy array if it's a torch tensor
      if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()

      # Ensure we have a 3-channel image in HWC format
      if len(frame.shape) == 2:  # Grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
      elif frame.shape[0] == 3:  # CHW format
        frame = frame.transpose(1, 2, 0)

      # Verify image dimensions and type
      if len(frame.shape) != 3 or frame.shape[2] != 3:
        raise ValueError(f"Invalid image shape: {frame.shape}. Expected HWC format with 3 channels")

      # Resize frame
      resized_frame = cv2.resize(
        frame,
        (model_width, model_height),
        interpolation=cv2.INTER_LINEAR
      )

      # Normalize and convert to CHW format
      normalized_frame = resized_frame.astype(np.float32) / 255.0
      input_array = normalized_frame.transpose(2, 0, 1)  # HWC to CHW
      input_array = np.expand_dims(input_array, axis=0)  # Add batch dimension

      # Run inference
      outputs = session.run(None, {input_name: input_array})

      # Debug output
      logger.debug(f"Model outputs shapes: {[out.shape for out in outputs]}")
      if len(outputs[0]) > 0:
        logger.debug(f"First detection values: {outputs[0][0]}")

      # Handle different output formats
      if len(outputs) == 1:
        detections = outputs[0]  # Standard YOLO output format
      elif len(outputs) >= 3:
        # Some models output separate tensors for boxes, scores, classes
        boxes = outputs[0]
        scores = outputs[1]
        class_ids = outputs[2]

        # Combine into [x1, y1, x2, y2, conf, class_id] format
        detections = np.concatenate([
          boxes,
          scores.reshape(-1, 1),
          class_ids.reshape(-1, 1)
        ], axis=1)
        detections = np.expand_dims(detections, axis=0)  # Add batch dim
      else:
        raise ValueError(f"Unexpected output format with {len(outputs)} outputs")

      # Scale bounding boxes back to original frame dimensions if we have detections
      if detections.shape[1] > 0:
        x_scale = frame.shape[1] / model_width
        y_scale = frame.shape[0] / model_height
        detections[0, :, [0, 2]] *= x_scale  # x1, x2
        detections[0, :, [1, 3]] *= y_scale  # y1, y2

      return self.process_detections(detections, model_name, class_filter)

    except Exception as e:
      logger.error(f"Error running inference with {model_name}: {str(e)}", exc_info=True)
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