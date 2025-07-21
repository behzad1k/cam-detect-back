import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

from app.core.schemas import ModelResult
from app.models.model_manager import model_manager

logger = logging.getLogger(__name__)


class FrameProcessor:
  @staticmethod
  def parse_websocket_data(data: bytes) -> Tuple[int, List[str], Optional[Dict[str, List[str]]], np.ndarray]:
    """Parse incoming WebSocket data"""
    # First 4 bytes contain the timestamp
    timestamp = int.from_bytes(data[:4], byteorder='big')
    timestamp_ms = timestamp * 1000

    # Next 1 byte contains model count
    model_count = data[4]

    # Parse model names and their class filters
    model_names = []
    class_filters = {}
    position = 5

    for _ in range(model_count):
      # Model name length and name
      name_length = data[position]
      position += 1
      model_name = data[position:position + name_length].decode('utf-8')
      model_names.append(model_name)
      position += name_length

      # Check if there are class filters for this model
      has_class_filter = data[position]
      position += 1

      if has_class_filter:
        # Number of classes to filter
        class_count = data[position]
        position += 1

        model_class_filters = []
        for _ in range(class_count):
          class_name_length = data[position]
          position += 1
          class_name = data[position:position + class_name_length].decode('utf-8')
          model_class_filters.append(class_name)
          position += class_name_length

        class_filters[model_name] = model_class_filters

    # Remaining bytes are the image data
    image_bytes = data[position:]

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
      raise ValueError("Could not decode image")

    return timestamp_ms, model_names, class_filters if class_filters else None, frame

  @staticmethod
  async def process_frame(
      frame: np.ndarray,
      model_names: List[str],
      class_filters: Optional[Dict[str, List[str]]] = None,
      timestamp_ms: int = 0
  ) -> Dict[str, ModelResult]:
    """Process frame with requested models and class filters"""
    results = {}

    for model_name in model_names:
      # Get class filter for this specific model
      model_class_filter = class_filters.get(model_name) if class_filters else None

      # Run inference
      result = model_manager.run_inference(frame, model_name, model_class_filter)
      results[model_name] = result

    return results

  @staticmethod
  def create_response(
      results: Dict[str, ModelResult],
      timestamp_ms: int,
      response_type: str = "detections"
  ) -> dict:
    """Create WebSocket response"""
    return {
      "type": response_type,
      "results": {k: v.dict() for k, v in results.items()},
      "timestamp": timestamp_ms
    }

  @staticmethod
  def create_error_response(error_message: str, timestamp_ms: int = 0) -> dict:
    """Create error response"""
    return {
      "type": "error",
      "message": error_message,
      "timestamp": timestamp_ms
    }


frame_processor = FrameProcessor()
