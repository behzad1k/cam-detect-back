# app/websocket/enhanced_handlers.py (Update your existing handlers)
import base64
import io
from PIL import Image
from fastapi import WebSocket, WebSocketDisconnect
from .tracking_handlers import TrackingWebSocketHandler
import json
from typing import Dict, List, Any, Optional
import numpy as np
import asyncio

class EnhancedWebSocketHandler:
  """Enhanced WebSocket handler with tracking integration"""

  def __init__(self):
    self.tracking_handler = TrackingWebSocketHandler()
    self.detection_results_cache = {}

  async def handle_message(self, websocket: WebSocket, message: str, stream_id: str):
    """Enhanced message handler with tracking support"""
    try:
      # Try to parse as JSON (control message)
      data = json.loads(message)

      # Check if it's a tracking message
      if data.get('type', '').startswith('tracking_') or data.get('type') in [
        'configure_tracking', 'start_tracking', 'stop_tracking',
        'get_tracker_stats', 'define_zone'
      ]:
        await self.tracking_handler.handle_tracking_message(websocket, data, stream_id)
      else:
        # Handle other message types (existing functionality)
        await self._handle_regular_message(websocket, data, stream_id)

    except json.JSONDecodeError:
      # Binary message (image data)
      await self._handle_binary_message(websocket, message, stream_id)

  async def _handle_binary_message(self, websocket: WebSocket, binary_data: bytes, stream_id: str):
    """Handle binary image data with integrated tracking"""
    try:
      # Parse binary data (your existing format)
      detections = await self._process_image_detections(binary_data)

      # Check if tracking is enabled for this stream
      if self.tracking_handler.tracking_enabled.get(stream_id, False):
        # Send detections to tracker
        tracking_message = {
          'type': 'update_detections',
          'detections': detections
        }
        await self.tracking_handler.handle_tracking_message(websocket, tracking_message, stream_id)
      else:
        # Send regular detection results
        await self._send_detection_results(websocket, stream_id, detections)

    except Exception as e:
      await self._send_error(websocket, f"Error processing binary message: {str(e)}")

  async def _process_image_detections(self, binary_data: bytes) -> List[Dict]:
    """Process image and return detection results"""
    # This would integrate with your existing detection logic
    # For now, returning mock structure - replace with your actual detection processing

    # Parse your existing binary format
    offset = 0
    timestamp = int.from_bytes(binary_data[offset:offset + 4], 'little')
    offset += 4

    model_count = binary_data[offset]
    offset += 1

    models_to_use = []
    for _ in range(model_count):
      # Parse model info (your existing format)
      model_name_length = binary_data[offset]
      offset += 1

      model_name = binary_data[offset:offset + model_name_length].decode('utf-8')
      offset += model_name_length

      has_class_filter = binary_data[offset]
      offset += 1

      class_filter = None
      if has_class_filter:
        class_count = binary_data[offset]
        offset += 1

        class_filter = []
        for _ in range(class_count):
          class_name_length = binary_data[offset]
          offset += 1
          class_name = binary_data[offset:offset + class_name_length].decode('utf-8')
          offset += class_name_length
          class_filter.append(class_name)

      models_to_use.append({
        'name': model_name,
        'class_filter': class_filter
      })

    # Image data
    image_data = binary_data[offset:]

    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_data))
    image_array = np.array(image)

    # Run detection (integrate with your existing model manager)
    all_detections = []

    # This is where you'd call your existing detection logic
    # For example:
    # for model_info in models_to_use:
    #     model_results = your_model_manager.detect(model_info['name'], image_array, model_info['class_filter'])
    #     all_detections.extend(model_results)

    # Mock detection results for demonstration
    mock_detections = [
      {
        'x1': 100, 'y1': 100, 'x2': 200, 'y2': 200,
        'confidence': 0.85, 'class_id': 0, 'label': 'person'
      },
      {
        'x1': 300, 'y1': 150, 'x2': 400, 'y2': 250,
        'confidence': 0.75, 'class_id': 1, 'label': 'vehicle'
      }
    ]

    return mock_detections

  async def _send_detection_results(self, websocket: WebSocket, stream_id: str, detections: List[Dict]):
    """Send detection results without tracking"""
    response = {
      'type': 'detections',
      'stream_id': stream_id,
      'results': {
        'detections': detections,
        'count': len(detections)
      },
      'timestamp': asyncio.get_event_loop().time()
    }

    await websocket.send_text(json.dumps(response))

  async def _handle_regular_message(self, websocket: WebSocket, data: Dict, stream_id: str):
    """Handle non-tracking messages"""
    # Your existing message handling logic
    pass

  async def _send_error(self, websocket: WebSocket, error_message: str):
    """Send error message"""
    await websocket.send_text(json.dumps({
      'type': 'error',
      'message': error_message
    }))

enhanced_handler = EnhancedWebSocketHandler()
