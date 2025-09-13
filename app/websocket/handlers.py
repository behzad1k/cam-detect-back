# app/websocket/unified_handlers.py
import json
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from io import BytesIO
from PIL import Image
import numpy as np

from ..core.schemas import Detection
from ..models import model_manager
from ..utils import logging as utils_logging
from ..tracking.tracker_manager import tracker_manager
from ..tracking.speed_calculator import speed_calculator
from ..tracking.analytics import analytics
logger = logging.getLogger(__name__)


class UnifiedWebSocketHandler:
  """Unified WebSocket handler with tracking integration and detection fixes"""

  def __init__(self):
    self.active_connections: Dict[str, WebSocket] = {}
    self.tracking_enabled: Dict[str, bool] = {}
    self.stream_configs: Dict[str, Dict] = {}
    self.detection_results_cache = {}

    # Initialize tracking components
    try:
      self.tracker_manager = tracker_manager
      self.speed_calculator = speed_calculator
      self.analytics = analytics
      self.tracking_available = True
      logger.info("✅ Tracking components initialized")
    except ImportError as e:
      logger.warning(f"⚠️ Tracking not available: NOOOOO {e}")
      self.tracking_available = False

  async def handle_websocket(self, websocket: WebSocket):
    """Main WebSocket connection handler"""
    await websocket.accept()
    connection_id = f"stream_{int(time.time() * 1000)}"
    self.active_connections[connection_id] = websocket

    logger.info(f"✅ WebSocket connected: {connection_id}")

    try:
      while True:
        message = await websocket.receive()

        if "text" in message:
          # Handle JSON control messages
          await self._handle_text_message(websocket, message["text"], connection_id)
        elif "bytes" in message:
          # Handle binary image data
          await self._handle_binary_message(websocket, message["bytes"], connection_id)
        else:
          logger.warning(f"Unknown message format: {list(message.keys())}")

    except WebSocketDisconnect:
      logger.info(f"🔌 WebSocket disconnected: {connection_id}")
    except Exception as e:
      logger.error(f"❌ WebSocket error: {e}")
      await self._send_error(websocket, f"Server error: {str(e)}")
    finally:
      self._cleanup_connection(connection_id)

  async def _handle_text_message(self, websocket: WebSocket, text_data: str, connection_id: str):
    """Handle JSON control messages (tracking and regular)"""
    try:
      data = json.loads(text_data)
      message_type = data.get("type")

      logger.info(f"🎛️ Handling control message: {message_type}")

      # Tracking messages
      if (
        message_type.startswith('tracking_') or message_type in [
        'configure_tracking', 'start_tracking', 'stop_tracking',
        'get_tracker_stats', 'define_zone'
      ]
      ):
        await self._handle_tracking_message(websocket, data, connection_id)

      # Regular control messages
      elif message_type == "configure_detection":
        await self._configure_detection(websocket, data, connection_id)
      elif message_type == "get_models":
        await self._get_models(websocket)
      else:
        logger.warning(f"❓ Unknown message type: {message_type}")
        await self._send_error(websocket, f"Unknown message type: {message_type}")

    except json.JSONDecodeError as e:
      logger.error(f"❌ Invalid JSON: {e}")
      await self._send_error(websocket, "Invalid JSON format")
    except Exception as e:
      logger.error(f"❌ Error handling text message: {e}")
      await self._send_error(websocket, f"Error processing message: {str(e)}")

  async def _handle_binary_message(self, websocket: WebSocket, binary_data: bytes, connection_id: str):
    """Handle binary image data - UNIFIED AND FIXED"""
    try:
      logger.info(f"📸 Processing frame: {len(binary_data)} bytes")

      # Parse binary data
      models_to_use, image_array = self._parse_binary_data(binary_data)

      if not models_to_use or image_array is None:
        await self._send_error(websocket, "Failed to parse binary data")
        return

      logger.info(f"🔍 Processing {len(models_to_use)} models, image: {image_array.shape}")

      # FIXED: Process with your existing model manager using correct format
      detection_results = await self._process_detections_fixed(models_to_use, image_array)

      # Check if tracking is enabled
      if self.tracking_available and self.tracking_enabled.get(connection_id, False):
        tracking_results = await self._process_with_tracking(detection_results, connection_id)
        await self._send_tracking_results(websocket, tracking_results, connection_id)
      else:
        # Send regular detection results
        await self._send_detection_results(websocket, detection_results, connection_id)

    except Exception as e:

      logger.error(f"❌ Binary processing error: {e}")
      await self._send_error(websocket, f"Frame processing error: {str(e)}")

  def _parse_binary_data(self, binary_data: bytes) -> tuple[List[Dict], np.ndarray]:
    """Parse binary data according to your format"""
    try:
      offset = 0

      # Read timestamp (4 bytes, little endian)
      timestamp = int.from_bytes(binary_data[offset:offset + 4], 'little')
      offset += 4

      # Read model count (1 byte)
      model_count = binary_data[offset]
      offset += 1

      logger.debug(f"⏰ Timestamp: {timestamp}, 🧠 Model count: {model_count}")

      # Parse models
      models_to_use = []
      for i in range(model_count):
        # Model name length and name
        name_length = binary_data[offset]
        offset += 1

        model_name = binary_data[offset:offset + name_length].decode('utf-8')
        offset += name_length

        # Class filter flag
        has_filter = binary_data[offset]
        offset += 1

        class_filter = None
        if has_filter:
          class_count = binary_data[offset]
          offset += 1

          class_filter = []
          for j in range(class_count):
            class_name_length = binary_data[offset]
            offset += 1
            class_name = binary_data[offset:offset + class_name_length].decode('utf-8')
            offset += class_name_length
            class_filter.append(class_name)

        models_to_use.append({
          'name': model_name,
          'class_filter': class_filter
        })

        logger.debug(f"  📋 Model {i + 1}: {model_name}, Filter: {class_filter}")

      # Image data (remaining bytes)
      image_data = binary_data[offset:]

      # Convert to numpy array
      image_io = BytesIO(image_data)
      pil_image = Image.open(image_io)

      # Ensure RGB format for YOLO
      if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

      image_array = np.array(pil_image)

      logger.debug(f"📷 Image converted: {image_array.shape}, dtype: {image_array.dtype}")

      return models_to_use, image_array

    except Exception as e:
      logger.error(f"❌ Binary parsing error: {e}")
      return [], None

  async def _process_detections_fixed(self, models_to_use: List[Dict], image_array: np.ndarray) -> Dict[str, Any]:
    """FIXED: Process detections with proper dictionary structure"""

    # CRITICAL FIX: Use dictionary, not list!
    detection_results = {}  # Dictionary for proper JSON structure

    for model_config in models_to_use:
      model_name = model_config['name']
      class_filter = model_config['class_filter']

      try:
        logger.info(f"🔍 Processing model: {model_name} with filter: {class_filter}")

        # FIXED: Call your model manager with correct parameters
        # Try different method names based on your model manager interface
        detections_response = None

        detections_response = model_manager.model_manager.run_inference(image_array, model_name, class_filter)

        # FIXED: Handle response properly based on your model manager's return format

        if detections_response is None:
          formatted_detections = []
          count = 0
          error = "No response from model"
        else:
          # Your model manager returns a dict like {"detections": [...], "count": N}
          formatted_detections = detections_response.detections
          count = detections_response.count
          error = detections_response.error

        # elif hasattr(detections_response, 'detections'):
        #   # Your model manager returns a Pydantic model or object
        #   formatted_detections = getattr(detections_response, 'detections', [])
        #   count = getattr(detections_response, 'count', len(formatted_detections))
        #   error = getattr(detections_response, 'error', None)

        # else:
        #   logger.warning(f"⚠️ Unexpected response type: {type(detections_response)}")
        #   formatted_detections = []
        #   count = 0
        #   error = f"Unexpected response type: {type(detections_response)}"

        # FIXED: Store as proper dictionary structure
        detection_results[model_name] = {
          "detections": formatted_detections,
          "count": count,
          "model": model_name,
          "error": error
        }

        logger.info(f"✅ {model_name}: {count} detections")

      except Exception as model_error:
        logger.error(f"❌ Model {model_name} error: {model_error}")
        detection_results[model_name] = {
          "detections": [],
          "count": 0,
          "model": model_name,
          "error": str(model_error)
        }

    return detection_results

  async def _process_with_tracking(self, detection_results: Dict[str, Any], connection_id: str) -> Dict[str, Any]:
    """Process detections with tracking if available"""
    try:
      if not self.tracking_enabled.get(connection_id, False):
        # Tracking not enabled for this connection
        return {
          "message": "Tracking not enabled",
          "detection_results": detection_results
        }

      if self.tracker_manager and self.speed_calculator:
        # Full tracking is available
        from ..tracking.tracker_manager import Detection
        all_detections = []
        for model_name, model_result in detection_results.items():
          for det in model_result.get('detections', []):
            detection = Detection(
              x1=float(det.x1),
              y1=float(det.y1),
              x2=float(det.x2),
              y2=float(det.y2),
              confidence=float(det.confidence),
              class_id=int(det.class_id),
              class_name=str(det.label)
            )
            all_detections.append(detection)

        # Update tracker
        tracked_objects = self.tracker_manager.update_tracker(connection_id, all_detections)

        # Build full tracking response
        return self._build_tracking_response(tracked_objects)

      elif hasattr(self, 'tracking_handler'):
        # Use basic tracking handler
        return {"message": "Basic tracking processed", "detection_results": detection_results}

      else:
        # Mock tracking response for development
        mock_tracking_results = {
          'tracked_objects': {},
          'zone_occupancy': {},
          'speed_analysis': {},
          'summary': {
            'total_tracks': 0,
            'active_tracks': 0,
            'class_counts': {}
          }
        }

        # Convert detections to mock tracked objects
        track_counter = 0
        for model_name, model_result in detection_results.items():
          for det in model_result.get('detections', []):
            track_id = f"mock_track_{track_counter}"
            mock_tracking_results['tracked_objects'][track_id] = {
              'track_id': track_id,
              'class_name': det.get('label', 'unknown'),
              'class_id': det.get('class_id', 0),
              'bbox': [det.get('x1', 0), det.get('y1', 0), det.get('x2', 0), det.get('y2', 0)],
              'centroid': [(det.get('x1', 0) + det.get('x2', 0)) / 2, (det.get('y1', 0) + det.get('y2', 0)) / 2],
              'confidence': det.get('confidence', 0),
              'age': 1,
              'hits': 1,
              'time_since_update': 0,
              'velocity': [0, 0],
              'speed_info': {
                'speed_px_per_sec': 0,
                'speed_m_per_sec': 0,
                'direction': 0
              },
              'trajectory_length': 1
            }
            track_counter += 1

            # Update summary
            class_name = det.get('label', 'unknown')
            mock_tracking_results['summary']['class_counts'][class_name] = \
              mock_tracking_results['summary']['class_counts'].get(class_name, 0) + 1

        mock_tracking_results['summary']['total_tracks'] = track_counter
        mock_tracking_results['summary']['active_tracks'] = track_counter

        return mock_tracking_results

    except Exception as e:
      logger.error(f"❌ Tracking processing error: {e}")
      return {
        "error": str(e),
        "detection_results": detection_results
      }

  def _build_tracking_response(self, tracked_objects) -> Dict[str, Any]:
    """Build complete tracking response"""
    try:
      tracking_results = {
        'tracked_objects': {},
        'zone_occupancy': {},
        'speed_analysis': {},
        'summary': {
          'total_tracks': len(tracked_objects),
          'active_tracks': len([obj for obj in tracked_objects.values() if obj.time_since_update < 5]),
          'class_counts': {}
        }
      }

      # Convert tracked objects to serializable format
      for track_id, obj in tracked_objects.items():
        # Calculate speed if speed calculator is available
        speed_data = {}
        if self.speed_calculator:
          speed_data = self.speed_calculator.calculate_speed(obj)
          avg_speed_data = self.speed_calculator.calculate_average_speed(obj)
          speed_data.update(avg_speed_data)

        tracking_results['tracked_objects'][track_id] = {
          'track_id': obj.track_id,
          'class_name': obj.class_name,
          'class_id': obj.class_id,
          'bbox': obj.bbox,
          'centroid': obj.centroid,
          'confidence': obj.confidence,
          'age': obj.age,
          'hits': obj.hits,
          'time_since_update': obj.time_since_update,
          'velocity': obj.velocity,
          'speed_info': speed_data,
          'trajectory_length': len(obj.trajectory)
        }

        # Update class counts
        class_name = obj.class_name
        tracking_results['summary']['class_counts'][class_name] = \
          tracking_results['summary']['class_counts'].get(class_name, 0) + 1

      # Zone occupancy if analytics is available
      if self.analytics and hasattr(self.analytics, 'zone_definitions') and self.analytics.zone_definitions:
        tracking_results['zone_occupancy'] = self.analytics.check_zone_occupancy(tracked_objects)

      # Speed analysis
      if self.speed_calculator:
        tracking_results['speed_analysis'] = {
          track_id: self.speed_calculator.calculate_speed(obj)
          for track_id, obj in tracked_objects.items()
        }

      return tracking_results

    except Exception as e:
      logger.error(f"❌ Build tracking response error: {e}")
      return {
        "error": str(e),
        "summary": {"total_tracks": 0, "active_tracks": 0, "class_counts": {}}
      }

  # Tracking message handlers
  async def _handle_tracking_message(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Handle tracking-specific messages"""
    message_type = data.get("type")

    try:
      if message_type == 'configure_tracking':
        await self._configure_tracking(websocket, data, connection_id)
      elif message_type == 'start_tracking':
        await self._start_tracking(websocket, connection_id)
      elif message_type == 'stop_tracking':
        await self._stop_tracking(websocket, connection_id)
      elif message_type == 'get_tracker_stats':
        await self._get_tracker_stats(websocket, connection_id)
      elif message_type == 'define_zone':
        await self._define_zone(websocket, data, connection_id)
      else:
        await self._send_error(websocket, f"Unknown tracking message: {message_type}")

    except Exception as e:
      await self._send_error(websocket, f"Tracking error: {str(e)}")

  async def _configure_tracking(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Configure tracking parameters"""
    config = data.get('config', {})
    logger.info(self.tracking_available)
    if self.tracking_available:
      tracker_type_str = config.get('tracker_type', 'centroid')
      tracker_params = config.get('tracker_params', {})

      # Import tracker type
      from ..tracking.tracker_manager import TrackerType
      tracker_type = TrackerType(tracker_type_str)

      # Configure speed calculator
      if 'speed_config' in config:
        speed_config = config['speed_config']
        self.speed_calculator.fps = speed_config.get('fps', 30)
        self.speed_calculator.pixel_to_meter_ratio = speed_config.get('pixel_to_meter_ratio', 1.0)

      # Create or update tracker
      success = self.tracker_manager.create_tracker(connection_id, tracker_type, **tracker_params)

      if success:
        self.stream_configs[connection_id] = config
        await self._send_response(websocket, {
          'type': 'tracking_configured',
          'stream_id': connection_id,
          'config': config,
          'success': True
        })
      else:
        await self._send_error(websocket, "Failed to configure tracker")
    else:
      await self._send_error(websocket, "Tracking not available")

  async def _start_tracking(self, websocket: WebSocket, connection_id: str):
    """Start tracking"""
    if self.tracking_available:
      self.tracking_enabled[connection_id] = True
      logger.info(f"▶️ Tracking started for {connection_id}")

      await self._send_response(websocket, {
        'type': 'tracking_started',
        'stream_id': connection_id
      })
    else:
      await self._send_error(websocket, "Tracking not available")

  async def _stop_tracking(self, websocket: WebSocket, connection_id: str):
    """Stop tracking"""
    self.tracking_enabled[connection_id] = False
    logger.info(f"⏹️ Tracking stopped for {connection_id}")

    await self._send_response(websocket, {
      'type': 'tracking_stopped',
      'stream_id': connection_id
    })

  async def _get_tracker_stats(self, websocket: WebSocket, connection_id: str):
    """Get tracker statistics"""
    if self.tracking_available:
      stats = self.tracker_manager.get_tracker_stats(connection_id)
      await self._send_response(websocket, {
        'type': 'tracker_stats',
        'stream_id': connection_id,
        'stats': stats
      })
    else:
      await self._send_error(websocket, "Tracking not available")

  async def _define_zone(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Define a zone for analytics"""
    if self.tracking_available:
      zone_id = data.get('zone_id')
      polygon_points = data.get('polygon_points', [])

      if zone_id and polygon_points:
        self.analytics.define_zone(zone_id, polygon_points)
        await self._send_response(websocket, {
          'type': 'zone_defined',
          'stream_id': connection_id,
          'zone_id': zone_id,
          'success': True
        })
      else:
        await self._send_error(websocket, "Invalid zone definition")
    else:
      await self._send_error(websocket, "Tracking not available")

  # Detection processing
  async def _configure_detection(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Configure detection parameters"""
    config = data.get('config', {})
    self.stream_configs[connection_id] = config

    await self._send_response(websocket, {
      'type': 'detection_configured',
      'stream_id': connection_id,
      'success': True
    })

  async def _get_models(self, websocket: WebSocket):
    """Get available models"""
    try:
      # Get models from your existing model manager
      if hasattr(model_manager, 'get_models'):
        models = model_manager.get_models()
      elif hasattr(model_manager, 'models'):
        models = model_manager.models
      else:
        models = {}

      await self._send_response(websocket, {
        'type': 'models_list',
        'models': models
      })
    except Exception as e:
      await self._send_error(websocket, f"Error getting models: {str(e)}")

  # Response senders - FIXED FORMATS
  async def _send_detection_results(self, websocket: WebSocket, results: Dict[str, Any], connection_id: str):
    """Send detection results in correct format"""
    response = {
      "type": "detections",
      "results": results,  # This should be a dict of {model_name: {detections: [...], count: N, ...}}
      "timestamp": int(time.time() * 1000)
    }

    # Debug logging
    total_detections = sum(
      model_result.get('count', 0)
      for model_result in results.values()
      if isinstance(model_result, dict)
    )
    logger.info(f"📤 Sending detection results: {total_detections} total detections")

    await websocket.send_text(json.dumps(self.serialize_detections(response)))

  def serialize_detections(self, obj):
    if isinstance(obj, Detection):
      return obj.dict()
    elif isinstance(obj, list):
      return [self.serialize_detections(item) for item in obj]
    elif isinstance(obj, dict):
      return {key: self.serialize_detections(value) for key, value in obj.items()}
    else:
      return obj

  async def _send_tracking_results(self, websocket: WebSocket, results: Dict[str, Any], connection_id: str):
    """Send tracking results"""
    response = {
      "type": "tracking_results",
      "stream_id": connection_id,
      "results": results,
      "timestamp": int(time.time() * 1000)
    }

    logger.info(f"📤 Sending tracking results: {results.get('summary', {}).get('active_tracks', 0)} active tracks")

    await websocket.send_text(json.dumps(self.serialize_detections(response)))

  async def _send_response(self, websocket: WebSocket, data: Dict):
    """Send response to WebSocket client"""
    try:
      await websocket.send_text(json.dumps(data))
    except Exception as e:
      logger.error(f"❌ Error sending response: {e}")

  async def _send_error(self, websocket: WebSocket, error_message: str):
    """Send error message"""
    try:
      await websocket.send_text(json.dumps({
        'type': 'error',
        'message': error_message
      }))
    except Exception as e:
      logger.error(f"❌ Error sending error message: {e}")

  def _cleanup_connection(self, connection_id: str):
    """Clean up connection resources"""
    if connection_id in self.active_connections:
      del self.active_connections[connection_id]
    if connection_id in self.tracking_enabled:
      del self.tracking_enabled[connection_id]
    if connection_id in self.stream_configs:
      del self.stream_configs[connection_id]


# Global unified handler instance
unified_handler = UnifiedWebSocketHandler()

# Replace your main WebSocket endpoint with this:
# In your main.py or wherever your WebSocket route is defined: