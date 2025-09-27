# app/websocket/unified_handlers.py - FIXED VERSION
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
    if self.tracking_enabled:
      self.enhanced_tracker = EnhancedTrackerManager(self.tracker_manager)

    def _add_calibration_data(self, tracking_results: Dict) -> Dict:
      # Update your existing method to use enhanced tracker
      if self.calibration_enabled and hasattr(self, 'enhanced_tracker'):
        # Set calibration on enhanced tracker
        self.enhanced_tracker.set_calibration(self.pixels_per_meter)

        # The enhanced tracker will add all measurements automatically
        # when you call update_tracker_with_measurements instead of update_tracker

      return tracking_results

    # Add new calibration command handlers
    async def _handle_calibration_command(self, websocket: WebSocket, data: Dict, connection_id: str):
      # ... your existing calibration handling ...

      # Also set calibration on enhanced tracker
      if self.calibration_enabled and hasattr(self, 'enhanced_tracker'):
        self.enhanced_tracker.set_calibration(self.pixels_per_meter)

    async def _configure_measurement_alerts(self, websocket: WebSocket, data: Dict, connection_id: str):
      # New method to configure distance/size alerts
      if hasattr(self, 'enhanced_tracker'):
        config = data.get('config', {})
        self.enhanced_tracker.configure_alerts(
          proximity_enabled=config.get('proximity_alerts', True),
          size_validation_enabled=config.get('size_validation', True),
          alert_distance=config.get('alert_distance', 2.0)
        )

        await self._send_response(websocket, {
          'type': 'measurement_alerts_configured',
          'config': config,
          'success': True
        })
    # Initialize tracking components
    try:
      self.tracker_manager = tracker_manager
      self.speed_calculator = speed_calculator
      self.analytics = analytics
      self.tracking_available = True
      logger.info("✅ Tracking components initialized")
    except ImportError as e:
      logger.warning(f"⚠️ Tracking not available: {e}")
      self.tracking_available = False

    # NEW: Simple calibration state (preserving your tracking while adding calibration)
    self.calibration_enabled = False
    self.pixels_per_meter = 100.0

  async def handle_websocket(self, websocket: WebSocket):
    """Main WebSocket connection handler - FIXED to handle both message types"""
    await websocket.accept()
    connection_id = f"stream_{int(time.time() * 1000)}"
    self.active_connections[connection_id] = websocket

    logger.info(f"✅ WebSocket connected: {connection_id}")

    try:
      while True:
        # FIXED: Safely handle both text and binary messages
        try:
          # Try to receive text first (JSON commands)
          message = await websocket.receive_text()
          await self._handle_text_message(websocket, message, connection_id)
        except:
          try:
            # If text fails, try binary (frame data)
            message = await websocket.receive_bytes()
            await self._handle_binary_message(websocket, message, connection_id)
          except WebSocketDisconnect:
            break
          except Exception as e:
            logger.error(f"❌ Message receive error: {e}")
            break

    except WebSocketDisconnect:
      logger.info(f"🔌 WebSocket disconnected: {connection_id}")
    except Exception as e:
      logger.error(f"❌ WebSocket error: {e}")
      await self._send_error(websocket, f"Server error: {str(e)}")
    finally:
      self._cleanup_connection(connection_id)

  async def _handle_text_message(self, websocket: WebSocket, text_data: str, connection_id: str):
    """Handle JSON control messages (tracking and calibration)"""
    try:
      data = json.loads(text_data)
      message_type = data.get("type")

      # Handle 'command' format for calibration (NEW)
      if "command" in data:
        await self._handle_calibration_command(websocket, data, connection_id)
        return

      logger.info(f"🎛️ Handling control message: {message_type}")

      # YOUR EXISTING TRACKING MESSAGES (UNCHANGED)
      if (message_type.startswith('tracking_') or message_type in [
        'configure_tracking', 'start_tracking', 'stop_tracking',
        'get_tracker_stats', 'define_zone'
      ]):
        await self._handle_tracking_message(websocket, data, connection_id)

      # YOUR EXISTING CONTROL MESSAGES (UNCHANGED)
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

  # NEW: Calibration command handler (doesn't interfere with your tracking)
  async def _handle_calibration_command(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Handle calibration commands"""
    try:
      command = data.get('command')
      payload = data.get('data', {})

      if command == "calibrate":
        await self._setup_calibration(websocket, payload)
      elif command == "get_calibration_info":
        await self._send_calibration_info(websocket)
      elif command == "save_calibration":
        await self._save_calibration(websocket, payload)
      elif command == "load_calibration":
        await self._load_calibration(websocket, payload)
      else:
        await self._send_error(websocket, f"Unknown calibration command: {command}")

    except Exception as e:
      await self._send_error(websocket, f"Calibration error: {str(e)}")

  async def _setup_calibration(self, websocket: WebSocket, payload: Dict):
    """Setup calibration (simple implementation)"""
    try:
      mode = payload.get('mode')
      points = payload.get('points', [])

      success = False
      error = None

      if mode == "reference_object" and len(points) >= 2:
        # Calculate pixels per meter
        p1, p2 = points[0], points[1]
        pixel_dist = ((p2['pixel_x'] - p1['pixel_x']) ** 2 + (p2['pixel_y'] - p1['pixel_y']) ** 2) ** 0.5
        real_dist = ((p2['real_x'] - p1['real_x']) ** 2 + (p2['real_y'] - p1['real_y']) ** 2) ** 0.5

        if pixel_dist > 0 and real_dist > 0:
          self.pixels_per_meter = pixel_dist / real_dist
          self.calibration_enabled = True
          success = True
        else:
          error = "Invalid calibration points"

      elif mode == "perspective_transform" and len(points) == 4:
        # Simple approximation for now
        self.calibration_enabled = True
        success = True
      else:
        error = f"Invalid calibration: {mode} with {len(points)} points"

      await websocket.send_text(json.dumps({
        "type": "calibration_result",
        "success": success,
        "calibration_info": {
          "calibrated": self.calibration_enabled,
          "mode": mode if success else None,
          "timestamp": time.time() if success else None,
          "frame_size": [640, 480],
          "meters_per_pixel": 1.0 / self.pixels_per_meter if success else None
        },
        "error": error
      }))

    except Exception as e:
      await websocket.send_text(json.dumps({
        "type": "calibration_result",
        "success": False,
        "error": str(e)
      }))

  async def _send_calibration_info(self, websocket: WebSocket):
    """Send calibration info"""
    await websocket.send_text(json.dumps({
      "type": "calibration_info",
      "data": {
        "calibrated": self.calibration_enabled,
        "mode": "reference_object" if self.calibration_enabled else None,
        "timestamp": time.time() if self.calibration_enabled else None,
        "frame_size": [640, 480],
        "meters_per_pixel": 1.0 / self.pixels_per_meter if self.calibration_enabled else None
      }
    }))

  async def _save_calibration(self, websocket: WebSocket, payload: Dict):
    """Save calibration"""
    filepath = payload.get('filepath', 'calibration.json')
    try:
      calibration_data = {
        "calibrated": self.calibration_enabled,
        "pixels_per_meter": self.pixels_per_meter,
        "timestamp": time.time()
      }

      with open(filepath, 'w') as f:
        json.dump(calibration_data, f)

      await websocket.send_text(json.dumps({
        "type": "calibration_save_result",
        "success": True,
        "filepath": filepath
      }))
    except Exception as e:
      await websocket.send_text(json.dumps({
        "type": "calibration_save_result",
        "success": False,
        "filepath": filepath,
        "error": str(e)
      }))

  async def _load_calibration(self, websocket: WebSocket, payload: Dict):
    """Load calibration"""
    filepath = payload.get('filepath', 'calibration.json')
    try:
      import os
      if not os.path.exists(filepath):
        raise FileNotFoundError(f"Calibration file not found: {filepath}")

      with open(filepath, 'r') as f:
        calibration_data = json.load(f)

      self.calibration_enabled = calibration_data.get('calibrated', False)
      self.pixels_per_meter = calibration_data.get('pixels_per_meter', 100.0)

      await websocket.send_text(json.dumps({
        "type": "calibration_load_result",
        "success": True,
        "filepath": filepath,
        "calibration_info": {
          "calibrated": self.calibration_enabled,
          "mode": "reference_object" if self.calibration_enabled else None,
          "timestamp": calibration_data.get('timestamp'),
          "frame_size": [640, 480],
          "meters_per_pixel": 1.0 / self.pixels_per_meter if self.calibration_enabled else None
        }
      }))
    except Exception as e:
      await websocket.send_text(json.dumps({
        "type": "calibration_load_result",
        "success": False,
        "filepath": filepath,
        "error": str(e)
      }))

  async def _handle_binary_message(self, websocket: WebSocket, binary_data: bytes, connection_id: str):
    """Handle binary image data - YOUR WORKING VERSION WITH ENHANCEMENTS"""
    try:
      logger.info(f"📸 Processing frame: {len(binary_data)} bytes")

      # Parse binary data (YOUR EXISTING LOGIC)
      models_to_use, image_array = self._parse_binary_data(binary_data)

      if not models_to_use or image_array is None:
        await self._send_error(websocket, "Failed to parse binary data")
        return

      logger.info(f"🔍 Processing {len(models_to_use)} models, image: {image_array.shape}")

      # Process with your existing model manager (FIXED)
      detection_results = await self._process_detections_fixed(models_to_use, image_array)

      # Check if tracking is enabled (YOUR EXISTING LOGIC)
      if self.tracking_available and self.tracking_enabled.get(connection_id, False):
        tracking_results = await self._process_with_tracking(detection_results, connection_id)

        # ENHANCED: Add calibration data to tracking results
        if self.calibration_enabled:
          tracking_results = self._add_calibration_data(tracking_results)

        await self._send_tracking_results(websocket, tracking_results, connection_id)
      else:
        # Send regular detection results (YOUR EXISTING BEHAVIOR)
        # ENHANCED: Add calibration data if available
        if self.calibration_enabled:
          detection_results = self._add_calibration_to_detections(detection_results)

        await self._send_detection_results(websocket, detection_results, connection_id)

    except Exception as e:
      logger.error(f"❌ Binary processing error: {e}")
      await self._send_error(websocket, f"Frame processing error: {str(e)}")

  def _add_calibration_data(self, tracking_results: Dict) -> Dict:
    """Add calibration data to tracking results"""
    if not self.calibration_enabled:
      return tracking_results

    # Add real-world coordinates and speeds to tracked objects
    enhanced_objects = {}
    for track_id, obj in tracking_results.get('tracked_objects', {}).items():
      enhanced_obj = obj.copy()

      # Add real-world coordinates
      if 'centroid' in obj:
        center_x, center_y = obj['centroid']
        real_x = center_x / self.pixels_per_meter
        real_y = center_y / self.pixels_per_meter
        enhanced_obj['center_meters'] = [real_x, real_y]

      # Add speed in km/h if available
      if 'speed_info' in obj and obj['speed_info'].get('speed_m_per_sec'):
        speed_ms = obj['speed_info']['speed_m_per_sec']
        enhanced_obj['speed_kmh'] = speed_ms * 3.6

      enhanced_objects[track_id] = enhanced_obj

    tracking_results['tracked_objects'] = enhanced_objects

    # Add calibration info
    tracking_results['calibration'] = {
      "calibrated": True,
      "mode": "reference_object",
      "meters_per_pixel": 1.0 / self.pixels_per_meter
    }

    return tracking_results

  def _add_calibration_to_detections(self, detection_results: Dict) -> Dict:
    """Add calibration data to detection results"""
    if not self.calibration_enabled:
      return detection_results

    # Add calibration info to response
    enhanced_results = detection_results.copy()
    enhanced_results['calibration'] = {
      "calibrated": True,
      "mode": "reference_object",
      "meters_per_pixel": 1.0 / self.pixels_per_meter
    }

    return enhanced_results

  def _parse_binary_data(self, binary_data: bytes) -> tuple[List[Dict], np.ndarray]:
    """Parse binary data according to your format - YOUR EXISTING CODE"""
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
    """Process detections with your existing model manager - FIXED"""
    detection_results = {}

    for model_config in models_to_use:
      model_name = model_config['name']
      class_filter = model_config['class_filter']

      try:
        logger.info(f"🔍 Processing model: {model_name} with filter: {class_filter}")

        # Use YOUR existing model manager interface
        detections_response = model_manager.model_manager.run_inference(image_array, model_name, class_filter)

        if detections_response is None:
          formatted_detections = []
          count = 0
          error = "No response from model"
        else:
          # Handle your model manager's response format
          formatted_detections = detections_response.detections
          count = detections_response.count
          error = detections_response.error

        # Store as proper dictionary structure (YOUR FORMAT)
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

  # ALL YOUR EXISTING TRACKING METHODS REMAIN UNCHANGED
  async def _process_with_tracking(self, detection_results: Dict[str, Any], connection_id: str) -> Dict[str, Any]:
    """Process detections with tracking if available - YOUR EXISTING CODE"""
    try:
      if not self.tracking_enabled.get(connection_id, False):
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
    """Build complete tracking response - YOUR EXISTING CODE"""
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

  # ALL YOUR EXISTING TRACKING METHODS REMAIN UNCHANGED
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
    """Send detection results in correct format - YOUR EXISTING FORMAT"""
    response = {
      "type": "detections",
      "results": results,
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
    """Serialize detection objects - YOUR EXISTING LOGIC"""
    if isinstance(obj, Detection):
      return obj.dict()
    elif isinstance(obj, list):
      return [self.serialize_detections(item) for item in obj]
    elif isinstance(obj, dict):
      return {key: self.serialize_detections(value) for key, value in obj.items()}
    else:
      return obj

  async def _send_tracking_results(self, websocket: WebSocket, results: Dict[str, Any], connection_id: str):
    """Send tracking results - YOUR EXISTING FORMAT"""
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
