# app/websocket/multi_camera_handlers.py
import json
import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
from io import BytesIO
from PIL import Image
import numpy as np

from ..core.schemas import Detection
from ..models import model_manager
from ..tracking.tracker_manager import tracker_manager
from ..tracking.speed_calculator import speed_calculator
from ..tracking.analytics import analytics

logger = logging.getLogger(__name__)


class CameraConnection:
  """Represents a single camera connection"""

  def __init__(self, camera_id: str, websocket: WebSocket, connection_id: str):
    self.camera_id = camera_id
    self.websocket = websocket
    self.connection_id = connection_id
    self.is_active = True
    self.is_streaming = False
    self.last_frame_time = 0
    self.frame_count = 0
    self.config = {}
    self.tracking_enabled = False

    # Camera specific settings
    self.camera_name = ""
    self.camera_location = ""
    self.stream_url = ""
    self.resolution = "1920x1080"
    self.fps = 30


class MultiCameraWebSocketHandler:
  """Enhanced WebSocket handler supporting multiple camera streams"""

  def __init__(self):
    # Connection management
    self.active_connections: Dict[str, WebSocket] = {}
    self.camera_connections: Dict[str, CameraConnection] = {}  # camera_id -> CameraConnection
    self.connection_cameras: Dict[str, Set[str]] = {}  # connection_id -> set of camera_ids

    # Tracking and detection state per camera
    self.camera_tracking_enabled: Dict[str, bool] = {}
    self.camera_configs: Dict[str, Dict] = {}
    self.camera_models: Dict[str, List[Dict]] = {}

    # Performance tracking
    self.camera_stats: Dict[str, Dict] = {}

    # Calibration per camera
    self.camera_calibration: Dict[str, Dict] = {}

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

  async def handle_websocket(self, websocket: WebSocket):
    """Main WebSocket connection handler"""
    await websocket.accept()
    connection_id = f"conn_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
    self.active_connections[connection_id] = websocket
    self.connection_cameras[connection_id] = set()

    logger.info(f"✅ WebSocket connected: {connection_id}")

    try:
      await self._send_connection_established(websocket, connection_id)

      while True:
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
      await self._cleanup_connection(connection_id)

  async def _send_connection_established(self, websocket: WebSocket, connection_id: str):
    """Send connection confirmation"""
    await self._send_response(websocket, {
      "type": "connection_established",
      "connection_id": connection_id,
      "max_cameras": 50,  # Set your limit
      "features": {
        "tracking": self.tracking_available,
        "calibration": True,
        "analytics": True
      }
    })

  async def _handle_text_message(self, websocket: WebSocket, text_data: str, connection_id: str):
    """Handle JSON control messages"""
    try:
      data = json.loads(text_data)
      message_type = data.get("type")
      camera_id = data.get("camera_id")

      logger.info(f"🎛️ Handling message: {message_type} for camera: {camera_id}")

      # Camera management messages
      if message_type == "add_camera":
        await self._handle_add_camera(websocket, data, connection_id)
      elif message_type == "remove_camera":
        await self._handle_remove_camera(websocket, data, connection_id)
      elif message_type == "start_camera":
        await self._handle_start_camera(websocket, data, connection_id)
      elif message_type == "stop_camera":
        await self._handle_stop_camera(websocket, data, connection_id)
      elif message_type == "list_cameras":
        await self._handle_list_cameras(websocket, connection_id)
      elif message_type == "get_camera_status":
        await self._handle_get_camera_status(websocket, data, connection_id)

      # Configuration messages
      elif message_type == "configure_camera":
        await self._handle_configure_camera(websocket, data, connection_id)
      elif message_type == "set_camera_models":
        await self._handle_set_camera_models(websocket, data, connection_id)

      # Tracking messages
      elif message_type in ["configure_tracking", "start_tracking", "stop_tracking", "get_tracker_stats",
                            "define_zone"]:
        await self._handle_tracking_message(websocket, data, connection_id)

      # Calibration messages
      elif "calibration" in message_type or "command" in data:
        await self._handle_calibration_message(websocket, data, connection_id)

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
    """Handle binary image data with camera identification"""
    try:
      # Parse binary data to extract camera_id and frame
      camera_id, models_to_use, image_array = self._parse_binary_data_with_camera(binary_data)

      if not camera_id:
        await self._send_error(websocket, "Camera ID not found in binary data")
        return

      if camera_id not in self.camera_connections:
        await self._send_error(websocket, f"Camera {camera_id} not registered")
        return

      camera_conn = self.camera_connections[camera_id]

      if not camera_conn.is_streaming:
        await self._send_error(websocket, f"Camera {camera_id} not streaming")
        return

      logger.info(f"📸 Processing frame for camera {camera_id}: {len(binary_data)} bytes")

      # Update camera stats
      self._update_camera_stats(camera_id)

      # Use camera-specific models if not provided
      if not models_to_use:
        models_to_use = self.camera_models.get(camera_id, [])

      if not models_to_use or image_array is None:
        await self._send_error(websocket, f"No models configured for camera {camera_id}")
        return

      # Process detections
      detection_results = await self._process_detections(models_to_use, image_array, camera_id)

      # Handle tracking if enabled for this camera
      if self.tracking_available and self.camera_tracking_enabled.get(camera_id, False):
        tracking_results = await self._process_camera_tracking(detection_results, camera_id)
        await self._send_camera_tracking_results(websocket, tracking_results, camera_id)
      else:
        await self._send_camera_detection_results(websocket, detection_results, camera_id)

    except Exception as e:
      logger.error(f"❌ Binary processing error: {e}")
      await self._send_error(websocket, f"Frame processing error: {str(e)}")

  def _parse_binary_data_with_camera(self, binary_data: bytes) -> tuple[str, List[Dict], np.ndarray]:
    """Parse binary data with camera identification"""
    try:
      offset = 0

      # Read camera ID length and camera ID (new addition)
      camera_id_length = binary_data[offset]
      offset += 1
      camera_id = binary_data[offset:offset + camera_id_length].decode('utf-8')
      offset += camera_id_length

      # Read timestamp (4 bytes, little endian)
      timestamp = int.from_bytes(binary_data[offset:offset + 4], 'little')
      offset += 4

      # Read model count (1 byte)
      model_count = binary_data[offset]
      offset += 1

      # Parse models (existing logic)
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

      # Image data (remaining bytes)
      image_data = binary_data[offset:]
      image_io = BytesIO(image_data)
      pil_image = Image.open(image_io)

      if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

      image_array = np.array(pil_image)

      return camera_id, models_to_use, image_array

    except Exception as e:
      logger.error(f"❌ Binary parsing error: {e}")
      return "", [], None

  # Camera Management Methods
  async def _handle_add_camera(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Add a new camera"""
    try:
      camera_id = data.get("camera_id")
      camera_config = data.get("config", {})

      if not camera_id:
        await self._send_error(websocket, "Camera ID is required")
        return

      if camera_id in self.camera_connections:
        await self._send_error(websocket, f"Camera {camera_id} already exists")
        return

      # Create camera connection
      camera_conn = CameraConnection(camera_id, websocket, connection_id)
      camera_conn.camera_name = camera_config.get("name", camera_id)
      camera_conn.camera_location = camera_config.get("location", "")
      camera_conn.stream_url = camera_config.get("source", "")
      camera_conn.resolution = camera_config.get("resolution", "1920x1080")
      camera_conn.fps = camera_config.get("fps", 30)

      self.camera_connections[camera_id] = camera_conn
      self.connection_cameras[connection_id].add(camera_id)
      self.camera_configs[camera_id] = camera_config
      self.camera_models[camera_id] = []  # Default to no models
      self.camera_tracking_enabled[camera_id] = False
      self.camera_stats[camera_id] = {
        "frames_processed": 0,
        "last_frame_time": 0,
        "fps": 0,
        "errors": 0
      }

      await self._send_response(websocket, {
        "type": "camera_added",
        "camera_id": camera_id,
        "status": "success",
        "config": camera_config
      })

      logger.info(f"✅ Added camera: {camera_id}")

    except Exception as e:
      logger.error(f"❌ Error adding camera: {e}")
      await self._send_error(websocket, f"Failed to add camera: {str(e)}")

  async def _handle_remove_camera(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Remove a camera"""
    try:
      camera_id = data.get("camera_id")

      if not camera_id or camera_id not in self.camera_connections:
        await self._send_error(websocket, f"Camera {camera_id} not found")
        return

      # Stop tracking if enabled
      if self.camera_tracking_enabled.get(camera_id, False):
        self.tracker_manager.remove_tracker(camera_id)

      # Clean up camera data
      del self.camera_connections[camera_id]
      self.connection_cameras[connection_id].discard(camera_id)
      self.camera_configs.pop(camera_id, None)
      self.camera_models.pop(camera_id, None)
      self.camera_tracking_enabled.pop(camera_id, None)
      self.camera_stats.pop(camera_id, None)
      self.camera_calibration.pop(camera_id, None)

      await self._send_response(websocket, {
        "type": "camera_removed",
        "camera_id": camera_id,
        "status": "success"
      })

      logger.info(f"🗑️ Removed camera: {camera_id}")

    except Exception as e:
      logger.error(f"❌ Error removing camera: {e}")
      await self._send_error(websocket, f"Failed to remove camera: {str(e)}")

  async def _handle_start_camera(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Start streaming for a camera"""
    camera_id = data.get("camera_id")

    if camera_id not in self.camera_connections:
      await self._send_error(websocket, f"Camera {camera_id} not found")
      return

    camera_conn = self.camera_connections[camera_id]
    camera_conn.is_streaming = True

    await self._send_response(websocket, {
      "type": "camera_started",
      "camera_id": camera_id,
      "status": "streaming"
    })

  async def _handle_stop_camera(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Stop streaming for a camera"""
    camera_id = data.get("camera_id")

    if camera_id not in self.camera_connections:
      await self._send_error(websocket, f"Camera {camera_id} not found")
      return

    camera_conn = self.camera_connections[camera_id]
    camera_conn.is_streaming = False

    await self._send_response(websocket, {
      "type": "camera_stopped",
      "camera_id": camera_id,
      "status": "stopped"
    })

  async def _handle_list_cameras(self, websocket: WebSocket, connection_id: str):
    """List all cameras for this connection"""
    camera_list = []
    for camera_id in self.connection_cameras.get(connection_id, set()):
      camera_conn = self.camera_connections.get(camera_id)
      if camera_conn:
        camera_list.append({
          "camera_id": camera_id,
          "name": camera_conn.camera_name,
          "location": camera_conn.camera_location,
          "is_streaming": camera_conn.is_streaming,
          "tracking_enabled": self.camera_tracking_enabled.get(camera_id, False),
          "models_count": len(self.camera_models.get(camera_id, [])),
          "stats": self.camera_stats.get(camera_id, {})
        })

    await self._send_response(websocket, {
      "type": "camera_list",
      "cameras": camera_list,
      "total_count": len(camera_list)
    })

  async def _handle_configure_camera(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Configure camera settings"""
    camera_id = data.get("camera_id")
    config = data.get("config", {})

    if camera_id not in self.camera_connections:
      await self._send_error(websocket, f"Camera {camera_id} not found")
      return

    # Update camera configuration
    self.camera_configs[camera_id].update(config)
    camera_conn = self.camera_connections[camera_id]

    if "name" in config:
      camera_conn.camera_name = config["name"]
    if "location" in config:
      camera_conn.camera_location = config["location"]
    if "fps" in config:
      camera_conn.fps = config["fps"]

    await self._send_response(websocket, {
      "type": "camera_configured",
      "camera_id": camera_id,
      "config": self.camera_configs[camera_id]
    })

  async def _handle_set_camera_models(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Set AI models for a specific camera"""
    camera_id = data.get("camera_id")
    models = data.get("models", [])

    if camera_id not in self.camera_connections:
      await self._send_error(websocket, f"Camera {camera_id} not found")
      return

    self.camera_models[camera_id] = models

    await self._send_response(websocket, {
      "type": "camera_models_set",
      "camera_id": camera_id,
      "models": models
    })

  # Detection and Tracking Methods
  async def _process_detections(self, models_to_use: List[Dict], image_array: np.ndarray, camera_id: str) -> Dict[
    str, Any]:
    """Process detections for a specific camera"""
    detection_results = {}

    for model_config in models_to_use:
      model_name = model_config['name']
      class_filter = model_config['class_filter']

      try:
        detections_response = model_manager.model_manager.run_inference(image_array, model_name, class_filter)

        if detections_response is None:
          formatted_detections = []
          count = 0
          error = "No response from model"
        else:
          formatted_detections = detections_response.detections
          count = detections_response.count
          error = detections_response.error

        detection_results[model_name] = {
          "detections": formatted_detections,
          "count": count,
          "model": model_name,
          "error": error
        }

      except Exception as model_error:
        logger.error(f"❌ Model {model_name} error for camera {camera_id}: {model_error}")
        detection_results[model_name] = {
          "detections": [],
          "count": 0,
          "model": model_name,
          "error": str(model_error)
        }

    return detection_results

  async def _process_camera_tracking(self, detection_results: Dict[str, Any], camera_id: str) -> Dict[str, Any]:
    """Process tracking for a specific camera"""
    try:
      # Convert detections to tracking format
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

      # Update tracker for this specific camera
      tracked_objects = self.tracker_manager.update_tracker(camera_id, all_detections)

      # Build tracking response
      tracking_results = self._build_camera_tracking_response(tracked_objects, camera_id)

      # Add calibration data if available
      if camera_id in self.camera_calibration:
        tracking_results = self._add_camera_calibration_data(tracking_results, camera_id)

      return tracking_results

    except Exception as e:
      logger.error(f"❌ Tracking error for camera {camera_id}: {e}")
      return {
        "error": str(e),
        "detection_results": detection_results
      }

  def _build_camera_tracking_response(self, tracked_objects, camera_id: str) -> Dict[str, Any]:
    """Build tracking response for a specific camera"""
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

    return tracking_results

  # Response Methods
  async def _send_camera_detection_results(self, websocket: WebSocket, results: Dict[str, Any], camera_id: str):
    """Send detection results for a specific camera"""
    response = {
      "type": "frame_data",
      "camera_id": camera_id,
      "results": results,
      "timestamp": int(time.time() * 1000)
    }

    await websocket.send_text(json.dumps(self._serialize_response(response)))

  async def _send_camera_tracking_results(self, websocket: WebSocket, results: Dict[str, Any], camera_id: str):
    """Send tracking results for a specific camera"""
    response = {
      "type": "tracking_results",
      "camera_id": camera_id,
      "results": results,
      "timestamp": int(time.time() * 1000)
    }

    await websocket.send_text(json.dumps(self._serialize_response(response)))

  def _serialize_response(self, obj):
    """Serialize response objects"""
    if hasattr(obj, 'dict'):
      return obj.dict()
    elif isinstance(obj, list):
      return [self._serialize_response(item) for item in obj]
    elif isinstance(obj, dict):
      return {key: self._serialize_response(value) for key, value in obj.items()}
    else:
      return obj

  def _update_camera_stats(self, camera_id: str):
    """Update performance stats for a camera"""
    current_time = time.time()
    stats = self.camera_stats[camera_id]

    stats["frames_processed"] += 1

    if stats["last_frame_time"] > 0:
      time_diff = current_time - stats["last_frame_time"]
      if time_diff > 0:
        stats["fps"] = 1.0 / time_diff

    stats["last_frame_time"] = current_time

  # Tracking message handlers (adapted for multi-camera)
  async def _handle_tracking_message(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Handle tracking messages for specific cameras"""
    message_type = data.get("type")
    camera_id = data.get("camera_id")

    if not camera_id:
      await self._send_error(websocket, "Camera ID required for tracking operations")
      return

    try:
      if message_type == 'configure_tracking':
        await self._configure_camera_tracking(websocket, data, camera_id)
      elif message_type == 'start_tracking':
        await self._start_camera_tracking(websocket, camera_id)
      elif message_type == 'stop_tracking':
        await self._stop_camera_tracking(websocket, camera_id)
      elif message_type == 'get_tracker_stats':
        await self._get_camera_tracker_stats(websocket, camera_id)
      elif message_type == 'define_zone':
        await self._define_camera_zone(websocket, data, camera_id)

    except Exception as e:
      await self._send_error(websocket, f"Tracking error for camera {camera_id}: {str(e)}")

  async def _configure_camera_tracking(self, websocket: WebSocket, data: Dict, camera_id: str):
    """Configure tracking for a specific camera"""
    config = data.get('config', {})

    if self.tracking_available:
      tracker_type_str = config.get('tracker_type', 'centroid')
      tracker_params = config.get('tracker_params', {})

      from ..tracking.tracker_manager import TrackerType
      tracker_type = TrackerType(tracker_type_str)

      # Configure speed calculator for this camera
      if 'speed_config' in config:
        speed_config = config['speed_config']
        self.speed_calculator.fps = speed_config.get('fps', 30)
        self.speed_calculator.pixel_to_meter_ratio = speed_config.get('pixel_to_meter_ratio', 1.0)

      # Create tracker for this camera
      success = self.tracker_manager.create_tracker(camera_id, tracker_type, **tracker_params)

      if success:
        self.camera_configs[camera_id]['tracking'] = config
        await self._send_response(websocket, {
          'type': 'tracking_configured',
          'camera_id': camera_id,
          'config': config,
          'success': True
        })
      else:
        await self._send_error(websocket, f"Failed to configure tracker for camera {camera_id}")
    else:
      await self._send_error(websocket, "Tracking not available")

  async def _start_camera_tracking(self, websocket: WebSocket, camera_id: str):
    """Start tracking for a specific camera"""
    if self.tracking_available:
      self.camera_tracking_enabled[camera_id] = True

      await self._send_response(websocket, {
        'type': 'tracking_started',
        'camera_id': camera_id
      })
    else:
      await self._send_error(websocket, "Tracking not available")

  async def _stop_camera_tracking(self, websocket: WebSocket, camera_id: str):
    """Stop tracking for a specific camera"""
    self.camera_tracking_enabled[camera_id] = False

    await self._send_response(websocket, {
      'type': 'tracking_stopped',
      'camera_id': camera_id
    })

  async def _get_camera_tracker_stats(self, websocket: WebSocket, camera_id: str):
    """Get tracker stats for a specific camera"""
    if self.tracking_available:
      stats = self.tracker_manager.get_tracker_stats(camera_id)
      await self._send_response(websocket, {
        'type': 'tracker_stats',
        'camera_id': camera_id,
        'stats': stats
      })
    else:
      await self._send_error(websocket, "Tracking not available")

  # Calibration methods (adapted for multi-camera)
  async def _handle_calibration_message(self, websocket: WebSocket, data: Dict, connection_id: str):
    """Handle calibration messages for specific cameras"""
    camera_id = data.get("camera_id")

    if not camera_id:
      await self._send_error(websocket, "Camera ID required for calibration")
      return

    if "command" in data:
      command = data.get('command')
      payload = data.get('data', {})

      if command == "calibrate":
        await self._setup_camera_calibration(websocket, payload, camera_id)
      elif command == "get_calibration_info":
        await self._send_camera_calibration_info(websocket, camera_id)
      # Add other calibration commands as needed

  def _add_camera_calibration_data(self, tracking_results: Dict, camera_id: str) -> Dict:
    """Add calibration data to tracking results for a specific camera"""
    calibration_data = self.camera_calibration.get(camera_id)
    if not calibration_data:
      return tracking_results

    # Add real-world coordinates if calibrated
    enhanced_objects = {}
    for track_id, obj in tracking_results.get('tracked_objects', {}).items():
      enhanced_obj = obj.copy()

      if calibration_data.get('calibrated', False):
        # Add calibration-enhanced data
        pixels_per_meter = calibration_data.get('pixels_per_meter', 100.0)
        if 'centroid' in obj:
          center_x, center_y = obj['centroid']
          real_x = center_x / pixels_per_meter
          real_y = center_y / pixels_per_meter
          enhanced_obj['center_meters'] = [real_x, real_y]

      enhanced_objects[track_id] = enhanced_obj

    tracking_results['tracked_objects'] = enhanced_objects
    tracking_results['calibration'] = calibration_data

    return tracking_results

  # Utility methods
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
        'message': error_message,
        'timestamp': int(time.time() * 1000)
      }))
    except Exception as e:
      logger.error(f"❌ Error sending error message: {e}")

  async def _cleanup_connection(self, connection_id: str):
    """Clean up connection resources"""
    try:
      # Remove all cameras for this connection
      camera_ids_to_remove = list(self.connection_cameras.get(connection_id, set()))
      for camera_id in camera_ids_to_remove:
        # Stop tracking if enabled
        if self.camera_tracking_enabled.get(camera_id, False):
          self.tracker_manager.remove_tracker(camera_id)

        # Clean up camera data
        self.camera_connections.pop(camera_id, None)
        self.camera_configs.pop(camera_id, None)
        self.camera_models.pop(camera_id, None)
        self.camera_tracking_enabled.pop(camera_id, None)
        self.camera_stats.pop(camera_id, None)
        self.camera_calibration.pop(camera_id, None)

      # Remove connection
      self.active_connections.pop(connection_id, None)
      self.connection_cameras.pop(connection_id, None)

      logger.info(f"🧹 Cleaned up connection {connection_id} and {len(camera_ids_to_remove)} cameras")

    except Exception as e:
      logger.error(f"❌ Error during cleanup: {e}")


# Global multi-camera handler instance
multi_camera_handler = MultiCameraWebSocketHandler()