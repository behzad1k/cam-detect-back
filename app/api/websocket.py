from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import json
import logging
import time
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Dict, Set

from app.database.base import get_db
from app.services.camera_service import camera_service
from app.services.stream_manager import stream_manager
from app.core.detection.yolo_detector import detector
from app.core.tracking.speed_calculator import speed_calculator
from app.schemas.detection import Detection

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
  """Manage WebSocket connections per camera"""

  def __init__(self):
    # camera_id -> Set of WebSocket connections
    self.active_connections: Dict[str, Set[WebSocket]] = {}

  async def connect(self, websocket: WebSocket, camera_id: str):
    """Connect a client to a camera stream"""
    await websocket.accept()

    if camera_id not in self.active_connections:
      self.active_connections[camera_id] = set()

    self.active_connections[camera_id].add(websocket)
    logger.info(f"🔌 Client connected to camera {camera_id}")

  def disconnect(self, websocket: WebSocket, camera_id: str):
    """Disconnect a client from a camera stream"""
    if camera_id in self.active_connections:
      self.active_connections[camera_id].discard(websocket)

      # Clean up empty sets
      if not self.active_connections[camera_id]:
        del self.active_connections[camera_id]

    logger.info(f"🔌 Client disconnected from camera {camera_id}")

  async def broadcast_to_camera(self, camera_id: str, message: dict):
    """Broadcast message to all clients watching a camera"""
    if camera_id not in self.active_connections:
      return

    disconnected = set()

    for connection in self.active_connections[camera_id]:
      try:
        await connection.send_json(message)
      except Exception as e:
        logger.error(f"Broadcast error: {e}")
        disconnected.add(connection)

    # Clean up disconnected clients
    for connection in disconnected:
      self.disconnect(connection, camera_id)


manager = ConnectionManager()


class WebSocketHandler:
  """Handle WebSocket frame processing"""

  @staticmethod
  async def parse_binary_frame(data: bytes) -> tuple:
    """Parse binary frame data"""
    try:
      offset = 0

      # Camera ID length and ID
      camera_id_len = data[offset]
      offset += 1
      camera_id = data[offset:offset + camera_id_len].decode('utf-8')
      offset += camera_id_len

      # Timestamp (4 bytes)
      timestamp = int.from_bytes(data[offset:offset + 4], 'little')
      offset += 4

      # Image data
      image_data = data[offset:]
      image_io = BytesIO(image_data)
      pil_image = Image.open(image_io)

      if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

      image_array = np.array(pil_image)

      return camera_id, timestamp, image_array

    except Exception as e:
      logger.error(f"Frame parsing error: {e}")
      return None, None, None

  @staticmethod
  async def process_frame(camera_id: str, image: np.ndarray, db: AsyncSession):
    """Process a single frame with tracking, speed, and distance detection"""
    # Get camera configuration
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
      return {"error": f"Camera {camera_id} not found"}

    # Ensure stream is active
    if not stream_manager.is_active(camera_id):
      stream_manager.add_stream(camera)

    stream = stream_manager.get_stream(camera_id)
    results = {}

    # Run detection on configured models and classes
    detected_objects = []

    if camera.features.get("detection", True):
      for model_name in camera.active_models:
        # Get class filter for this model from camera config
        class_filter = camera.features.get("class_filters", {}).get(model_name)

        model_result = detector.detect(image, model_name, class_filter)
        results[model_name] = model_result.dict()

        # Collect detections for tracking
        for det_dict in model_result.detections:
          det = Detection(**det_dict)
          detected_objects.append(det)

    # Run tracking with object filtering
    tracking_data = None
    if camera.features.get("tracking", False) and stream.tracker:
      # Filter objects for tracking based on selected classes
      tracking_classes = camera.features.get("tracking_classes", [])

      if tracking_classes:
        filtered_detections = [
          det for det in detected_objects
          if det.label in tracking_classes
        ]
      else:
        filtered_detections = detected_objects

      # Update tracker
      tracked_objects = stream.tracker.update(filtered_detections)

      # Build tracking response
      tracking_data = {
        "tracked_objects": {},
        "summary": {
          "total_tracks": len(tracked_objects),
          "active_tracks": sum(
            1 for obj in tracked_objects.values()
            if obj.time_since_update < 5
          )
        }
      }

      for track_id, obj in tracked_objects.items():
        obj_data = {
          "track_id": obj.track_id,
          "class_name": obj.class_name,
          "bbox": obj.bbox,
          "centroid": obj.centroid,
          "confidence": obj.confidence,
          "age": obj.age,
          "velocity": obj.velocity,
          "distance_traveled": obj.distance_traveled,
        }

        # Calculate speed if enabled and object is in speed classes
        speed_classes = camera.features.get("speed_classes", [])
        if camera.features.get("speed", False) and (
          not speed_classes or obj.class_name in speed_classes
        ):
          speed_data = speed_calculator.calculate_speed(
            obj.velocity,
            camera.pixels_per_meter if camera.is_calibrated else None
          )
          obj_data.update(speed_data)

        # Calculate distance if enabled and object is in distance classes
        distance_classes = camera.features.get("distance_classes", [])
        if camera.features.get("distance", False) and camera.is_calibrated and (
          not distance_classes or obj.class_name in distance_classes
        ):
          # Calculate real-world position
          real_x = obj.centroid[0] / camera.pixels_per_meter
          real_y = obj.centroid[1] / camera.pixels_per_meter

          obj_data["position_meters"] = {
            "x": real_x,
            "y": real_y
          }

          # Calculate distance from reference point (camera position)
          distance_from_camera = np.sqrt(real_x ** 2 + real_y ** 2)
          obj_data["distance_from_camera_m"] = distance_from_camera

        tracking_data["tracked_objects"][track_id] = obj_data

      results["tracking"] = tracking_data

    return {
      "camera_id": camera_id,
      "timestamp": int(time.time() * 1000),
      "results": results,
      "calibrated": camera.is_calibrated
    }


@router.websocket("/ws/camera/{camera_id}")
async def camera_websocket(
  websocket: WebSocket,
  camera_id: str
):
  """WebSocket endpoint for individual camera streams"""
  await manager.connect(websocket, camera_id)

  # Get database session
  async for db in get_db():
    try:
      while True:
        # Receive binary frame
        data = await websocket.receive_bytes()

        # Parse frame
        parsed_camera_id, timestamp, image = await WebSocketHandler.parse_binary_frame(data)

        if parsed_camera_id != camera_id:
          await websocket.send_json({
            "error": f"Camera ID mismatch: expected {camera_id}, got {parsed_camera_id}"
          })
          continue

        if image is None:
          await websocket.send_json({"error": "Invalid frame data"})
          continue

        # Process frame
        result = await WebSocketHandler.process_frame(camera_id, image, db)

        # Send response to this client
        await websocket.send_json(result)

        # Optionally broadcast to other clients watching same camera
        # await manager.broadcast_to_camera(camera_id, result)

    except WebSocketDisconnect:
      manager.disconnect(websocket, camera_id)
      logger.info(f"🔌 WebSocket disconnected for camera {camera_id}")
      break
    except Exception as e:
      logger.error(f"WebSocket error for camera {camera_id}: {e}")
      try:
        await websocket.send_json({"error": str(e)})
      except:
        break
      break


@router.websocket("/ws")
async def legacy_websocket(websocket: WebSocket):
  """Legacy WebSocket endpoint (backward compatibility)"""
  await websocket.accept()
  logger.info("🔌 Legacy WebSocket connected")

  async for db in get_db():
    try:
      while True:
        data = await websocket.receive_bytes()

        camera_id, timestamp, image = await WebSocketHandler.parse_binary_frame(data)

        if camera_id is None:
          await websocket.send_json({"error": "Invalid frame data"})
          continue

        result = await WebSocketHandler.process_frame(camera_id, image, db)
        await websocket.send_json(result)

    except WebSocketDisconnect:
      logger.info("🔌 Legacy WebSocket disconnected")
      break
    except Exception as e:
      logger.error(f"Legacy WebSocket error: {e}")
      break