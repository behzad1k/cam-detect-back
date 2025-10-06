from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import json
import logging
import time
import numpy as np
from io import BytesIO
from PIL import Image

from app.database.base import get_db
from app.services.camera_service import camera_service
from app.services.stream_manager import stream_manager
from app.core.detection.yolo_detector import detector
from app.core.tracking.speed_calculator import speed_calculator
from app.schemas.detection import Detection

router = APIRouter()
logger = logging.getLogger(__name__)


class WebSocketHandler:
  """Handle WebSocket connections for camera streams"""

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
    """Process a single frame"""
    # Get camera configuration
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
      return {"error": f"Camera {camera_id} not found"}

    # Ensure stream is active
    if not stream_manager.is_active(camera_id):
      stream_manager.add_stream(camera)

    stream = stream_manager.get_stream(camera_id)
    results = {}

    # Run detection
    if camera.features.get("detection", True):
      for model_name in camera.active_models:
        model_result = detector.detect(image, model_name)
        results[model_name] = model_result.dict()

    # Run tracking
    if camera.features.get("tracking", False) and stream.tracker:
      all_detections = []

      # Collect all detections
      for model_result in results.values():
        for det_dict in model_result["detections"]:
          det = Detection(**det_dict)
          all_detections.append(det)

      # Update tracker
      tracked_objects = stream.tracker.update(all_detections)

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
        # Calculate speed
        speed_data = {}
        if camera.features.get("speed", False):
          speed_data = speed_calculator.calculate_speed(
            obj.velocity,
            camera.pixels_per_meter if camera.is_calibrated else None
          )

        tracking_data["tracked_objects"][track_id] = {
          "track_id": obj.track_id,
          "class_name": obj.class_name,
          "bbox": obj.bbox,
          "centroid": obj.centroid,
          "confidence": obj.confidence,
          "age": obj.age,
          "velocity": obj.velocity,
          "distance_traveled": obj.distance_traveled,
          **speed_data
        }

      results["tracking"] = tracking_data

    return {
      "camera_id": camera_id,
      "timestamp": int(time.time() * 1000),
      "results": results,
      "calibrated": camera.is_calibrated
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
  """WebSocket endpoint for real-time frame processing"""
  await websocket.accept()
  logger.info("🔌 WebSocket connected")

  # Get database session
  async for db in get_db():
    try:
      while True:
        # Receive binary frame
        data = await websocket.receive_bytes()

        # Parse frame
        camera_id, timestamp, image = await WebSocketHandler.parse_binary_frame(data)

        if camera_id is None:
          await websocket.send_json({"error": "Invalid frame data"})
          continue

        # Process frame
        result = await WebSocketHandler.process_frame(camera_id, image, db)

        # Send response
        await websocket.send_json(result)

    except WebSocketDisconnect:
      logger.info("🔌 WebSocket disconnected")
      break
    except Exception as e:
      logger.error(f"WebSocket error: {e}")
      try:
        await websocket.send_json({"error": str(e)})
      except:
        break