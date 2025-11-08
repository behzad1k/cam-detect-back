# COMPLETE REPLACEMENT for app/api/websocket.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import json
import logging
import time
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Set, Optional
import asyncio

from app.database.base import get_db
from app.services.camera_service import camera_service
from app.services.stream_manager import stream_manager
from app.core.detection.yolo_detector import detector
from app.core.tracking.speed_calculator import speed_calculator
from app.schemas.detection import Detection

router = APIRouter()
logger = logging.getLogger(__name__)


class RTSPStreamManager:
  """Manage RTSP streams for cameras"""

  def __init__(self):
    self.streams: Dict[str, cv2.VideoCapture] = {}
    self.running: Dict[str, bool] = {}

  def start_stream(self, camera_id: str, rtsp_url: str) -> bool:
    """Start streaming from RTSP URL"""
    if camera_id in self.streams:
      self.stop_stream(camera_id)

    logger.info(f"Opening RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
      logger.error(f"Failed to open RTSP stream: {rtsp_url}")
      return False

    self.streams[camera_id] = cap
    self.running[camera_id] = True
    logger.info(f"✅ Started RTSP stream for camera {camera_id}")
    return True

  def stop_stream(self, camera_id: str):
    """Stop streaming for a camera"""
    self.running[camera_id] = False
    if camera_id in self.streams:
      self.streams[camera_id].release()
      del self.streams[camera_id]
      logger.info(f"🔴 Stopped RTSP stream for camera {camera_id}")

  def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
    """Get a frame from the stream"""
    if camera_id not in self.streams:
      return None

    cap = self.streams[camera_id]
    ret, frame = cap.read()

    if not ret:
      logger.warning(f"Failed to read frame from camera {camera_id}")
      return None

    return frame

  def is_running(self, camera_id: str) -> bool:
    """Check if stream is running"""
    return self.running.get(camera_id, False)


# Global instances
rtsp_manager = RTSPStreamManager()


class ConnectionManager:
  """Manage WebSocket connections per camera"""

  def __init__(self):
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

      if not self.active_connections[camera_id]:
        del self.active_connections[camera_id]

    logger.info(f"🔌 Client disconnected from camera {camera_id}")


manager = ConnectionManager()


class WebSocketHandler:
  """Handle WebSocket frame processing"""

  @staticmethod
  async def process_frame(camera_id: str, image: np.ndarray, db: AsyncSession):
    """Process a single frame with tracking, speed, and distance detection"""
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
      return {"error": f"Camera {camera_id} not found"}

    if not stream_manager.is_active(camera_id):
      stream_manager.add_stream(camera)

    stream = stream_manager.get_stream(camera_id)
    results = {}

    detected_objects = []

    if camera.features.get("detection", True):
      for model_name in camera.active_models:
        class_filter = camera.features.get("class_filters", {}).get(model_name)

        model_result = detector.detect(image, model_name, class_filter)
        results[model_name] = model_result.dict()

        for det_dict in model_result.detections:
          det = Detection(**det_dict)
          detected_objects.append(det)

    tracking_data = None
    if camera.features.get("tracking", False) and stream.tracker:
      tracking_classes = camera.features.get("tracking_classes", [])

      if tracking_classes:
        filtered_detections = [
          det for det in detected_objects
          if det.label in tracking_classes
        ]
      else:
        filtered_detections = detected_objects

      tracked_objects = stream.tracker.update(filtered_detections)

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

        speed_classes = camera.features.get("speed_classes", [])
        if camera.features.get("speed", False) and (
          not speed_classes or obj.class_name in speed_classes
        ):
          speed_data = speed_calculator.calculate_speed(
            obj.velocity,
            camera.pixels_per_meter if camera.is_calibrated else None
          )
          obj_data.update(speed_data)

        distance_classes = camera.features.get("distance_classes", [])
        if camera.features.get("distance", False) and camera.is_calibrated and (
          not distance_classes or obj.class_name in distance_classes
        ):
          real_x = obj.centroid[0] / camera.pixels_per_meter
          real_y = obj.centroid[1] / camera.pixels_per_meter

          obj_data["position_meters"] = {
            "x": real_x,
            "y": real_y
          }

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
async def camera_websocket(websocket: WebSocket, camera_id: str):
  """WebSocket endpoint for individual camera streams"""
  await manager.connect(websocket, camera_id)

  async for db in get_db():
    try:
      # Get camera configuration
      camera = await camera_service.get_camera(db, camera_id)
      if not camera:
        await websocket.send_json({"error": f"Camera {camera_id} not found"})
        break

      logger.info(f"📹 Camera {camera_id} config: rtsp_url={camera.rtsp_url}")

      # If camera has RTSP URL, stream from it
      if camera.rtsp_url:
        logger.info(f"🎥 Starting RTSP stream for {camera_id}: {camera.rtsp_url}")

        if not rtsp_manager.start_stream(camera_id, camera.rtsp_url):
          await websocket.send_json({
            "camera_id": camera_id,
            "error": f"Failed to open RTSP stream: {camera.rtsp_url}"
          })
          break

        # Send initial connection message
        await websocket.send_json({
          "camera_id": camera_id,
          "status": "connected",
          "message": "RTSP stream started"
        })

        frame_count = 0
        fps = camera.fps if camera.fps else 15
        frame_delay = fps / 1.0

        # Stream frames continuously
        while rtsp_manager.is_running(camera_id):
          frame = rtsp_manager.get_frame(camera_id)

          if frame is None:
            logger.warning(f"No frame from camera {camera_id}, retrying...")
            await asyncio.sleep(0.1)
            continue

          frame_count += 1

          # Process frame with detection/tracking
          result = await WebSocketHandler.process_frame(camera_id, frame, db)

          # Encode frame to base64
          _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
          frame_base64 = base64.b64encode(buffer).decode('utf-8')
          result['frame'] = frame_base64

          # Send response
          await websocket.send_json(result)

          if frame_count % 30 == 0:
            logger.info(f"📊 Sent {frame_count} frames for camera {camera_id}")

          # Control frame rate
          await asyncio.sleep(frame_delay)

      else:
        # No RTSP URL - expect frames from client (webcam mode)
        logger.info(f"📷 Waiting for frames from client for camera {camera_id}")

        await websocket.send_json({
          "camera_id": camera_id,
          "status": "connected",
          "message": "Ready to receive frames"
        })

        # while True:
        #   data = await websocket.receive_bytes()
        #
        #   # Parse binary frame (legacy format)
        #   try:
        #     offset = 0
        #     camera_id_len = data[offset]
        #     offset += 1
        #     parsed_camera_id = data[offset:offset + camera_id_len].decode('utf-8')
        #     offset += camera_id_len
        #
        #     timestamp = int.from_bytes(data[offset:offset + 4], 'little')
        #     offset += 4
        #
        #     image_data = data[offset:]
        #     image_io = BytesIO(image_data)
        #     pil_image = Image.open(image_io)
        #
        #     if pil_image.mode != 'RGB':
        #       pil_image = pil_image.convert('RGB')
        #
        #     image_array = np.array(pil_image)
        #
        #     if parsed_camera_id != camera_id:
        #       await websocket.send_json({"error": f"Camera ID mismatch"})
        #       continue
        #
        #     result = await WebSocketHandler.process_frame(camera_id, image_array, db)
        #     await websocket.send_json(result)
        #
        #   except Exception as e:
        #     logger.error(f"Error parsing frame: {e}")
        #     await websocket.send_json({"error": "Invalid frame data"})

    except WebSocketDisconnect:
      rtsp_manager.stop_stream(camera_id)
      manager.disconnect(websocket, camera_id)
      logger.info(f"🔴 WebSocket disconnected for camera {camera_id}")
      break
    except Exception as e:
      logger.error(f"❌ WebSocket error for camera {camera_id}: {e}", exc_info=True)
      rtsp_manager.stop_stream(camera_id)
      try:
        await websocket.send_json({"error": str(e)})
      except:
        pass
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

        offset = 0
        camera_id_len = data[offset]
        offset += 1
        camera_id = data[offset:offset + camera_id_len].decode('utf-8')
        offset += camera_id_len

        timestamp = int.from_bytes(data[offset:offset + 4], 'little')
        offset += 4

        image_data = data[offset:]
        image_io = BytesIO(image_data)
        pil_image = Image.open(image_io)

        if pil_image.mode != 'RGB':
          pil_image = pil_image.convert('RGB')

        image_array = np.array(pil_image)

        result = await WebSocketHandler.process_frame(camera_id, image_array, db)
        await websocket.send_json(result)

    except WebSocketDisconnect:
      logger.info("🔴 Legacy WebSocket disconnected")
      break
    except Exception as e:
      logger.error(f"❌ Legacy WebSocket error: {e}")
      break