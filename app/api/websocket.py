# app/api/websocket.py - FIXED VERSION

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
from concurrent.futures import ThreadPoolExecutor

from app.database.base import get_db
from app.services.camera_service import camera_service
from app.services.stream_manager import stream_manager
from app.core.detection.yolo_detector import detector
from app.core.tracking.speed_calculator import speed_calculator
from app.schemas.detection import Detection
from app.config import settings
from app.utils.FPSRateLimiter import FPSRateLimiter

router = APIRouter()
logger = logging.getLogger(__name__)

# CRITICAL FIX: Thread pool for CPU-intensive work
thread_pool = ThreadPoolExecutor(max_workers=4)


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

    # OPTIMIZATION: Set buffer size to 1 for low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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


# CRITICAL FIX: Synchronous processing function for thread pool
def process_frame_sync(camera, image: np.ndarray) -> dict:
    """
    Process frame synchronously in thread pool
    This prevents blocking the asyncio event loop
    """
    camera_id = camera.id

    if not stream_manager.is_active(camera_id):
      stream_manager.add_stream(camera)

    stream = stream_manager.get_stream(camera_id)
    results = {}
    detected_objects = []

    # DETECTION with proper error handling
    if camera.features.get("detection", True):
      if not camera.active_models or len(camera.active_models) == 0:
        logger.warning(f"⚠️ Camera {camera_id} has detection enabled but no active models!")
      else:
        logger.debug(f"🔍 Running detection with models: {camera.active_models}")

      for model_name in camera.active_models:
        try:
          # Check if model file exists
          if model_name not in settings.AVAILABLE_MODELS:
            logger.error(f"❌ Model {model_name} not in AVAILABLE_MODELS")
            results[model_name] = {
              "detections": [],
              "count": 0,
              "model": model_name,
              "error": f"Model {model_name} not configured"
            }
            continue

          # Try to load model if not loaded
          if model_name not in detector.models:
            logger.info(f"Loading model {model_name}...")
            success = detector.load_model(model_name)
            if not success:
              results[model_name] = {
                "detections": [],
                "count": 0,
                "model": model_name,
                "error": f"Failed to load model"
              }
              continue

          class_filter = camera.features.get("class_filters", {}).get(model_name)

          logger.debug(f"🤖 Running model: {model_name} with filter: {class_filter}")
          model_result = detector.detect(image, model_name, class_filter)

          # FIX: Convert to dict properly
          results[model_name] = model_result.dict()

          logger.debug(f"✅ {model_name}: {model_result.count} detections")

          # FIX: model_result.detections are already Detection objects
          # Don't try to convert them again
          detected_objects.extend(model_result.detections)

        except Exception as e:
          logger.error(f"❌ Error running model {model_name}: {e}", exc_info=True)
          results[model_name] = {
            "detections": [],
            "count": 0,
            "model": model_name,
            "error": str(e)
          }

    # TRACKING
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
  """WebSocket endpoint for individual camera streams - OPTIMIZED WITH FPS LIMITER"""
  await manager.connect(websocket, camera_id)

  # FIX: Get camera ONCE at the start, not for every frame
  async for db in get_db():
    try:
      camera = await camera_service.get_camera(db, camera_id)
      if not camera:
        await websocket.send_json({"error": f"Camera {camera_id} not found"})
        break

      logger.info(f"📹 Camera {camera_id} config: rtsp_url={camera.rtsp_url}")
      logger.info(f"📹 Active models: {camera.active_models}")

      # Frame rate configuration with FPS limiter
      target_fps = camera.fps if camera.fps else 15
      fps_limiter = FPSRateLimiter(target_fps)  # NEW: Initialize FPS limiter

      logger.info(f"🎥 Target FPS: {target_fps}")

      # Handle webcam
      if camera.rtsp_url and camera.rtsp_url.startswith('webcam://'):
        await websocket.send_json({
          "camera_id": camera_id,
          "status": "connected",
          "message": "Webcam mode - ready to receive frames",
          "fps": target_fps
        })

        # Webcam mode: wait for client to send frames
        logger.info(f"📹 Webcam mode enabled for camera {camera_id}")

        frame_count = 0
        processed_count = 0

        while True:
          try:
            # Receive binary frame from client
            data = await websocket.receive_bytes()

            frame_count += 1

            # NEW: Check if we should process this frame based on FPS limiter
            if not fps_limiter.should_process_frame():
              continue  # Skip this frame to maintain target FPS

            # Parse binary data
            offset = 0
            camera_id_len = data[offset]
            offset += 1
            received_camera_id = data[offset:offset + camera_id_len].decode('utf-8')
            offset += camera_id_len

            # Verify camera ID matches
            if received_camera_id != camera_id:
              logger.warning(f"Camera ID mismatch: expected {camera_id}, got {received_camera_id}")
              continue

            timestamp = int.from_bytes(data[offset:offset + 4], 'little')
            offset += 4

            # Extract image data
            image_data = data[offset:]
            image_io = BytesIO(image_data)
            pil_image = Image.open(image_io)

            if pil_image.mode != 'RGB':
              pil_image = pil_image.convert('RGB')

            image_array = np.array(pil_image)

            processed_count += 1

            # CRITICAL FIX: Process frame in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                thread_pool,
                process_frame_sync,
                camera,
                image_array
            )

            # Encode frame back to client
            _, buffer = cv2.imencode('.jpg', image_array, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            result['frame'] = frame_base64

            # Send response
            await websocket.send_json(result)

            if processed_count % 30 == 0:
              actual_fps = fps_limiter.get_actual_fps()
              logger.info(f"📊 Webcam camera {camera_id}:")
              logger.info(f"   Received: {frame_count} frames")
              logger.info(f"   Processed: {processed_count} frames")
              logger.info(f"   Target FPS: {target_fps}")
              logger.info(f"   Actual FPS: {actual_fps:.2f}")

          except WebSocketDisconnect:
            logger.info(f"🔴 Webcam WebSocket disconnected for camera {camera_id}")
            break
          except Exception as e:
            logger.error(f"❌ Error processing webcam frame for {camera_id}: {e}", exc_info=True)
            await asyncio.sleep(0.1)
            continue

        break

      # Handle RTSP stream
      if camera.rtsp_url and not camera.rtsp_url.startswith('webcam://'):
        logger.info(f"🎥 Starting RTSP stream for {camera_id}: {camera.rtsp_url}")

        if not rtsp_manager.start_stream(camera_id, camera.rtsp_url):
          await websocket.send_json({
            "camera_id": camera_id,
            "error": f"Failed to open RTSP stream: {camera.rtsp_url}"
          })
          break

        await websocket.send_json({
          "camera_id": camera_id,
          "status": "connected",
          "message": "RTSP stream started",
          "fps": target_fps
        })

        frame_count = 0
        processed_count = 0

        while rtsp_manager.is_running(camera_id):
          # Read frame from RTSP manager
          frame = rtsp_manager.get_frame(camera_id)

          if frame is None:
            await asyncio.sleep(0.01)
            continue

          frame_count += 1

          # NEW: Check if we should process this frame based on FPS limiter
          if not fps_limiter.should_process_frame():
            continue  # Skip this frame to maintain target FPS

          processed_count += 1

          # OPTIMIZATION: Resize frame for faster processing
          # if frame.shape[0] > 720:
          #   scale = 720 / frame.shape[0]
          #   new_width = int(frame.shape[1] * scale)
          #   frame = cv2.resize(frame, (new_width, 720))

          # FIX: Pass camera object instead of doing DB query
          loop = asyncio.get_event_loop()
          result = await loop.run_in_executor(
            thread_pool,
            process_frame_sync,
            camera,
            frame
          )
          # OPTIMIZATION: Reduce JPEG quality for faster encoding
          _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
          frame_base64 = base64.b64encode(buffer).decode('utf-8')
          result['frame'] = frame_base64

          # Send response
          await websocket.send_json(result)

          if processed_count % 30 == 0:
            actual_fps = fps_limiter.get_actual_fps()
            logger.info(f"📊 RTSP camera {camera_id}:")
            logger.info(f"   Read: {frame_count} frames")
            logger.info(f"   Processed: {processed_count} frames")
            logger.info(f"   Target FPS: {target_fps}")
            logger.info(f"   Actual FPS: {actual_fps:.2f}")

            if result.get('results'):
              total_detections = sum(
                r.get('count', 0) for r in result['results'].values()
                if isinstance(r, dict) and 'count' in r
              )
              logger.info(f"   Total detections: {total_detections}")

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
