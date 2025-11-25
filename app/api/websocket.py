# app/api/websocket.py - UPDATED VERSION WITH HTTP AND RTSP SUPPORT

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


class StreamManager:
  """Manage RTSP and HTTP streams for cameras"""

  def __init__(self):
    self.streams: Dict[str, cv2.VideoCapture] = {}
    self.running: Dict[str, bool] = {}
    self.stream_types: Dict[str, str] = {}  # Track if stream is 'rtsp' or 'http'

  def start_stream(self, camera_id: str, stream_url: str) -> bool:
    """Start streaming from RTSP or HTTP URL"""
    if camera_id in self.streams:
      self.stop_stream(camera_id)

    # Determine stream type
    stream_type = "http" if stream_url.startswith("http://") or stream_url.startswith("https://") else "rtsp"

    logger.info(f"Opening {stream_type.upper()} stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    # OPTIMIZATION: Set buffer size to 1 for low latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
      logger.error(f"Failed to open {stream_type.upper()} stream: {stream_url}")
      return False

    self.streams[camera_id] = cap
    self.running[camera_id] = True
    self.stream_types[camera_id] = stream_type
    logger.info(f"✅ Started {stream_type.upper()} stream for camera {camera_id}")
    return True

  def stop_stream(self, camera_id: str):
    """Stop streaming for a camera"""
    self.running[camera_id] = False
    if camera_id in self.streams:
      self.streams[camera_id].release()
      del self.streams[camera_id]

      stream_type = self.stream_types.get(camera_id, "stream")
      logger.info(f"🔴 Stopped {stream_type.upper()} stream for camera {camera_id}")

      if camera_id in self.stream_types:
        del self.stream_types[camera_id]

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

  def get_stream_type(self, camera_id: str) -> Optional[str]:
    """Get the type of stream (rtsp or http)"""
    return self.stream_types.get(camera_id)


stream_manager_ws = StreamManager()


def calculate_distance_from_camera(centroid, pixels_per_meter, camera_width, camera_height):
  """Calculate distance from camera using calibration"""
  # Convert pixel position to meters
  real_x = centroid[0] / pixels_per_meter
  real_y = centroid[1] / pixels_per_meter

  # Calculate distance (assuming camera is at origin)
  distance = np.sqrt(real_x ** 2 + real_y ** 2)

  return {
    "position_meters": {"x": real_x, "y": real_y},
    "distance_from_camera_m": distance,
    "distance_from_camera_ft": distance * 3.28084
  }


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
  try:
    camera_id = camera.id

    if not stream_manager.is_active(camera_id):
      logger.info(f"🔄 Adding stream for camera {camera_id}")
      stream_manager.add_stream(camera)

      # Get the EXISTING stream (don't create new one!)
    stream = stream_manager.get_stream(camera_id)
    if not stream:
      logger.error(f"❌ Failed to get stream for camera {camera_id}")
      return {
        "camera_id": camera_id,
        "timestamp": int(time.time() * 1000),
        "results": {},
        "calibrated": False,
        "error": "Stream not available"
      }
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

          # Get class filter from detection_classes or class_filters
          class_filter = None

          # First try detection_classes (new way)
          detection_classes = camera.features.get("detection_classes", [])
          if detection_classes:
            # Filter classes for this specific model
            from app.config import settings as app_settings

            # Build reverse mapping: class -> model
            class_to_model = {}
            for available_model, model_file in app_settings.AVAILABLE_MODELS.items():
              try:
                if available_model in detector.model_classes:
                  model_classes = detector.model_classes[available_model]
                  for class_id, class_name in model_classes.items():
                    class_to_model[class_name] = available_model
              except Exception as e:
                logger.warning(f"Could not map classes for {available_model}: {e}")

            # Filter detection_classes to only include classes from this model
            class_filter = [
              cls for cls in detection_classes
              if class_to_model.get(cls) == model_name
            ]

            if not class_filter:
              logger.debug(f"No classes selected for {model_name}, using all classes")
              class_filter = None
          else:
            # Fall back to old class_filters way
            class_filter = camera.features.get("class_filters", {}).get(model_name)

          logger.debug(f"🤖 Running model: {model_name} with filter: {class_filter}")
          model_result = detector.detect(image, model_name, class_filter)

          # Convert to dict properly
          results[model_name] = model_result.dict()

          logger.debug(f"✅ {model_name}: {model_result.count} detections")

          # Add detections to list for tracking
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
      try:
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
          # Calculate time in seconds based on age (frames) and FPS
          time_in_seconds = obj.age / camera.fps if camera.fps > 0 else obj.age / 30

          obj_data = {
            "track_id": obj.track_id,
            "class_name": obj.class_name,
            "bbox": obj.bbox,
            "centroid": obj.centroid,
            "confidence": obj.confidence,
            "age": obj.age,  # Keep age in frames for reference
            "velocity": obj.velocity,
            "distance_traveled": obj.distance_traveled,
            "time_in_frame_seconds": time_in_seconds,  # NEW: Actual time in seconds
            "time_in_frame_frames": obj.age  # Keep frame count for debugging
          }

          # SPEED CALCULATION
          speed_classes = camera.features.get("speed_classes", [])
          if camera.features.get("speed", False) and (
            not speed_classes or obj.class_name in speed_classes
          ):
            try:
              speed_data = speed_calculator.calculate_speed(
                obj.velocity,
                camera.pixels_per_meter if camera.is_calibrated else None
              )
              obj_data.update(speed_data)
            except Exception as e:
              logger.error(f"Error calculating speed: {e}")

          # DISTANCE CALCULATION - FIXED
          distance_classes = camera.features.get("distance_classes", [])
          if camera.features.get("distance", False) and camera.is_calibrated and (
            not distance_classes or obj.class_name in distance_classes
          ):
            try:
              # Convert pixel position to meters
              real_x = obj.centroid[0] / camera.pixels_per_meter
              real_y = obj.centroid[1] / camera.pixels_per_meter

              # Calculate distance (assuming camera is at origin)
              distance = np.sqrt(real_x ** 2 + real_y ** 2)

              obj_data["position_meters"] = {"x": real_x, "y": real_y}
              obj_data["distance_from_camera_m"] = distance
              obj_data["distance_from_camera_ft"] = distance * 3.28084
            except Exception as e:
              logger.error(f"Error calculating distance: {e}")

          tracking_data["tracked_objects"][track_id] = obj_data

        results["tracking"] = tracking_data
      except Exception as e:
        logger.error(f"❌ Error in tracking: {e}", exc_info=True)

    return {
      "camera_id": camera_id,
      "timestamp": int(time.time() * 1000),
      "results": results,
      "calibrated": camera.is_calibrated
    }

  except Exception as e:
    logger.error(f"❌ Critical error in process_frame_sync: {e}", exc_info=True)
    # CRITICAL: Always return a valid dict, never None
    return {
      "camera_id": camera.id if camera else "unknown",
      "timestamp": int(time.time() * 1000),
      "results": {},
      "calibrated": False,
      "error": str(e)
    }


@router.websocket("/ws/camera/{camera_id}")
async def camera_websocket(websocket: WebSocket, camera_id: str):
  """WebSocket endpoint for individual camera streams - SUPPORTS BOTH HTTP AND RTSP"""
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

      # Handle RTSP or HTTP stream
      if camera.rtsp_url:
        # Determine stream type based on URL
        stream_type = "HTTP" if camera.rtsp_url.startswith("http://") or camera.rtsp_url.startswith(
          "https://") else "RTSP"

        logger.info(f"🎥 Starting {stream_type} stream for {camera_id}: {camera.rtsp_url}")

        if not stream_manager_ws.start_stream(camera_id, camera.rtsp_url):
          await websocket.send_json({
            "camera_id": camera_id,
            "error": f"Failed to open {stream_type} stream: {camera.rtsp_url}"
          })
          break

        await websocket.send_json({
          "camera_id": camera_id,
          "status": "connected",
          "message": f"{stream_type} stream started",
          "stream_type": stream_type.lower(),
          "fps": target_fps
        })

        frame_count = 0
        processed_count = 0

        while stream_manager_ws.is_running(camera_id):
          try:
            # Read frame from stream manager
            frame = stream_manager_ws.get_frame(camera_id)

            if frame is None:
              await asyncio.sleep(0.01)
              continue

            frame_count += 1

            # Check if we should process this frame based on FPS limiter
            if not fps_limiter.should_process_frame():
              continue  # Skip this frame to maintain target FPS

            processed_count += 1

            # FIX: Pass camera object instead of doing DB query
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
              thread_pool,
              process_frame_sync,
              camera,
              frame
            )

            # CRITICAL FIX: Check if result is None
            if result is None:
              logger.error(f"❌ process_frame_sync returned None for camera {camera_id}")
              continue

            # OPTIMIZATION: Reduce JPEG quality for faster encoding
            send_frames = True  # Set to True if frontend wants frames (legacy mode)

            if send_frames:
              try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                result['frame'] = frame_base64
              except Exception as e:
                logger.error(f"❌ Error encoding frame: {e}")
                continue
            else:
              # Don't send frame data - frontend gets video from HLS
              result['frame'] = None

            # Send response
            try:
              await websocket.send_json(result)
            except Exception as e:
              logger.error(f"❌ Error sending WebSocket message: {e}")
              break

            if processed_count % 30 == 0:
              actual_fps = fps_limiter.get_actual_fps()
              current_stream_type = stream_manager_ws.get_stream_type(camera_id)
              logger.info(f"📊 {current_stream_type.upper()} camera {camera_id}:")
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

          except Exception as e:
            logger.error(f"❌ Error in stream processing loop: {e}", exc_info=True)
            await asyncio.sleep(0.1)
            continue

    except WebSocketDisconnect:
      stream_manager_ws.stop_stream(camera_id)
      manager.disconnect(websocket, camera_id)
      logger.info(f"🔴 WebSocket disconnected for camera {camera_id}")
      break
    except Exception as e:
      logger.error(f"❌ WebSocket error for camera {camera_id}: {e}", exc_info=True)
      stream_manager_ws.stop_stream(camera_id)
      try:
        await websocket.send_json({"error": str(e)})
      except:
        pass
      break