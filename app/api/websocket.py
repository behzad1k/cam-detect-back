# app/api/websocket.py - UPDATED VERSION WITH HTTP AND RTSP SUPPORT

import asyncio
import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Optional, Set

import cv2
import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.detection.yolo_detector import detector
from app.core.tracking.speed_calculator import speed_calculator
from app.database.base import get_db
from app.schemas.alerts import AlertConfiguration
from app.schemas.detection import Detection
from app.services.alert_manager import alert_manager
from app.services.camera_service import camera_service
from app.services.stream_manager import stream_manager
from app.utils.FPSRateLimiter import FPSRateLimiter

router = APIRouter()
logger = logging.getLogger(__name__)

# CRITICAL FIX: Thread pool for CPU-intensive work
thread_pool = ThreadPoolExecutor(max_workers=4)


class AsyncFrameProcessor:
    """Process frames asynchronously without blocking WebSocket"""

    def __init__(self, camera_id: str, max_queue_size: int = 10):
        self.camera_id = camera_id
        self.frame_queue = asyncio.Queue(maxsize=max_queue_size)
        self.result_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = False
        self.process_task = None
        self.frames_submitted = 0
        self.frames_processed = 0
        self.frames_dropped = 0

    async def start(self):
        """Start background processing task"""
        self.processing = True
        self.process_task = asyncio.create_task(self._process_loop())
        logger.info(f"✅ Started async processor for camera {self.camera_id}")

    async def _process_loop(self):
        """Background processing loop"""
        loop = asyncio.get_event_loop()

        while self.processing:
            try:
                # Wait for frame with timeout
                frame, camera = await asyncio.wait_for(
                    self.frame_queue.get(), timeout=1.0
                )

                # Process in thread pool (non-blocking)
                result = await loop.run_in_executor(
                    thread_pool, process_frame_sync, camera, frame
                )

                self.frames_processed += 1

                # Put result (drop oldest if queue full)
                try:
                    self.result_queue.put_nowait(result)
                except asyncio.QueueFull:
                    try:
                        # Drop oldest result
                        old_result = self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                        logger.debug(f"Dropped old result for camera {self.camera_id}")
                    except:
                        pass

            except asyncio.TimeoutError:
                # No frames in queue, continue waiting
                continue
            except asyncio.CancelledError:
                logger.info(f"Processing task cancelled for camera {self.camera_id}")
                break
            except Exception as e:
                logger.error(
                    f"❌ Processing error for camera {self.camera_id}: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(0.1)

    async def submit_frame(self, frame: np.ndarray, camera):
        """Submit frame for processing (non-blocking)"""
        self.frames_submitted += 1

        try:
            # Try to add to queue
            self.frame_queue.put_nowait((frame, camera))
        except asyncio.QueueFull:
            # Queue full, drop this frame
            self.frames_dropped += 1
            if self.frames_dropped % 10 == 1:
                logger.warning(
                    f"⚠️ Processor queue full for camera {self.camera_id}, "
                    f"dropped {self.frames_dropped} frames so far"
                )

    async def get_result(self, timeout: float = 0.01):
        """Get latest result (non-blocking)"""
        try:
            return await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except asyncio.QueueEmpty:
            return None

    def get_stats(self) -> dict:
        """Get processor statistics"""
        return {
            "submitted": self.frames_submitted,
            "processed": self.frames_processed,
            "dropped": self.frames_dropped,
            "queue_size": self.frame_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
        }

    async def stop(self):
        """Stop processing"""
        self.processing = False
        if self.process_task:
            self.process_task.cancel()
            try:
                await self.process_task
            except asyncio.CancelledError:
                pass
        logger.info(f"🛑 Stopped async processor for camera {self.camera_id}")


class StreamManager:
    """Manage RTSP, HTTP, and WEBCAM streams for cameras"""

    def __init__(self):
        self.streams: Dict[str, cv2.VideoCapture] = {}
        self.running: Dict[str, bool] = {}
        self.stream_types: Dict[str, str] = {}
        self.failed_reads: Dict[str, int] = {}

    def start_stream(self, camera_id: str, stream_url: str) -> bool:
        """Start streaming from RTSP, HTTP, or WEBCAM URL"""
        if camera_id in self.streams:
            self.stop_stream(camera_id)

        # Determine stream type
        if stream_url.startswith("webcam://"):
            stream_type = "webcam"
            device_index = int(stream_url.replace("webcam://", ""))
            logger.info(f"Opening WEBCAM stream: device {device_index}")
            cap = cv2.VideoCapture(device_index)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        elif stream_url.startswith("http://") or stream_url.startswith("https://"):
            stream_type = "http"
            logger.info(f"Opening HTTP stream: {stream_url}")

            try:
                logger.info("Trying FFmpeg backend for MJPEG stream...")
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    logger.info(f"✅ Using FFmpeg backend for HTTP stream")
                    cap.release()
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else:
                    logger.warning(
                        f"FFmpeg test failed, falling back to default backend"
                    )
                    cap.release()
                    cap = cv2.VideoCapture(stream_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception as e:
                logger.warning(f"FFmpeg backend error: {e}, using default backend")
                cap = cv2.VideoCapture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        else:
            stream_type = "rtsp"
            logger.info(f"Opening RTSP stream: {stream_url}")
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Larger buffer for RTSP

        if not cap.isOpened():
            logger.error(f"Failed to open {stream_type.upper()} stream: {stream_url}")
            return False

        self.streams[camera_id] = cap
        self.running[camera_id] = True
        self.stream_types[camera_id] = stream_type
        self.failed_reads[camera_id] = 0

        if stream_type == "webcam":
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"✅ Webcam opened: {width}x{height} @ {fps}fps")

        logger.info(f"✅ Started {stream_type.upper()} stream for camera {camera_id}")
        return True

    def stop_stream(self, camera_id: str):
        """Stop streaming for a camera"""
        self.running[camera_id] = False
        if camera_id in self.streams:
            try:
                self.streams[camera_id].release()
            except:
                pass
            del self.streams[camera_id]

            stream_type = self.stream_types.get(camera_id, "stream")
            logger.info(
                f"🔴 Stopped {stream_type.upper()} stream for camera {camera_id}"
            )

            if camera_id in self.stream_types:
                del self.stream_types[camera_id]
            if camera_id in self.failed_reads:
                del self.failed_reads[camera_id]

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get a frame from the stream"""
        if camera_id not in self.streams:
            return None

        cap = self.streams[camera_id]
        ret, frame = cap.read()

        if not ret:
            self.failed_reads[camera_id] = self.failed_reads.get(camera_id, 0) + 1

            if self.failed_reads[camera_id] % 50 == 1:
                logger.warning(
                    f"Failed to read frame from camera {camera_id} "
                    f"({self.failed_reads[camera_id]} failures so far)"
                )

            return None

        # Success! Reset failure counter
        if self.failed_reads.get(camera_id, 0) > 0:
            logger.info(
                f"✅ Camera {camera_id} recovered after {self.failed_reads[camera_id]} failures"
            )
            self.failed_reads[camera_id] = 0

        return frame

    def is_running(self, camera_id: str) -> bool:
        """Check if stream is running"""
        return self.running.get(camera_id, False)

    def get_stream_type(self, camera_id: str) -> Optional[str]:
        """Get the type of stream (rtsp, http, or webcam)"""
        return self.stream_types.get(camera_id)


stream_manager_ws = StreamManager()


def calculate_distance_from_camera(
    centroid, pixels_per_meter, camera_width, camera_height
):
    """Calculate distance from camera using calibration"""
    real_x = centroid[0] / pixels_per_meter
    real_y = centroid[1] / pixels_per_meter

    distance = np.sqrt(real_x**2 + real_y**2)

    return {
        "position_meters": {"x": real_x, "y": real_y},
        "distance_from_camera_m": distance,
        "distance_from_camera_ft": distance * 3.28084,
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


def serialize_for_websocket(obj: Any) -> Any:
    """Recursively serialize objects for WebSocket JSON"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_websocket(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_websocket(item) for item in obj]
    else:
        return obj


def process_frame_sync(camera, image: np.ndarray) -> dict:
    """
    Process frame synchronously in thread pool
    This prevents blocking the asyncio event loop
    """
    try:
        camera_id = camera.id

        # OPTIMIZATION: Resize large frames for detection
        original_shape = image.shape
        scale = 1.0

        if image.shape[1] > 1280:
            scale = 1280 / image.shape[1]
            new_width = 1280
            new_height = int(image.shape[0] * scale)
            image_detect = cv2.resize(image, (new_width, new_height))
            logger.debug(
                f"Resized {original_shape[1]}x{original_shape[0]} → {new_width}x{new_height}"
            )
        else:
            image_detect = image

        if not stream_manager.is_active(camera_id):
            logger.info(f"🔄 Adding stream for camera {camera_id}")
            stream_manager.add_stream(camera)

        stream = stream_manager.get_stream(camera_id)
        if not stream:
            logger.error(f"❌ Failed to get stream for camera {camera_id}")
            return {
                "camera_id": camera_id,
                "timestamp": int(time.time() * 1000),
                "results": {},
                "calibrated": False,
                "error": "Stream not available",
            }

        results = {}
        detected_objects = []

        # DETECTION
        if camera.features.get("detection", True):
            if not camera.active_models or len(camera.active_models) == 0:
                logger.warning(
                    f"⚠️ Camera {camera_id} has detection enabled but no active models!"
                )
            else:
                logger.debug(
                    f"🔍 Running detection with models: {camera.active_models}"
                )

            for model_name in camera.active_models:
                try:
                    if model_name not in settings.AVAILABLE_MODELS:
                        logger.error(f"❌ Model {model_name} not in AVAILABLE_MODELS")
                        results[model_name] = {
                            "detections": [],
                            "count": 0,
                            "model": model_name,
                            "error": f"Model {model_name} not configured",
                        }
                        continue

                    if model_name not in detector.models:
                        logger.info(f"Loading model {model_name}...")
                        success = detector.load_model(model_name)
                        if not success:
                            results[model_name] = {
                                "detections": [],
                                "count": 0,
                                "model": model_name,
                                "error": f"Failed to load model",
                            }
                            continue

                    # Get class filter
                    class_filter = None
                    detection_classes = camera.features.get("detection_classes", [])
                    if detection_classes:
                        from app.config import settings as app_settings

                        class_to_model = {}
                        for (
                            available_model,
                            model_file,
                        ) in app_settings.AVAILABLE_MODELS.items():
                            try:
                                if available_model in detector.model_classes:
                                    model_classes = detector.model_classes[
                                        available_model
                                    ]
                                    for class_id, class_name in model_classes.items():
                                        class_to_model[class_name] = available_model
                            except Exception as e:
                                logger.warning(
                                    f"Could not map classes for {available_model}: {e}"
                                )

                        class_filter = [
                            cls
                            for cls in detection_classes
                            if class_to_model.get(cls) == model_name
                        ]

                        if not class_filter:
                            logger.debug(
                                f"No classes selected for {model_name}, using all classes"
                            )
                            class_filter = None
                    else:
                        class_filter = camera.features.get("class_filters", {}).get(
                            model_name
                        )

                    logger.debug(
                        f"🤖 Running model: {model_name} with filter: {class_filter}"
                    )
                    model_result = detector.detect(
                        image_detect, model_name, class_filter
                    )

                    results[model_name] = model_result.dict()
                    logger.debug(f"✅ {model_name}: {model_result.count} detections")

                    detected_objects.extend(model_result.detections)

                except Exception as e:
                    logger.error(
                        f"❌ Error running model {model_name}: {e}", exc_info=True
                    )
                    results[model_name] = {
                        "detections": [],
                        "count": 0,
                        "model": model_name,
                        "error": str(e),
                    }

        # TRACKING
        tracking_data = None
        if camera.features.get("tracking", False) and stream.tracker:
            try:
                tracking_classes = camera.features.get("tracking_classes", [])

                if tracking_classes:
                    filtered_detections = [
                        det for det in detected_objects if det.label in tracking_classes
                    ]
                else:
                    filtered_detections = detected_objects

                tracked_objects = stream.tracker.update(filtered_detections)

                tracking_data = {
                    "tracked_objects": {},
                    "summary": {
                        "total_tracks": len(tracked_objects),
                        "active_tracks": sum(
                            1
                            for obj in tracked_objects.values()
                            if obj.time_since_update < 5
                        ),
                    },
                }

                for track_id, obj in tracked_objects.items():
                    time_in_seconds = (
                        obj.age / camera.fps if camera.fps > 0 else obj.age / 30
                    )

                    obj_data = {
                        "track_id": obj.track_id,
                        "class_name": obj.class_name,
                        "bbox": obj.bbox,
                        "centroid": obj.centroid,
                        "confidence": obj.confidence,
                        "age": obj.age,
                        "velocity": obj.velocity,
                        "distance_traveled": obj.distance_traveled,
                        "time_in_frame_seconds": time_in_seconds,
                        "time_in_frame_frames": obj.age,
                    }

                    # SPEED CALCULATION
                    speed_classes = camera.features.get("speed_classes", [])
                    if camera.features.get("speed", False) and (
                        not speed_classes or obj.class_name in speed_classes
                    ):
                        try:
                            speed_data = speed_calculator.calculate_speed(
                                obj.velocity,
                                camera.pixels_per_meter
                                if camera.is_calibrated
                                else None,
                            )
                            obj_data.update(speed_data)
                        except Exception as e:
                            logger.error(f"Error calculating speed: {e}")

                    # DISTANCE CALCULATION
                    distance_classes = camera.features.get("distance_classes", [])
                    if (
                        camera.features.get("distance", False)
                        and camera.is_calibrated
                        and (not distance_classes or obj.class_name in distance_classes)
                    ):
                        try:
                            real_x = obj.centroid[0] / camera.pixels_per_meter
                            real_y = obj.centroid[1] / camera.pixels_per_meter

                            distance = np.sqrt(real_x**2 + real_y**2)

                            obj_data["position_meters"] = {"x": real_x, "y": real_y}
                            obj_data["distance_from_camera_m"] = distance
                            obj_data["distance_from_camera_ft"] = distance * 3.28084
                        except Exception as e:
                            logger.error(f"Error calculating distance: {e}")

                    tracking_data["tracked_objects"][track_id] = obj_data

                results["tracking"] = tracking_data
            except Exception as e:
                logger.error(f"❌ Error in tracking: {e}", exc_info=True)

        # ALERT PROCESSING
        if camera.alert_config and tracking_data:
            try:
                alert_config = AlertConfiguration(**camera.alert_config)

                alerts = alert_manager.process_tracking_data(
                    camera_id=camera_id,
                    camera_name=camera.name,
                    camera_email=camera.alert_email,
                    alert_config=alert_config,
                    tracking_data=tracking_data,
                )

                if alerts:
                    results["alerts"] = alerts
                    logger.info(
                        f"🚨 {len(alerts)} alert(s) triggered for camera {camera_id}"
                    )

            except Exception as e:
                logger.error(f"❌ Alert processing error: {e}", exc_info=True)

        return serialize_for_websocket(
            {
                "camera_id": camera_id,
                "timestamp": int(time.time() * 1000),
                "results": results,
                "calibrated": camera.is_calibrated,
            }
        )

    except Exception as e:
        logger.error(f"❌ Critical error in process_frame_sync: {e}", exc_info=True)
        return {
            "camera_id": camera.id if camera else "unknown",
            "timestamp": int(time.time() * 1000),
            "results": {},
            "calibrated": False,
            "error": str(e),
        }


@router.websocket("/ws/camera/{camera_id}")
async def camera_websocket(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for camera streams - ASYNC PROCESSING VERSION"""
    await manager.connect(websocket, camera_id)

    processor = None

    async for db in get_db():
        try:
            camera = await camera_service.get_camera(db, camera_id)
            if not camera:
                await websocket.send_json({"error": f"Camera {camera_id} not found"})
                break

            logger.info(f"📹 Camera {camera_id} config: rtsp_url={camera.rtsp_url}")
            logger.info(f"📹 Active models: {camera.active_models}")

            # Frame rate configuration
            target_fps = camera.fps if camera.fps else 15
            fps_limiter = FPSRateLimiter(target_fps)

            logger.info(f"🎥 Target FPS: {target_fps}")

            # Start stream
            if camera.rtsp_url:
                stream_type = (
                    "HTTP"
                    if camera.rtsp_url.startswith("http://")
                    or camera.rtsp_url.startswith("https://")
                    else "RTSP"
                )

                logger.info(
                    f"🎥 Starting {stream_type} stream for {camera_id}: {camera.rtsp_url}"
                )

                if not stream_manager_ws.start_stream(camera_id, camera.rtsp_url):
                    await websocket.send_json(
                        {
                            "camera_id": camera_id,
                            "error": f"Failed to open {stream_type} stream: {camera.rtsp_url}",
                        }
                    )
                    break

                await websocket.send_json(
                    {
                        "camera_id": camera_id,
                        "status": "connected",
                        "message": f"{stream_type} stream started",
                        "stream_type": stream_type.lower(),
                        "fps": target_fps,
                    }
                )

                # Create async processor
                processor = AsyncFrameProcessor(camera_id, max_queue_size=10)
                await processor.start()

                frame_count = 0
                read_count = 0
                last_stats_time = time.time()
                last_result_time = time.time()

                while stream_manager_ws.is_running(camera_id):
                    try:
                        # Read frame from stream
                        frame = stream_manager_ws.get_frame(camera_id)

                        if frame is None:
                            await asyncio.sleep(0.01)
                            continue

                        read_count += 1

                        # Check if we should process this frame
                        if not fps_limiter.should_process_frame():
                            continue

                        frame_count += 1

                        # Submit frame to async processor (non-blocking)
                        await processor.submit_frame(frame, camera)

                        # Try to get result (non-blocking with short timeout)
                        result = await processor.get_result(timeout=0.001)

                        if result is not None:
                            last_result_time = time.time()

                            # OPTIONAL: Add frame to result
                            # Set to False to improve performance
                            send_frames = True

                            if send_frames:
                                try:
                                    _, buffer = cv2.imencode(
                                        ".webp", frame, [cv2.IMWRITE_WEBP_QUALITY, 80]
                                    )
                                    frame_base64 = base64.b64encode(buffer).decode(
                                        "utf-8"
                                    )
                                    result["frame"] = frame_base64
                                except Exception as e:
                                    logger.error(f"❌ Error encoding frame: {e}")
                            else:
                                result["frame"] = None

                            # Send result
                            try:
                                await websocket.send_json(result)
                            except Exception as e:
                                logger.error(f"❌ Error sending WebSocket message: {e}")
                                break

                        # Log stats periodically
                        current_time = time.time()
                        if current_time - last_stats_time >= 30:
                            actual_fps = fps_limiter.get_actual_fps()
                            processor_stats = processor.get_stats()
                            current_stream_type = stream_manager_ws.get_stream_type(
                                camera_id
                            )

                            logger.info(
                                f"📊 {current_stream_type.upper()} camera {camera_id}:"
                            )
                            logger.info(f"   Read: {read_count} frames")
                            logger.info(f"   Processed: {frame_count} frames")
                            logger.info(f"   Target FPS: {target_fps}")
                            logger.info(f"   Actual FPS: {actual_fps:.2f}")
                            logger.info(
                                f"   Processor: submitted={processor_stats['submitted']}, "
                                f"processed={processor_stats['processed']}, "
                                f"dropped={processor_stats['dropped']}"
                            )
                            logger.info(
                                f"   Queue: frames={processor_stats['queue_size']}, "
                                f"results={processor_stats['result_queue_size']}"
                            )

                            if result and result.get("results"):
                                total_detections = sum(
                                    r.get("count", 0)
                                    for r in result["results"].values()
                                    if isinstance(r, dict) and "count" in r
                                )
                                logger.info(f"   Total detections: {total_detections}")

                                if result["results"].get("tracking"):
                                    tracking = result["results"]["tracking"]
                                    logger.info(
                                        f"   Active tracks: {tracking['summary']['active_tracks']}"
                                    )

                            last_stats_time = current_time

                        # Small yield to prevent CPU hogging
                        await asyncio.sleep(0.001)

                    except Exception as e:
                        logger.error(
                            f"❌ Error in stream processing loop: {e}", exc_info=True
                        )
                        await asyncio.sleep(0.1)
                        continue

        except WebSocketDisconnect:
            stream_manager_ws.stop_stream(camera_id)
            if processor:
                await processor.stop()
            manager.disconnect(websocket, camera_id)
            logger.info(f"🔴 WebSocket disconnected for camera {camera_id}")
            break
        except Exception as e:
            logger.error(
                f"❌ WebSocket error for camera {camera_id}: {e}", exc_info=True
            )
            stream_manager_ws.stop_stream(camera_id)
            if processor:
                await processor.stop()
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
            break
        finally:
            # Cleanup
            if processor:
                await processor.stop()
