import json
import logging
import time
import cv2
from fastapi import WebSocket, WebSocketDisconnect

from app.websocket.connection_manager import manager
from app.utils.frame_processor import frame_processor

logger = logging.getLogger(__name__)


async def websocket_endpoint(websocket: WebSocket, frame_skip: int = 0):
  """Main WebSocket endpoint handler with frame skipping"""

  frame_skip = 0
  last_processing_time = time.time()


  await manager.connect(websocket)
  frame_counter = 2  # Initialize frame counter

  try:
    while True:
      # Receive binary data from client
      data = await websocket.receive_bytes()
      frame_counter += 1

      # Auto-adjust frame skip based on processing time
      current_time = time.time()
      processing_delay = current_time - last_processing_time

      if processing_delay > 0.1:  # If processing takes more than 100ms
        frame_skip = min(frame_skip + 1, 10)  # Gradually increase skip up to 10
        logger.warning(f"Skipping frame {frame_counter} (frame_skip={frame_skip})")
      elif frame_skip > 0:
        frame_skip -= 1  # Gradually decrease skip when system can handle it

      # Skip frames according to frame_skip rate
      # if frame_skip > 0 and frame_counter % (frame_skip + 1) != 0:
      #   logger.debug(f"Skipping frame {frame_counter} (frame_skip={frame_skip})")
      #   continue

      try:
        # Parse the incoming data
        timestamp_ms, model_names, class_filters, frame = frame_processor.parse_websocket_data(data)

        # Process frame with requested models and class filters
        results = await frame_processor.process_frame(
          frame,
          model_names,
          class_filters,
          timestamp_ms
        )
        logger.debug(f"Processing frame {frame_counter}: {len(data)} bytes, models: {model_names}")

        # Create and send response
        response = frame_processor.create_response(results, timestamp_ms)
        await manager.send_json_message(response, websocket)

      except Exception as e:
        logger.error(f"Error processing frame {frame_counter}: {e}")
        error_response = frame_processor.create_error_response(
          str(e),
          timestamp_ms if 'timestamp_ms' in locals() else 0
        )
        await manager.send_json_message(error_response, websocket)

  except WebSocketDisconnect:
    manager.disconnect(websocket)
  except Exception as e:
    logger.error(f"WebSocket error: {e}")
    manager.disconnect(websocket)