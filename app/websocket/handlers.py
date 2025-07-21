import json
import logging
from fastapi import WebSocket, WebSocketDisconnect

from app.websocket.connection_manager import manager
from app.utils.frame_processor import frame_processor

logger = logging.getLogger(__name__)


async def websocket_endpoint(websocket: WebSocket):
  """Main WebSocket endpoint handler"""
  await manager.connect(websocket)

  try:
    while True:
      # Receive binary data from client
      data = await websocket.receive_bytes()

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

        # Create and send response
        response = frame_processor.create_response(results, timestamp_ms)
        await manager.send_json_message(response, websocket)

      except Exception as e:
        logger.error(f"Error processing frame: {e}")
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
