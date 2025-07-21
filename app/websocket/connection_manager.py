import json
import logging
from typing import List
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
  def __init__(self):
    self.active_connections: List[WebSocket] = []

  async def connect(self, websocket: WebSocket):
    await websocket.accept()
    self.active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

  def disconnect(self, websocket: WebSocket):
    if websocket in self.active_connections:
      self.active_connections.remove(websocket)
    logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

  async def send_personal_message(self, message: str, websocket: WebSocket):
    try:
      await websocket.send_text(message)
    except Exception as e:
      logger.error(f"Error sending message: {e}")
      self.disconnect(websocket)

  async def send_json_message(self, data: dict, websocket: WebSocket):
    await self.send_personal_message(json.dumps(data), websocket)


manager = ConnectionManager()
