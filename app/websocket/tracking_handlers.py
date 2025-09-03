# app/websocket/tracking_handlers.py
import json
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import WebSocket
import cv2
import numpy as np
from ..core.schemas import TrackingMessage, TrackingConfig, TrackingResponse

from ..tracking.tracker_manager import (
  tracker_manager,
  Detection, TrackerType, TrackedObject
)
from ..tracking.analytics import analytics
from ..tracking.speed_calculator import speed_calculator

class TrackingWebSocketHandler:
  """Handle WebSocket messages for object tracking"""

  def __init__(self):
    self.active_connections: Dict[str, WebSocket] = {}
    self.stream_configs: Dict[str, Dict] = {}
    self.tracking_enabled: Dict[str, bool] = {}

  async def handle_tracking_message(self, websocket: WebSocket, message: Dict[str, Any], stream_id: str):
    """Handle different types of tracking messages"""
    message_type = message.get('type')

    try:
      if message_type == 'configure_tracking':
        await self._handle_configure_tracking(websocket, message, stream_id)
      elif message_type == 'start_tracking':
        await self._handle_start_tracking(websocket, message, stream_id)
      elif message_type == 'stop_tracking':
        await self._handle_stop_tracking(websocket, stream_id)
      elif message_type == 'get_tracker_stats':
        await self._handle_get_stats(websocket, stream_id)
      elif message_type == 'define_zone':
        await self._handle_define_zone(websocket, message, stream_id)
      elif message_type == 'update_detections':
        await self._handle_update_detections(websocket, message, stream_id)
      else:
        await self._send_error(websocket, f"Unknown message type: {message_type}")

    except Exception as e:
      await self._send_error(websocket, f"Error handling message: {str(e)}")

  async def _handle_configure_tracking(self, websocket: WebSocket, message: Dict, stream_id: str):
    """Configure tracking parameters"""
    config = message.get('config', {})

    tracker_type = TrackerType(config.get('tracker_type', 'centroid'))
    tracker_params = config.get('tracker_params', {})

    # Configure speed calculation
    if 'speed_config' in config:
      speed_config = config['speed_config']
      speed_calculator.fps = speed_config.get('fps', 30)
      speed_calculator.pixel_to_meter_ratio = speed_config.get('pixel_to_meter_ratio', 1.0)

    # Create or update tracker
    success = tracker_manager.create_tracker(stream_id, tracker_type, **tracker_params)

    if success:
      self.stream_configs[stream_id] = config
      await self._send_response(websocket, {
        'type': 'tracking_configured',
        'stream_id': stream_id,
        'config': config,
        'success': True
      })
    else:
      await self._send_error(websocket, "Failed to configure tracker")

  async def _handle_start_tracking(self, websocket: WebSocket, message: Dict, stream_id: str):
    """Start tracking for a stream"""
    self.tracking_enabled[stream_id] = True
    self.active_connections[stream_id] = websocket

    await self._send_response(websocket, {
      'type': 'tracking_started',
      'stream_id': stream_id,
      'message': 'Object tracking started'
    })

  async def _handle_stop_tracking(self, websocket: WebSocket, stream_id: str):
    """Stop tracking for a stream"""
    self.tracking_enabled[stream_id] = False
    if stream_id in self.active_connections:
      del self.active_connections[stream_id]

    # Optionally remove tracker (or keep for resume)
    # tracker_manager.remove_tracker(stream_id)

    await self._send_response(websocket, {
      'type': 'tracking_stopped',
      'stream_id': stream_id,
      'message': 'Object tracking stopped'
    })

  async def _handle_get_stats(self, websocket: WebSocket, stream_id: str):
    """Get tracker statistics"""
    stats = tracker_manager.get_tracker_stats(stream_id)

    await self._send_response(websocket, {
      'type': 'tracker_stats',
      'stream_id': stream_id,
      'stats': stats
    })

  async def _handle_define_zone(self, websocket: WebSocket, message: Dict, stream_id: str):
    """Define a zone for analytics"""
    zone_id = message.get('zone_id')
    polygon_points = message.get('polygon_points', [])

    if zone_id and polygon_points:
      analytics.define_zone(zone_id, polygon_points)

      await self._send_response(websocket, {
        'type': 'zone_defined',
        'stream_id': stream_id,
        'zone_id': zone_id,
        'success': True
      })
    else:
      await self._send_error(websocket, "Invalid zone definition")

  async def _handle_update_detections(self, websocket: WebSocket, message: Dict, stream_id: str):
    """Process new detections and update tracking"""
    if not self.tracking_enabled.get(stream_id, False):
      return

    detections_data = message.get('detections', [])

    # Convert to Detection objects
    detections = []
    for det_data in detections_data:
      detection = Detection(
        x1=det_data['x1'],
        y1=det_data['y1'],
        x2=det_data['x2'],
        y2=det_data['y2'],
        confidence=det_data['confidence'],
        class_id=det_data['class_id'],
        class_name=det_data['label']
      )
      detections.append(detection)

    # Update tracker
    tracked_objects = tracker_manager.update_tracker(stream_id, detections)

    # Calculate analytics
    tracking_results = await self._calculate_tracking_analytics(stream_id, tracked_objects)

    # Send results back
    await self._send_tracking_results(websocket, stream_id, tracking_results)

  async def _calculate_tracking_analytics(self, stream_id: str, tracked_objects: Dict[str, TrackedObject]) -> Dict:
    """Calculate comprehensive tracking analytics"""
    results = {
      'tracked_objects': {},
      'zone_occupancy': {},
      'speed_analysis': {},
      'line_crossings': [],
      'summary': {
        'total_tracks': len(tracked_objects),
        'active_tracks': 0,
        'class_counts': {}
      }
    }

    # Process each tracked object
    for track_id, obj in tracked_objects.items():
      if obj.time_since_update < 5:  # Consider active if updated recently
        results['summary']['active_tracks'] += 1

      # Count classes
      class_name = obj.class_name
      results['summary']['class_counts'][class_name] = results['summary']['class_counts'].get(class_name, 0) + 1

      # Speed analysis
      speed_data = speed_calculator.calculate_speed(obj)
      avg_speed_data = speed_calculator.calculate_average_speed(obj)

      # Combine speed data
      combined_speed = {**speed_data, **avg_speed_data}
      results['speed_analysis'][track_id] = combined_speed

      # Object data
      results['tracked_objects'][track_id] = {
        'track_id': track_id,
        'class_name': obj.class_name,
        'class_id': obj.class_id,
        'bbox': obj.bbox,
        'centroid': obj.centroid,
        'confidence': obj.confidence,
        'age': obj.age,
        'hits': obj.hits,
        'time_since_update': obj.time_since_update,
        'velocity': obj.velocity,
        'trajectory_length': len(obj.trajectory),
        'metadata': obj.metadata,
        'speed_info': combined_speed
      }

    # Zone occupancy analysis
    if analytics.zone_definitions:
      results['zone_occupancy'] = analytics.check_zone_occupancy(tracked_objects)

    return results

  async def _send_tracking_results(self, websocket: WebSocket, stream_id: str, results: Dict):
    """Send tracking results to client"""
    response = {
      'type': 'tracking_results',
      'stream_id': stream_id,
      'timestamp': asyncio.get_event_loop().time(),
      'results': results
    }

    await self._send_response(websocket, response)

  async def _send_response(self, websocket: WebSocket, data: Dict):
    """Send response to WebSocket client"""
    try:
      await websocket.send_text(json.dumps(data))
    except Exception as e:
      print(f"Error sending WebSocket message: {e}")

  async def _send_error(self, websocket: WebSocket, error_message: str):
    """Send error message to client"""
    await self._send_response(websocket, {
      'type': 'error',
      'message': error_message
    })

