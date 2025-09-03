# app/api/tracking_routes.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from ..core.schemas import TrackingConfig, ZoneDefinition, LineDefinition, TrackerTypeEnum
from ..tracking import tracker_manager, analytics
from ..utils.visualization import TrackingVisualizer

router = APIRouter(prefix="/tracking", tags=["tracking"])


@router.post("/streams/{stream_id}/configure")
async def configure_tracking(stream_id: str, config: TrackingConfig):
  """Configure tracking for a specific stream"""
  try:
    tracker_type = tracker_manager.TrackerType(config.tracker_type.value)
    success = tracker_manager.create_tracker(
      stream_id,
      tracker_type,
      **config.tracker_params
    )

    if not success:
      raise HTTPException(status_code=400, detail="Failed to configure tracker")

    return {
      "message": "Tracking configured successfully",
      "stream_id": stream_id,
      "config": config.dict()
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/{stream_id}/status")
async def get_tracking_status(stream_id: str):
  """Get tracking status and statistics for a stream"""
  try:
    stats = tracker_manager.get_tracker_stats(stream_id)
    tracked_objects = tracker_manager.get_tracked_objects(stream_id)

    return {
      "stream_id": stream_id,
      "is_active": stream_id in tracker_manager.active_streams,
      "stats": stats,
      "object_count": len(tracked_objects)
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/{stream_id}/objects")
async def get_tracked_objects(stream_id: str):
  """Get all currently tracked objects for a stream"""
  try:
    tracked_objects = tracker_manager.get_tracked_objects(stream_id)

    # Convert to serializable format
    serialized_objects = {}
    for track_id, obj in tracked_objects.items():
      serialized_objects[track_id] = {
        "track_id": obj.track_id,
        "class_name": obj.class_name,
        "class_id": obj.class_id,
        "bbox": obj.bbox,
        "centroid": obj.centroid,
        "confidence": obj.confidence,
        "age": obj.age,
        "hits": obj.hits,
        "time_since_update": obj.time_since_update,
        "velocity": obj.velocity,
        "trajectory_length": len(obj.trajectory),
        "last_seen": obj.last_seen
      }

    return {
      "stream_id": stream_id,
      "tracked_objects": serialized_objects,
      "total_count": len(serialized_objects)
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@router.post("/streams/{stream_id}/zones")
async def define_zone(stream_id: str, zone: ZoneDefinition):
  """Define a zone for analytics"""
  try:
    analytics.define_zone(zone.zone_id, zone.polygon_points)

    return {
      "message": "Zone defined successfully",
      "stream_id": stream_id,
      "zone_id": zone.zone_id
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/{stream_id}/zones")
async def get_zones(stream_id: str):
  """Get all defined zones for a stream"""
  return {
    "stream_id": stream_id,
    "zones": analytics.zone_definitions
  }


@router.get("/streams/{stream_id}/analytics")
async def get_analytics(stream_id: str):
  """Get comprehensive analytics for tracked objects"""
  try:
    tracked_objects = tracker_manager.get_tracked_objects(stream_id)

    if not tracked_objects:
      return {
        "stream_id": stream_id,
        "analytics": {},
        "message": "No tracked objects found"
      }

    # Zone occupancy
    zone_occupancy = analytics.check_zone_occupancy(tracked_objects)

    # Speed analysis
    from ..tracking.tracker_manager import speed_calculator
    speed_analysis = {}
    for track_id, obj in tracked_objects.items():
      speed_data = speed_calculator.calculate_speed(obj)
      avg_speed_data = speed_calculator.calculate_average_speed(obj)
      speed_analysis[track_id] = {**speed_data, **avg_speed_data}

    # Movement patterns
    movement_patterns = {}
    for track_id, obj in tracked_objects.items():
      if len(obj.trajectory) > 2:
        trajectory_points = list(obj.trajectory)
        movement_patterns[track_id] = {
          "trajectory_length": len(trajectory_points),
          "start_position": trajectory_points[0],
          "current_position": trajectory_points[-1],
          "path_covered": len(trajectory_points)
        }

    return {
      "stream_id": stream_id,
      "analytics": {
        "zone_occupancy": zone_occupancy,
        "speed_analysis": speed_analysis,
        "movement_patterns": movement_patterns,
        "summary": {
          "total_objects": len(tracked_objects),
          "active_objects": len([obj for obj in tracked_objects.values()
                                 if obj.time_since_update < 5]),
          "zones_monitored": len(analytics.zone_definitions)
        }
      }
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@router.delete("/streams/{stream_id}")
async def remove_tracker(stream_id: str):
  """Remove tracker for a stream"""
  try:
    tracker_manager.remove_tracker(stream_id)
    return {
      "message": "Tracker removed successfully",
      "stream_id": stream_id
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams")
async def list_active_streams():
  """List all active tracking streams"""
  return {
    "active_streams": list(tracker_manager.active_streams),
    "total_count": len(tracker_manager.active_streams)
  }


@router.post("/streams/{stream_id}/reset")
async def reset_tracker(stream_id: str):
  """Reset tracker for a stream (clear all tracked objects)"""
  try:
    if stream_id in tracker_manager.trackers:
      # Get current config
      current_config = tracker_manager.tracker_configs[stream_id]

      # Remove and recreate
      tracker_manager.remove_tracker(stream_id)
      success = tracker_manager.create_tracker(
        stream_id,
        current_config['type'],
        **current_config['config']
      )

      if not success:
        raise HTTPException(status_code=400, detail="Failed to reset tracker")

    return {
      "message": "Tracker reset successfully",
      "stream_id": stream_id
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

