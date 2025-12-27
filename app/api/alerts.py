# app/api/alerts.py
"""
Alert API Endpoints - WebSocket Version
- Alert configuration
- Alert history
- Alert statistics
(Alerts sent via existing WebSocket connections)
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.base import get_db
from app.schemas.alerts import Alert, AlertConfiguration, AlertStats
from app.services.alert_manager import alert_manager
from app.services.camera_service import camera_service

router = APIRouter(prefix="/alerts", tags=["alerts"])
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ENDPOINTS ====================


@router.get("/{camera_id}/config", response_model=AlertConfiguration)
async def get_alert_config(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Get alert configuration for a camera"""
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    # Return alert config or default
    if camera.alert_config:
        return AlertConfiguration(**camera.alert_config)
    else:
        return AlertConfiguration()


@router.put("/{camera_id}/config", response_model=AlertConfiguration)
async def update_alert_config(
    camera_id: str, config: AlertConfiguration, db: AsyncSession = Depends(get_db)
):
    """
    Update alert configuration for a camera

    Example:
    ```json
    {
      "speed_alerts": [
        {
          "enabled": true,
          "object_class": "car",
          "threshold_kmh": 50.0,
          "condition": "over"
        }
      ],
      "tracking_alerts": [
        {
          "enabled": true,
          "object_class": "person",
          "threshold_seconds": 30.0,
          "condition": "over"
        }
      ],
      "distance_alerts": [
        {
          "enabled": true,
          "object_class": "person",
          "threshold_meters": 2.0,
          "condition": "under"
        }
      ],
      "email_enabled": true,
      "cooldown_seconds": 60
    }
    ```

    Note: Alerts will be sent via the existing WebSocket connection for this camera.
    """
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    # Update alert config
    camera.alert_config = config.dict()

    try:
        await db.commit()
        await db.refresh(camera)

        logger.info(f"✅ Alert configuration updated for camera {camera_id}")
        logger.info(f"   Speed alerts: {len(config.speed_alerts)}")
        logger.info(f"   Tracking alerts: {len(config.tracking_alerts)}")
        logger.info(f"   Distance alerts: {len(config.distance_alerts)}")
        logger.info(f"   Alerts will be sent via WebSocket for camera {camera_id}")

        return config

    except Exception as e:
        await db.rollback()
        logger.error(f"❌ Failed to update alert config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{camera_id}/email")
async def update_alert_email(
    camera_id: str, email: str, db: AsyncSession = Depends(get_db)
):
    """Update alert email address for a camera"""
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    camera.alert_email = email

    try:
        await db.commit()
        await db.refresh(camera)

        logger.info(f"✅ Alert email updated for camera {camera_id}: {email}")

        return {"success": True, "camera_id": camera_id, "email": email}

    except Exception as e:
        await db.rollback()
        logger.error(f"❌ Failed to update alert email: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HISTORY & STATS ENDPOINTS ====================


@router.get("/{camera_id}/history", response_model=List[Alert])
async def get_alert_history(
    camera_id: str, limit: int = 10, db: AsyncSession = Depends(get_db)
):
    """Get recent alert history for a camera"""
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    alerts = alert_manager.get_recent_alerts(camera_id, limit)
    return alerts


@router.get("/{camera_id}/stats", response_model=AlertStats)
async def get_alert_stats(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Get alert statistics for a camera"""
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    stats = alert_manager.get_alert_stats(camera_id)
    return AlertStats(**stats)


# ==================== TEST ENDPOINTS ====================


@router.post("/{camera_id}/test-email")
async def test_email_alert(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Send test email alert"""
    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    if not camera.alert_email:
        raise HTTPException(
            status_code=400, detail="No alert email configured for this camera"
        )

    # Create test alert
    from datetime import datetime

    test_alert = Alert(
        alert_id="test_alert",
        camera_id=camera_id,
        camera_name=camera.name,
        timestamp=datetime.now(),
        alert_type="speed",
        object_class="test_object",
        track_id="test_track",
        threshold_value=50.0,
        actual_value=65.5,
        condition="over",
        unit="km/h",
    )

    try:
        alert_manager.send_email_alert(camera.alert_email, test_alert)
        return {"success": True, "message": f"Test email sent to {camera.alert_email}"}
    except Exception as e:
        logger.error(f"Test email failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to send test email: {str(e)}"
        )
