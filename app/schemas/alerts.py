# app/schemas/alerts.py
"""
Alert Configuration and Response Schemas
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SpeedAlertRule(BaseModel):
    """Speed alert configuration"""

    enabled: bool = Field(
        True, description="Enable/disable this alert rule", example=True
    )
    object_class: str = Field(
        ...,
        description="Object class to monitor (e.g., person, car, motorcycle)",
        example="car",
    )
    threshold_kmh: float = Field(
        ..., description="Speed threshold in km/h", example=50.0, gt=0
    )
    condition: Literal["over", "under"] = Field(
        "over", description="Alert when speed goes over/under threshold", example="over"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "object_class": "car",
                "threshold_kmh": 50.0,
                "condition": "over",
            }
        }


class TrackingAlertRule(BaseModel):
    """Tracking/dwell time alert configuration"""

    enabled: bool = Field(
        True, description="Enable/disable this alert rule", example=True
    )
    object_class: str = Field(
        ..., description="Object class to monitor (e.g., person, car)", example="person"
    )
    threshold_seconds: float = Field(
        0.0, description="Time threshold in seconds", example=30.0, ge=0
    )
    condition: Literal["over", "under"] = Field(
        "over", description="Alert when time goes over/under threshold", example="over"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "object_class": "person",
                "threshold_seconds": 30.0,
                "condition": "over",
            }
        }


class DistanceAlertRule(BaseModel):
    """Distance alert configuration"""

    enabled: bool = Field(
        True, description="Enable/disable this alert rule", example=True
    )
    object_class: str = Field(
        ..., description="Object class to monitor (e.g., person, car)", example="person"
    )
    threshold_meters: float = Field(
        ..., description="Distance threshold in meters", example=5.0, gt=0
    )
    condition: Literal["over", "under"] = Field(
        "under",
        description="Alert when distance goes over/under threshold",
        example="under",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "object_class": "person",
                "threshold_meters": 2.0,
                "condition": "under",
            }
        }


class AlertConfiguration(BaseModel):
    """Complete alert configuration for a camera"""

    speed_alerts: List[SpeedAlertRule] = Field(
        default_factory=list,
        description="List of speed alert rules (requires camera calibration and speed feature enabled)",
    )
    tracking_alerts: List[TrackingAlertRule] = Field(
        default_factory=list,
        description="List of tracking/dwell time alert rules (requires tracking feature enabled)",
    )
    distance_alerts: List[DistanceAlertRule] = Field(
        default_factory=list,
        description="List of distance alert rules (requires camera calibration and distance feature enabled)",
    )
    email_enabled: bool = Field(
        True, description="Enable/disable email notifications for alerts"
    )
    cooldown_seconds: int = Field(
        60,
        description="Cooldown period to prevent alert spam (same alert won't trigger within this period)",
        example=60,
        ge=0,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "speed_alerts": [
                    {
                        "enabled": True,
                        "object_class": "car",
                        "threshold_kmh": 50.0,
                        "condition": "over",
                    },
                    {
                        "enabled": True,
                        "object_class": "motorcycle",
                        "threshold_kmh": 40.0,
                        "condition": "over",
                    },
                ],
                "tracking_alerts": [
                    {
                        "enabled": True,
                        "object_class": "person",
                        "threshold_seconds": 30.0,
                        "condition": "over",
                    }
                ],
                "distance_alerts": [
                    {
                        "enabled": True,
                        "object_class": "person",
                        "threshold_meters": 2.0,
                        "condition": "under",
                    }
                ],
                "email_enabled": True,
                "cooldown_seconds": 60,
            }
        }


class Alert(BaseModel):
    """Individual alert instance"""

    alert_id: str = Field(
        ..., description="Unique alert identifier", example="alert_a1b2c3d4"
    )
    camera_id: str = Field(
        ..., description="Camera ID that triggered the alert", example="cam_001"
    )
    camera_name: str = Field(
        ..., description="Camera display name", example="Entrance Camera"
    )
    timestamp: datetime = Field(
        ..., description="Alert timestamp", example="2024-01-01T12:00:00Z"
    )
    alert_type: Literal["speed", "tracking", "distance"] = Field(
        ..., description="Type of alert", example="speed"
    )
    object_class: str = Field(
        ..., description="Object class that triggered alert", example="car"
    )
    track_id: str = Field(
        ..., description="Tracking ID of the object", example="track_42"
    )

    # Alert details
    threshold_value: float = Field(
        ..., description="Configured threshold value", example=50.0
    )
    actual_value: float = Field(
        ..., description="Actual measured value that triggered alert", example=65.5
    )
    condition: str = Field(
        ..., description="Alert condition (over/under)", example="over"
    )
    unit: str = Field(
        ..., description="Measurement unit (km/h, seconds, meters)", example="km/h"
    )

    # Location info
    bbox: Optional[List[float]] = Field(
        None, description="Bounding box [x1, y1, x2, y2]", example=[100, 200, 300, 400]
    )
    centroid: Optional[List[float]] = Field(
        None, description="Object centroid [x, y]", example=[200, 300]
    )

    # Snapshot (optional)
    snapshot_base64: Optional[str] = Field(
        None, description="Base64-encoded snapshot image"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "alert_a1b2c3d4",
                "camera_id": "cam_001",
                "camera_name": "Entrance Camera",
                "timestamp": "2024-01-01T12:00:00Z",
                "alert_type": "speed",
                "object_class": "car",
                "track_id": "track_42",
                "threshold_value": 50.0,
                "actual_value": 65.5,
                "condition": "over",
                "unit": "km/h",
                "bbox": [100, 200, 300, 400],
                "centroid": [200, 300],
            }
        }


class AlertResponse(BaseModel):
    """WebSocket alert response (delivered via camera WebSocket)"""

    alerts: List[Alert] = Field(..., description="List of alerts in this frame")

    class Config:
        json_schema_extra = {
            "example": {
                "alerts": [
                    {
                        "alert_id": "alert_a1b2c3d4",
                        "camera_id": "cam_001",
                        "camera_name": "Entrance Camera",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "alert_type": "speed",
                        "object_class": "car",
                        "track_id": "track_42",
                        "threshold_value": 50.0,
                        "actual_value": 65.5,
                        "condition": "over",
                        "unit": "km/h",
                        "bbox": [100, 200, 300, 400],
                        "centroid": [200, 300],
                    }
                ]
            }
        }


class AlertStats(BaseModel):
    """Alert statistics for a camera"""

    camera_id: str = Field(..., description="Camera ID", example="cam_001")
    total_alerts: int = Field(..., description="Total number of alerts", example=127)
    alerts_by_type: Dict[str, int] = Field(
        ...,
        description="Alert counts by type",
        example={"speed": 45, "tracking": 62, "distance": 20},
    )
    alerts_by_class: Dict[str, int] = Field(
        ...,
        description="Alert counts by object class",
        example={"car": 45, "person": 82},
    )
    last_alert_time: Optional[datetime] = Field(
        None,
        description="Timestamp of most recent alert",
        example="2024-01-01T12:30:00Z",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "camera_id": "cam_001",
                "total_alerts": 127,
                "alerts_by_type": {"speed": 45, "tracking": 62, "distance": 20},
                "alerts_by_class": {"car": 45, "person": 82},
                "last_alert_time": "2024-01-01T12:30:00Z",
            }
        }


class AlertHistoryResponse(BaseModel):
    """Response for alert history endpoint"""

    camera_id: str = Field(..., description="Camera ID", example="cam_001")
    total: int = Field(..., description="Total alerts in history", example=127)
    limit: int = Field(..., description="Number of alerts returned", example=10)
    alerts: List[Alert] = Field(..., description="List of recent alerts")

    class Config:
        json_schema_extra = {
            "example": {
                "camera_id": "cam_001",
                "total": 127,
                "limit": 10,
                "alerts": [
                    {
                        "alert_id": "alert_a1b2c3d4",
                        "camera_id": "cam_001",
                        "camera_name": "Entrance Camera",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "alert_type": "speed",
                        "object_class": "car",
                        "track_id": "track_42",
                        "threshold_value": 50.0,
                        "actual_value": 65.5,
                        "condition": "over",
                        "unit": "km/h",
                    }
                ],
            }
        }


class TestEmailRequest(BaseModel):
    """Request to send a test email"""

    email: Optional[str] = Field(
        None,
        description="Email address to send test to (uses camera's alert_email if not provided)",
        example="test@company.com",
    )

    class Config:
        json_schema_extra = {"example": {"email": "test@company.com"}}


class TestEmailResponse(BaseModel):
    """Response from test email endpoint"""

    success: bool = Field(
        ..., description="Whether email was sent successfully", example=True
    )
    message: str = Field(
        ..., description="Status message", example="Test email sent successfully"
    )
    recipient: str = Field(
        ...,
        description="Email address that received the test",
        example="test@company.com",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Test email sent successfully",
                "recipient": "test@company.com",
            }
        }
