# app/database/models.py - UPDATED WITH SEPARATE CONNECTION FIELDS

import uuid

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.database.base import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(64), nullable=False)
    location = Column(String(64), nullable=True)

    # Legacy field - keep for backward compatibility
    rtsp_url = Column(String(255), nullable=True)

    # NEW: Separate connection fields
    ip_address = Column(String(64), nullable=True)  # 192.168.1.12
    username = Column(String(64), nullable=True)  # admin
    password = Column(String(128), nullable=True)  # Behzad8690

    # Ports
    rtsp_port = Column(Integer, default=554, nullable=True)
    onvif_port = Column(Integer, default=80, nullable=True)
    http_port = Column(Integer, default=80, nullable=True)

    # Stream details (parsed from RTSP URL)
    channel = Column(String(16), default="1", nullable=True)  # 1
    subtype = Column(String(16), default="0", nullable=True)  # 0=main, 1=sub
    stream_path = Column(String(255), nullable=True)  # cam/realmonitor (for Dahua)
    manufacturer = Column(
        String(64), default="dahua", nullable=True
    )  # dahua, hikvision, etc.

    # Resolution
    width = Column(Integer, default=640)
    height = Column(Integer, default=480)
    fps = Column(Integer, default=30)

    # Calibration
    is_calibrated = Column(Boolean, default=False)
    pixels_per_meter = Column(Float, nullable=True)
    calibration_mode = Column(String(64), nullable=True)
    calibration_points = Column(JSON, nullable=True)

    # Features enabled
    features = Column(JSON, default=dict)

    # Models to use
    active_models = Column(JSON, default=list)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<Camera {self.name} ({self.id})>"

    @property
    def onvif_url(self):
        """Generate ONVIF URL from components"""
        if not self.ip_address:
            return None
        return f"onvif://{self.username}:{self.password}@{self.ip_address}:{self.onvif_port}/onvif/device_service"

    @property
    def rtsp_stream_url(self):
        """Generate RTSP URL from components"""
        if not self.ip_address:
            return self.rtsp_url  # Fallback to legacy

        # Build manufacturer-specific RTSP URL
        if self.manufacturer in ["dahua", "amcrest"]:
            # rtsp://admin:Behzad8690@192.168.1.12:554/cam/realmonitor?channel=1&subtype=1
            return f"rtsp://{self.username}:{self.password}@{self.ip_address}:{self.rtsp_port}/cam/realmonitor?channel={self.channel}&subtype={self.subtype}"
        elif self.manufacturer == "hikvision":
            # rtsp://admin:password@192.168.1.12:554/Streaming/Channels/101
            stream_id = "101" if self.subtype == "0" else "102"
            return f"rtsp://{self.username}:{self.password}@{self.ip_address}:{self.rtsp_port}/Streaming/Channels/{stream_id}"
        elif self.stream_path:
            # Custom path
            return f"rtsp://{self.username}:{self.password}@{self.ip_address}:{self.rtsp_port}/{self.stream_path}"
        else:
            # Fallback to legacy
            return self.rtsp_url or ""
