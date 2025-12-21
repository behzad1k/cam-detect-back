# app/schemas/ptz.py
"""
PTZ (Pan-Tilt-Zoom) Control Schemas
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PTZDiscoveryRequest(BaseModel):
    """Request to discover ONVIF camera capabilities"""

    ip: str = Field(..., description="Camera IP address", example="192.168.1.12")
    port: int = Field(80, description="ONVIF port (default: 80)", example=80)
    username: str = Field(..., description="ONVIF username", example="admin")
    password: str = Field(..., description="ONVIF password", example="Behzad8690")

    class Config:
        json_schema_extra = {
            "example": {
                "ip": "192.168.1.12",
                "port": 80,
                "username": "admin",
                "password": "Behzad8690",
            }
        }


class DeviceInfo(BaseModel):
    """ONVIF device information"""

    manufacturer: str = Field(..., description="Camera manufacturer", example="Dahua")
    model: str = Field(
        ..., description="Camera model", example="IPC-HDW1239T1-A-LED-S5"
    )
    firmware: str = Field(
        ..., description="Firmware version", example="2.840.0000000.48.R"
    )
    serial: str = Field(..., description="Serial number", example="4N22BEA00000001")
    hardware_id: Optional[str] = Field(
        None, description="Hardware ID", example="IPC-HX1X3X"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "manufacturer": "Dahua",
                "model": "IPC-HDW1239T1-A-LED-S5",
                "firmware": "2.840.0000000.48.R",
                "serial": "4N22BEA00000001",
                "hardware_id": "IPC-HX1X3X",
            }
        }


class DeviceCapabilities(BaseModel):
    """ONVIF device capabilities"""

    ptz: bool = Field(..., description="PTZ control supported", example=True)
    audio: bool = Field(..., description="Audio supported", example=True)
    events: bool = Field(..., description="Events supported", example=True)
    imaging: bool = Field(..., description="Imaging control supported", example=True)
    analytics: bool = Field(False, description="Analytics supported", example=False)
    device_io: bool = Field(False, description="Device I/O supported", example=False)
    recording: bool = Field(False, description="Recording supported", example=False)
    profiles: int = Field(..., description="Number of media profiles", example=2)

    class Config:
        json_schema_extra = {
            "example": {
                "ptz": True,
                "audio": True,
                "events": True,
                "imaging": True,
                "analytics": False,
                "device_io": False,
                "recording": False,
                "profiles": 2,
            }
        }


class StreamInfo(BaseModel):
    """Stream profile information"""

    name: str = Field(..., description="Profile name", example="MainStream")
    token: str = Field(..., description="Profile token", example="Profile_1")
    uri: str = Field(
        ...,
        description="RTSP stream URI",
        example="rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
    )
    width: int = Field(..., description="Video width", example=1920)
    height: int = Field(..., description="Video height", example=1080)
    fps: int = Field(..., description="Frames per second", example=25)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "MainStream",
                "token": "Profile_1",
                "uri": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
                "width": 1920,
                "height": 1080,
                "fps": 25,
            }
        }


class ConnectionInfo(BaseModel):
    """Connection information"""

    ip: str = Field(..., description="Camera IP address", example="192.168.1.12")
    port: int = Field(..., description="ONVIF port", example=80)
    onvif_enabled: bool = Field(..., description="ONVIF protocol enabled", example=True)
    auth_method: str = Field(
        ..., description="Authentication method used", example="digest"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ip": "192.168.1.12",
                "port": 80,
                "onvif_enabled": True,
                "auth_method": "digest",
            }
        }


class PTZDiscoveryResponse(BaseModel):
    """Response from ONVIF discovery"""

    success: bool = Field(..., description="Discovery successful", example=True)
    device: DeviceInfo = Field(..., description="Device information")
    capabilities: DeviceCapabilities = Field(..., description="Device capabilities")
    streams: List[StreamInfo] = Field(..., description="Available stream profiles")
    connection: ConnectionInfo = Field(..., description="Connection details")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "device": {
                    "manufacturer": "Dahua",
                    "model": "IPC-HDW1239T1-A-LED-S5",
                    "firmware": "2.840.0000000.48.R",
                    "serial": "4N22BEA00000001",
                    "hardware_id": "IPC-HX1X3X",
                },
                "capabilities": {
                    "ptz": True,
                    "audio": True,
                    "events": True,
                    "imaging": True,
                    "analytics": False,
                    "device_io": False,
                    "recording": False,
                    "profiles": 2,
                },
                "streams": [
                    {
                        "name": "MainStream",
                        "token": "Profile_1",
                        "uri": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0",
                        "width": 1920,
                        "height": 1080,
                        "fps": 25,
                    },
                    {
                        "name": "SubStream",
                        "token": "Profile_2",
                        "uri": "rtsp://admin:password@192.168.1.12:554/cam/realmonitor?channel=1&subtype=1",
                        "width": 640,
                        "height": 480,
                        "fps": 25,
                    },
                ],
                "connection": {
                    "ip": "192.168.1.12",
                    "port": 80,
                    "onvif_enabled": True,
                    "auth_method": "digest",
                },
            }
        }


class PTZMoveRequest(BaseModel):
    """Request to move PTZ camera"""

    direction: Literal["up", "down", "left", "right", "zoom_in", "zoom_out", "stop"] = (
        Field(..., description="PTZ movement direction", example="right")
    )
    speed: Optional[float] = Field(
        0.5, description="Movement speed (0.0 to 1.0)", example=0.5, ge=0.0, le=1.0
    )

    class Config:
        json_schema_extra = {"example": {"direction": "right", "speed": 0.5}}


class PTZMoveResponse(BaseModel):
    """Response from PTZ move command"""

    success: bool = Field(
        ..., description="Command executed successfully", example=True
    )
    message: str = Field(..., description="Status message", example="PTZ moved right")
    direction: str = Field(..., description="Movement direction", example="right")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "PTZ moved right",
                "direction": "right",
            }
        }


class PTZPresetRequest(BaseModel):
    """Request to manage PTZ presets"""

    action: Literal["set", "goto", "remove"] = Field(
        ...,
        description="Preset action: set (create), goto (move to), remove (delete)",
        example="goto",
    )
    preset_name: str = Field(
        ..., description="Preset name/identifier", example="entrance_view"
    )

    class Config:
        json_schema_extra = {
            "example": {"action": "goto", "preset_name": "entrance_view"}
        }


class PTZPresetResponse(BaseModel):
    """Response from PTZ preset command"""

    success: bool = Field(
        ..., description="Command executed successfully", example=True
    )
    message: str = Field(
        ..., description="Status message", example="Moved to preset: entrance_view"
    )
    action: str = Field(..., description="Action performed", example="goto")
    preset_name: str = Field(..., description="Preset name", example="entrance_view")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Moved to preset: entrance_view",
                "action": "goto",
                "preset_name": "entrance_view",
            }
        }


class PTZStatusResponse(BaseModel):
    """Current PTZ status"""

    pan: float = Field(..., description="Pan position (-1.0 to 1.0)", example=0.25)
    tilt: float = Field(..., description="Tilt position (-1.0 to 1.0)", example=-0.15)
    zoom: float = Field(..., description="Zoom level (0.0 to 1.0)", example=0.5)
    moving: bool = Field(..., description="Camera currently moving", example=False)

    class Config:
        json_schema_extra = {
            "example": {"pan": 0.25, "tilt": -0.15, "zoom": 0.5, "moving": False}
        }


class PTZErrorResponse(BaseModel):
    """Error response from PTZ operations"""

    error: str = Field(..., description="Error type", example="Authentication Failed")
    message: str = Field(
        ...,
        description="Detailed error message",
        example="Invalid credentials or ONVIF not enabled",
    )
    camera_ip: Optional[str] = Field(
        None, description="Camera IP address", example="192.168.1.12"
    )
    hints: Optional[List[str]] = Field(
        None,
        description="Troubleshooting hints",
        example=[
            "Verify camera credentials are correct",
            "Check if ONVIF is enabled in camera settings",
            "Ensure user has ONVIF permissions",
        ],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Authentication Failed",
                "message": "All authentication methods failed. Invalid credentials or ONVIF not enabled.",
                "camera_ip": "192.168.1.12",
                "hints": [
                    "Verify camera credentials are correct",
                    "Check if ONVIF is enabled in camera settings",
                    "For Dahua: Setup > Network > ONVIF must be enabled",
                    "For Hikvision: Configuration > Network > Advanced Settings > Integration Protocol",
                    "Ensure user has ONVIF permissions",
                    "Try resetting camera ONVIF password",
                ],
            }
        }
