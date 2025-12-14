# app/api/onvif.py
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/onvif", tags=["onvif"])
logger = logging.getLogger(__name__)


class ONVIFDiscoverRequest(BaseModel):
    ip: str
    port: int = 80
    username: str
    password: str


@router.post("/discover")
async def discover_camera(request: ONVIFDiscoverRequest):
    """Discover camera capabilities via ONVIF"""
    try:
        from onvif import ONVIFCamera

        # Connect to camera
        cam = ONVIFCamera(request.ip, request.port, request.username, request.password)

        # Get device info
        device_info = cam.devicemgmt.GetDeviceInformation()

        # Get media profiles and stream URIs
        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()

        streams = []
        for profile in profiles:
            try:
                stream_uri = media_service.GetStreamUri(
                    {
                        "StreamSetup": {
                            "Stream": "RTP-Unicast",
                            "Transport": {"Protocol": "RTSP"},
                        },
                        "ProfileToken": profile.token,
                    }
                )

                streams.append(
                    {
                        "name": profile.Name,
                        "token": profile.token,
                        "uri": stream_uri.Uri,
                        "width": profile.VideoEncoderConfiguration.Resolution.Width
                        or 0,
                        "height": profile.VideoEncoderConfiguration.Resolution.Height
                        or 0,
                        "fps": profile.VideoEncoderConfiguration.RateControl.FrameRateLimit
                        or 10
                        if hasattr(profile.VideoEncoderConfiguration, "RateControl")
                        else None,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not get stream for profile {profile.Name}: {e}")

        # Check PTZ capability
        has_ptz = False
        try:
            ptz_service = cam.create_ptz_service()
            has_ptz = True
        except:
            pass

        return {
            "success": True,
            "device": {
                "manufacturer": device_info.Manufacturer,
                "model": device_info.Model,
                "firmware": device_info.FirmwareVersion,
                "serial": device_info.SerialNumber,
                "hardware_id": device_info.HardwareId,
            },
            "capabilities": {
                "ptz": has_ptz,
                "audio": any("Audio" in s["name"] for s in streams),
            },
            "streams": streams,
        }

    except Exception as e:
        logger.error(f"ONVIF discovery failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ptz/move")
async def ptz_move(
    ip: str,
    username: str,
    password: str,
    x: float = 0.0,  # -1.0 to 1.0 (left/right)
    y: float = 0.0,  # -1.0 to 1.0 (down/up)
    z: float = 0.0,  # -1.0 to 1.0 (zoom out/in)
):
    """Control PTZ camera movement"""
    try:
        from onvif import ONVIFCamera

        cam = ONVIFCamera(ip, 80, username, password)
        ptz_service = cam.create_ptz_service()

        # Get profile token
        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token

        # Move camera
        request = ptz_service.create_type("ContinuousMove")
        request.ProfileToken = token
        request.Velocity = {"PanTilt": {"x": x, "y": y}, "Zoom": {"x": z}}

        ptz_service.ContinuousMove(request)

        return {"success": True, "message": f"PTZ moved: x={x}, y={y}, z={z}"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ptz/stop")
async def ptz_stop(ip: str, username: str, password: str):
    """Stop PTZ movement"""
    try:
        from onvif import ONVIFCamera

        cam = ONVIFCamera(ip, 80, username, password)
        ptz_service = cam.create_ptz_service()

        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token

        request = ptz_service.create_type("Stop")
        request.ProfileToken = token
        request.PanTilt = True
        request.Zoom = True

        ptz_service.Stop(request)

        return {"success": True, "message": "PTZ stopped"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
