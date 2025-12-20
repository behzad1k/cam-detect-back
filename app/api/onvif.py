# app/api/onvif.py - FIXED VERSION WITH PROPER DAHUA AUTHENTICATION
import logging
import os
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/onvif", tags=["onvif"])
logger = logging.getLogger(__name__)


def get_wsdl_path():
    """Find WSDL path for onvif-zeep - critical for Dahua cameras"""
    try:
        import onvif

        onvif_dir = os.path.dirname(onvif.__file__)

        # Try standard location
        wsdl_path = os.path.join(onvif_dir, "wsdl")
        if os.path.exists(wsdl_path):
            devicemgmt = os.path.join(wsdl_path, "devicemgmt.wsdl")
            if os.path.exists(devicemgmt):
                logger.info(f"✅ Found WSDL at: {wsdl_path}")
                return wsdl_path

        # Search recursively in site-packages
        from pathlib import Path

        site_packages = Path(sys.prefix) / "lib"
        if site_packages.exists():
            for wsdl_dir in site_packages.rglob("wsdl"):
                devicemgmt = wsdl_dir / "devicemgmt.wsdl"
                if devicemgmt.exists():
                    logger.info(f"✅ Found WSDL at: {wsdl_dir}")
                    return str(wsdl_dir)
    except Exception as e:
        logger.warning(f"WSDL detection error: {e}")

    logger.error("❌ WSDL path not found - ONVIF may not work properly")
    return None


# Initialize WSDL path on module load
WSDL_PATH = get_wsdl_path()
if WSDL_PATH:
    logger.info(f"✅ ONVIF WSDL Path: {WSDL_PATH}")
else:
    logger.warning("⚠️ WSDL path not found - ONVIF authentication may fail")


class ONVIFDiscoverRequest(BaseModel):
    ip: str
    port: int = 80
    username: str
    password: str


@router.post("/discover")
async def discover_camera(request: ONVIFDiscoverRequest):
    """
    Discover camera capabilities via ONVIF

    Fixed for Dahua cameras with proper WS-Security authentication
    """
    try:
        from onvif import ONVIFCamera

        logger.info(f"🔍 ONVIF Discovery Request:")
        logger.info(f"   IP: {request.ip}:{request.port}")
        logger.info(f"   Username: {request.username}")
        logger.info(f"   Password length: {len(request.password)} chars")

        # CRITICAL FIX: Pass WSDL path for proper Dahua authentication
        if not WSDL_PATH:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "ONVIF WSDL files not found",
                    "message": "Please ensure onvif-zeep is properly installed",
                    "fix": "Run: pip install onvif-zeep",
                },
            )

        logger.info(f"📁 Using WSDL: {WSDL_PATH}")

        # Create ONVIF camera with WSDL path
        # This is CRITICAL for Dahua cameras
        cam = ONVIFCamera(
            request.ip,
            request.port,
            request.username,
            request.password,
            WSDL_PATH,  # Must include this for proper WS-Security
        )

        logger.info("⏳ Connecting to camera...")

        # Get device info with detailed error handling
        try:
            device_info = cam.devicemgmt.GetDeviceInformation()
            logger.info("✅ Successfully retrieved device information")

        except Exception as auth_error:
            error_str = str(auth_error)
            logger.error(f"❌ Device info failed: {error_str}")

            # Provide specific error messages
            if "not Authorized" in error_str or "Invalid username" in error_str:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "Authentication Failed",
                        "message": "Invalid username or password",
                        "camera_ip": request.ip,
                        "username_tried": request.username,
                        "hints": [
                            "Verify camera credentials are correct",
                            "Check if ONVIF is enabled in camera settings",
                            "For Dahua: Setup > Network > ONVIF must be enabled",
                            "Ensure user has ONVIF permissions",
                            "Try resetting camera ONVIF password",
                        ],
                    },
                )
            elif "Connection" in error_str or "timeout" in error_str:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Connection Failed",
                        "message": "Cannot connect to camera",
                        "camera_ip": request.ip,
                        "port": request.port,
                        "hints": [
                            "Verify camera IP address is correct",
                            "Check camera is powered on and connected to network",
                            "Verify ONVIF port (usually 80 or 8080)",
                            "Check firewall settings",
                        ],
                    },
                )
            else:
                raise

        # Get media profiles and stream URIs
        logger.info("⏳ Getting media profiles...")
        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()

        logger.info(f"✅ Found {len(profiles)} media profile(s)")

        streams = []
        for profile in profiles:
            try:
                logger.info(f"⏳ Processing profile: {profile.Name}")

                stream_uri = media_service.GetStreamUri(
                    {
                        "StreamSetup": {
                            "Stream": "RTP-Unicast",
                            "Transport": {"Protocol": "RTSP"},
                        },
                        "ProfileToken": profile.token,
                    }
                )

                # Extract resolution
                width = 0
                height = 0
                fps = None

                if hasattr(profile, "VideoEncoderConfiguration"):
                    vec = profile.VideoEncoderConfiguration
                    if hasattr(vec, "Resolution"):
                        width = vec.Resolution.Width or 0
                        height = vec.Resolution.Height or 0
                    if hasattr(vec, "RateControl") and hasattr(
                        vec.RateControl, "FrameRateLimit"
                    ):
                        fps = vec.RateControl.FrameRateLimit

                stream_info = {
                    "name": profile.Name,
                    "token": profile.token,
                    "uri": stream_uri.Uri,
                    "width": width,
                    "height": height,
                    "fps": fps or 25,
                }

                streams.append(stream_info)
                logger.info(
                    f"   ✅ {profile.Name}: {width}x{height} @ {fps or '?'} fps"
                )

            except Exception as e:
                logger.warning(
                    f"   ⚠️ Could not get stream for profile {profile.Name}: {e}"
                )

        # Check PTZ capability
        has_ptz = False
        try:
            logger.info("⏳ Checking PTZ capabilities...")
            ptz_service = cam.create_ptz_service()
            has_ptz = True
            logger.info("✅ PTZ is supported")
        except:
            logger.info("ℹ️ PTZ not supported")

        # Success response
        result = {
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
                "audio": any("Audio" in s.get("name", "") for s in streams),
                "profiles": len(streams),
            },
            "streams": streams,
            "connection": {
                "ip": request.ip,
                "port": request.port,
                "onvif_enabled": True,
            },
        }

        logger.info("✅ ONVIF discovery completed successfully")
        logger.info(f"   Manufacturer: {device_info.Manufacturer}")
        logger.info(f"   Model: {device_info.Model}")
        logger.info(f"   Streams: {len(streams)}")
        logger.info(f"   PTZ: {'Yes' if has_ptz else 'No'}")

        return result

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ ONVIF discovery failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "ONVIF Discovery Failed",
                "message": str(e),
                "type": type(e).__name__,
                "hints": [
                    "Verify camera supports ONVIF",
                    "Check camera network settings",
                    "Ensure onvif-zeep is installed: pip install onvif-zeep",
                ],
            },
        )


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

        if not WSDL_PATH:
            raise HTTPException(status_code=500, detail="WSDL path not configured")

        cam = ONVIFCamera(ip, 80, username, password, WSDL_PATH)
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

        if not WSDL_PATH:
            raise HTTPException(status_code=500, detail="WSDL path not configured")

        cam = ONVIFCamera(ip, 80, username, password, WSDL_PATH)
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
