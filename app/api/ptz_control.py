# app/api/ptz_control.py - WITH BOTH CAMERA_ID AND DIRECT IP ENDPOINTS

import logging
import os
import sys
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.base import get_db
from app.services.camera_service import camera_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["PTZ Control"])


# ============================================================================
# WSDL PATH DETECTION
# ============================================================================


def get_wsdl_path():
    """Robust WSDL path detection"""
    try:
        import onvif

        onvif_dir = os.path.dirname(onvif.__file__)

        for subdir in ["wsdl", "wsdl_files", "wsdls"]:
            wsdl_path = os.path.join(onvif_dir, subdir)
            if os.path.exists(wsdl_path):
                devicemgmt = os.path.join(wsdl_path, "devicemgmt.wsdl")
                if os.path.exists(devicemgmt):
                    logger.info(f"✅ Found WSDL at: {wsdl_path}")
                    return wsdl_path

        # Search recursively
        from pathlib import Path

        site_packages = Path(sys.prefix) / "lib"
        if site_packages.exists():
            for wsdl_dir in site_packages.rglob("wsdl"):
                devicemgmt = wsdl_dir / "devicemgmt.wsdl"
                if devicemgmt.exists():
                    return str(wsdl_dir)

    except Exception as e:
        logger.warning(f"WSDL detection error: {e}")

    return None


WSDL_PATH = get_wsdl_path()
if WSDL_PATH:
    logger.info(f"✅ WSDL Path: {WSDL_PATH}")
else:
    logger.error("❌ WSDL files not found")


# ============================================================================
# REQUEST MODELS (for camera_id endpoints)
# ============================================================================


class PTZMoveRequest(BaseModel):
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    timeout: int = 1


class PTZPresetRequest(BaseModel):
    preset_token: str


# ============================================================================
# DIRECT IP ENDPOINTS (Query Parameters)
# ============================================================================


@router.post("/ptz/move")
async def ptz_move_direct(
    ip: str = Query(..., description="Camera IP address"),
    username: str = Query(..., description="ONVIF username"),
    password: str = Query(..., description="ONVIF password"),
    x: float = Query(0.0, description="Pan: -1.0 (left) to 1.0 (right)"),
    y: float = Query(0.0, description="Tilt: -1.0 (down) to 1.0 (up)"),
    z: float = Query(0.0, description="Zoom: -1.0 (out) to 1.0 (in)"),
    port: int = Query(80, description="ONVIF port"),
):
    """
    Control PTZ camera movement using direct IP/credentials

    Example:
    POST /api/v1/ptz/move?ip=192.168.1.12&username=admin&password=pass&x=0.5&y=0&z=0
    """
    try:
        from onvif import ONVIFCamera

        logger.info(f"🔌 Direct PTZ move: {username}@{ip}:{port} (x={x}, y={y}, z={z})")

        # Create ONVIF camera
        cam = ONVIFCamera(ip, port, username, password, WSDL_PATH)

        # Get PTZ service
        ptz_service = cam.create_ptz_service()

        # Get profile token
        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()

        if not profiles:
            raise HTTPException(status_code=404, detail="No media profiles found")

        token = profiles[0].token

        # Create move request
        request = ptz_service.create_type("ContinuousMove")
        request.ProfileToken = token

        # Create velocity structure
        velocity = ptz_service.create_type("PTZSpeed")
        velocity.PanTilt = ptz_service.create_type("Vector2D")
        velocity.PanTilt.x = x
        velocity.PanTilt.y = y

        velocity.Zoom = ptz_service.create_type("Vector1D")
        velocity.Zoom.x = z

        request.Velocity = velocity

        # Execute move
        ptz_service.ContinuousMove(request)

        logger.info(f"✅ PTZ moved successfully")

        return {
            "success": True,
            "message": f"PTZ moved: x={x}, y={y}, z={z}",
            "ip": ip,
            "pan": x,
            "tilt": y,
            "zoom": z,
        }

    except Exception as e:
        logger.error(f"❌ PTZ move failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ptz/stop")
async def ptz_stop_direct(
    ip: str = Query(..., description="Camera IP address"),
    username: str = Query(..., description="ONVIF username"),
    password: str = Query(..., description="ONVIF password"),
    port: int = Query(80, description="ONVIF port"),
):
    """
    Stop PTZ movement using direct IP/credentials

    Example:
    POST /api/v1/ptz/stop?ip=192.168.1.12&username=admin&password=pass
    """
    try:
        from onvif import ONVIFCamera

        logger.info(f"🛑 Direct PTZ stop: {username}@{ip}:{port}")

        cam = ONVIFCamera(ip, port, username, password, WSDL_PATH)
        ptz_service = cam.create_ptz_service()

        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token

        # Stop request
        request = ptz_service.create_type("Stop")
        request.ProfileToken = token
        request.PanTilt = True
        request.Zoom = True

        ptz_service.Stop(request)

        logger.info(f"✅ PTZ stopped")

        return {"success": True, "message": "PTZ stopped", "ip": ip}

    except Exception as e:
        logger.error(f"❌ PTZ stop failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ptz/position")
async def ptz_get_position_direct(
    ip: str = Query(..., description="Camera IP address"),
    username: str = Query(..., description="ONVIF username"),
    password: str = Query(..., description="ONVIF password"),
    port: int = Query(80, description="ONVIF port"),
):
    """
    Get current PTZ position using direct IP/credentials

    Example:
    GET /api/v1/ptz/position?ip=192.168.1.12&username=admin&password=pass
    """
    try:
        from onvif import ONVIFCamera

        cam = ONVIFCamera(ip, port, username, password, WSDL_PATH)
        ptz_service = cam.create_ptz_service()

        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token

        # Get status
        status = ptz_service.GetStatus({"ProfileToken": token})

        position = {"pan": 0.0, "tilt": 0.0, "zoom": 0.0}

        if status and status.Position:
            if status.Position.PanTilt:
                position["pan"] = status.Position.PanTilt.x
                position["tilt"] = status.Position.PanTilt.y
            if status.Position.Zoom:
                position["zoom"] = status.Position.Zoom.x

        return {"success": True, "ip": ip, "position": position}

    except Exception as e:
        logger.error(f"❌ Get position failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ptz/presets")
async def ptz_list_presets_direct(
    ip: str = Query(..., description="Camera IP address"),
    username: str = Query(..., description="ONVIF username"),
    password: str = Query(..., description="ONVIF password"),
    port: int = Query(80, description="ONVIF port"),
):
    """
    List PTZ presets using direct IP/credentials

    Example:
    GET /api/v1/ptz/presets?ip=192.168.1.12&username=admin&password=pass
    """
    try:
        from onvif import ONVIFCamera

        cam = ONVIFCamera(ip, port, username, password, WSDL_PATH)
        ptz_service = cam.create_ptz_service()

        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token

        presets = ptz_service.GetPresets({"ProfileToken": token})

        preset_list = []
        for preset in presets:
            preset_list.append(
                {
                    "token": preset.token,
                    "name": preset.Name if hasattr(preset, "Name") else preset.token,
                }
            )

        return {"success": True, "ip": ip, "presets": preset_list}

    except Exception as e:
        logger.error(f"❌ List presets failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ptz/presets/goto")
async def ptz_goto_preset_direct(
    ip: str = Query(..., description="Camera IP address"),
    username: str = Query(..., description="ONVIF username"),
    password: str = Query(..., description="ONVIF password"),
    preset_token: str = Query(..., description="Preset token to go to"),
    port: int = Query(80, description="ONVIF port"),
):
    """
    Go to preset using direct IP/credentials

    Example:
    POST /api/v1/ptz/presets/goto?ip=192.168.1.12&username=admin&password=pass&preset_token=1
    """
    try:
        from onvif import ONVIFCamera

        cam = ONVIFCamera(ip, port, username, password, WSDL_PATH)
        ptz_service = cam.create_ptz_service()

        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token

        ptz_service.GotoPreset({"ProfileToken": token, "PresetToken": preset_token})

        return {
            "success": True,
            "ip": ip,
            "preset_token": preset_token,
            "message": f"Moved to preset {preset_token}",
        }

    except Exception as e:
        logger.error(f"❌ Goto preset failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# CAMERA_ID ENDPOINTS (Using Database)
# ============================================================================


async def get_onvif_camera(camera_id: str, db: AsyncSession):
    """Build ONVIF camera from database fields"""
    from onvif import ONVIFCamera

    if WSDL_PATH is None:
        raise HTTPException(status_code=500, detail="ONVIF WSDL files not found")

    camera = await camera_service.get_camera(db, camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    ip_address = camera.ip_address
    username = camera.username or "admin"
    password = camera.password or ""
    onvif_port = camera.onvif_port or 80

    if not ip_address:
        raise HTTPException(status_code=400, detail="Camera IP not configured")

    try:
        onvif_cam = ONVIFCamera(ip_address, onvif_port, username, password, WSDL_PATH)
        return onvif_cam, camera
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")


async def get_ptz_service(camera_id: str, db: AsyncSession):
    """Get PTZ service"""
    onvif_cam, camera = await get_onvif_camera(camera_id, db)

    try:
        ptz_service = onvif_cam.create_ptz_service()
        return ptz_service, onvif_cam, camera
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PTZ service failed: {str(e)}")


@router.post("/ptz/{camera_id}/move")
async def ptz_move_by_id(
    camera_id: str, request: PTZMoveRequest, db: AsyncSession = Depends(get_db)
):
    """
    Move PTZ camera using camera_id from database

    Example:
    POST /api/v1/ptz/{camera_id}/move
    Body: {"pan": 0.3, "tilt": 0, "zoom": 0, "timeout": 2}
    """
    ptz_service, onvif_cam, camera = await get_ptz_service(camera_id, db)

    try:
        media_service = onvif_cam.create_media_service()
        profiles = media_service.GetProfiles()

        if not profiles:
            raise HTTPException(status_code=404, detail="No media profiles found")

        profile_token = profiles[0].token

        # Create move request
        move_request = ptz_service.create_type("ContinuousMove")
        move_request.ProfileToken = profile_token

        # Create velocity
        velocity = ptz_service.create_type("PTZSpeed")
        velocity.PanTilt = ptz_service.create_type("Vector2D")
        velocity.PanTilt.x = request.pan
        velocity.PanTilt.y = request.tilt

        velocity.Zoom = ptz_service.create_type("Vector1D")
        velocity.Zoom.x = request.zoom

        move_request.Velocity = velocity

        # Set timeout
        if request.timeout:
            from datetime import timedelta

            move_request.Timeout = timedelta(seconds=request.timeout)

        ptz_service.ContinuousMove(move_request)

        logger.info(f"✅ PTZ moved (camera_id): pan={request.pan}, tilt={request.tilt}")

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "pan": request.pan,
            "tilt": request.tilt,
            "zoom": request.zoom,
        }

    except Exception as e:
        logger.error(f"❌ Move failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ptz/{camera_id}/stop")
async def ptz_stop_by_id(camera_id: str, db: AsyncSession = Depends(get_db)):
    """
    Stop PTZ movement using camera_id

    Example:
    POST /api/v1/ptz/{camera_id}/stop
    """
    ptz_service, onvif_cam, camera = await get_ptz_service(camera_id, db)

    try:
        media_service = onvif_cam.create_media_service()
        profiles = media_service.GetProfiles()
        profile_token = profiles[0].token

        stop_request = ptz_service.create_type("Stop")
        stop_request.ProfileToken = profile_token
        stop_request.PanTilt = True
        stop_request.Zoom = True

        ptz_service.Stop(stop_request)

        return {"success": True, "camera_id": camera_id, "camera_name": camera.name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ptz/{camera_id}/position")
async def ptz_get_position_by_id(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Get PTZ position using camera_id"""
    ptz_service, onvif_cam, camera = await get_ptz_service(camera_id, db)

    try:
        media_service = onvif_cam.create_media_service()
        profiles = media_service.GetProfiles()
        profile_token = profiles[0].token

        status = ptz_service.GetStatus({"ProfileToken": profile_token})

        position = {"pan": 0.0, "tilt": 0.0, "zoom": 0.0}

        if status and status.Position:
            if status.Position.PanTilt:
                position["pan"] = status.Position.PanTilt.x
                position["tilt"] = status.Position.PanTilt.y
            if status.Position.Zoom:
                position["zoom"] = status.Position.Zoom.x

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "position": position,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ptz/{camera_id}/presets")
async def ptz_list_presets_by_id(camera_id: str, db: AsyncSession = Depends(get_db)):
    """List PTZ presets using camera_id"""
    ptz_service, onvif_cam, camera = await get_ptz_service(camera_id, db)

    try:
        media_service = onvif_cam.create_media_service()
        profiles = media_service.GetProfiles()
        profile_token = profiles[0].token

        presets = ptz_service.GetPresets({"ProfileToken": profile_token})

        preset_list = []
        for preset in presets:
            preset_list.append(
                {
                    "token": preset.token,
                    "name": preset.Name if hasattr(preset, "Name") else preset.token,
                }
            )

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "presets": preset_list,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ptz/{camera_id}/presets/goto")
async def ptz_goto_preset_by_id(
    camera_id: str, request: PTZPresetRequest, db: AsyncSession = Depends(get_db)
):
    """Go to preset using camera_id"""
    ptz_service, onvif_cam, camera = await get_ptz_service(camera_id, db)

    try:
        media_service = onvif_cam.create_media_service()
        profiles = media_service.GetProfiles()
        profile_token = profiles[0].token

        ptz_service.GotoPreset(
            {"ProfileToken": profile_token, "PresetToken": request.preset_token}
        )

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "preset_token": request.preset_token,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ptz/{camera_id}/capabilities")
async def get_ptz_capabilities_by_id(
    camera_id: str, db: AsyncSession = Depends(get_db)
):
    """Get PTZ capabilities using camera_id"""
    try:
        onvif_cam, camera = await get_onvif_camera(camera_id, db)
        capabilities = onvif_cam.devicemgmt.GetCapabilities()

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "ptz_supported": hasattr(capabilities, "PTZ")
            and capabilities.PTZ is not None,
            "connection": {
                "ip": camera.ip_address,
                "onvif_port": camera.onvif_port,
                "manufacturer": camera.manufacturer,
            },
            "wsdl_path": WSDL_PATH,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
