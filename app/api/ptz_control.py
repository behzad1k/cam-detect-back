# app/api/ptz_control.py - UPDATED WITH NEW ONVIF HANDLER

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.base import get_db
from app.services.camera_service import camera_service
from app.services.onvif_handler import ONVIFConfig, ONVIFHandler, discover_camera

logger = logging.getLogger(__name__)

router = APIRouter(tags=["PTZ Control"])


# ============================================================================
# REQUEST MODELS
# ============================================================================


class ONVIFDiscoverRequest(BaseModel):
    ip: str
    port: int = 80
    username: str
    password: str


class PTZMoveRequest(BaseModel):
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    timeout: int = 1


class PTZPresetRequest(BaseModel):
    preset_token: str


class PTZSetPresetRequest(BaseModel):
    preset_name: str
    preset_token: Optional[str] = None


# ============================================================================
# DIRECT IP ENDPOINTS (Query Parameters)
# ============================================================================


@router.post("/ptz/discover")
async def discover_camera_endpoint(
    request: ONVIFDiscoverRequest, db: AsyncSession = Depends(get_db)
):
    """
    Discover camera capabilities via ONVIF

    Supports multiple authentication methods:
    - SOAP with WS-Security
    - HTTP Digest Auth
    - HTTP Basic Auth

    Automatically detects which method works.
    """
    try:
        logger.info(f"🔍 ONVIF Discovery Request:")
        logger.info(f"   IP: {request.ip}:{request.port}")
        logger.info(f"   Username: {request.username}")
        logger.info(f"   Password length: {len(request.password)} chars")

        # Use new handler
        capabilities = await discover_camera(
            ip=request.ip,
            port=request.port,
            username=request.username,
            password=request.password,
        )

        # Format response for frontend
        result = {
            "success": True,
            "device": {
                "manufacturer": capabilities.manufacturer,
                "model": capabilities.model,
                "firmware": capabilities.firmware,
                "serial": capabilities.serial,
                "hardware_id": capabilities.hardware_id,
            },
            "capabilities": {
                "ptz": capabilities.capabilities.get("ptz", False),
                "audio": capabilities.capabilities.get("audio", False),
                "events": capabilities.capabilities.get("events", False),
                "imaging": capabilities.capabilities.get("imaging", False),
                "analytics": capabilities.capabilities.get("analytics", False),
                "device_io": capabilities.capabilities.get("device_io", False),
                "recording": capabilities.capabilities.get("recording", False),
                "profiles": len(capabilities.profiles),
            },
            "streams": [
                {
                    "name": p["name"],
                    "token": p["token"],
                    "uri": p.get("uri", ""),
                    "width": p["width"],
                    "height": p["height"],
                    "fps": p["fps"],
                }
                for p in capabilities.profiles
            ],
            "connection": {
                "ip": request.ip,
                "port": request.port,
                "onvif_enabled": True,
                "auth_method": capabilities.auth_method.value
                if capabilities.auth_method
                else "unknown",
            },
        }

        logger.info("✅ ONVIF discovery completed successfully")
        logger.info(f"   Manufacturer: {capabilities.manufacturer}")
        logger.info(f"   Model: {capabilities.model}")
        logger.info(
            f"   Auth Method: {capabilities.auth_method.value if capabilities.auth_method else 'unknown'}"
        )
        logger.info(f"   Streams: {len(capabilities.profiles)}")
        logger.info(
            f"   PTZ: {'Yes' if capabilities.capabilities.get('ptz') else 'No'}"
        )
        logger.info(
            f"   Events: {'Yes' if capabilities.capabilities.get('events') else 'No'}"
        )

        return result

    except Exception as e:
        logger.error(f"❌ ONVIF discovery failed: {e}", exc_info=True)

        error_str = str(e)

        # Provide helpful error messages
        if "All authentication methods failed" in error_str:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Authentication Failed",
                    "message": "All authentication methods failed. Invalid credentials or ONVIF not enabled.",
                    "camera_ip": request.ip,
                    "username_tried": request.username,
                    "hints": [
                        "Verify camera credentials are correct",
                        "Check if ONVIF is enabled in camera settings",
                        "For Dahua: Setup > Network > ONVIF must be enabled",
                        "For Hikvision: Configuration > Network > Advanced Settings > Integration Protocol",
                        "Ensure user has ONVIF permissions",
                        "Try resetting camera ONVIF password",
                    ],
                },
            )
        elif "timeout" in error_str.lower() or "connection" in error_str.lower():
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
                        "Verify ONVIF port (usually 80, 8080, or 8000)",
                        "Check firewall settings",
                        "Try pinging the camera first",
                    ],
                },
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "ONVIF Discovery Failed",
                    "message": str(e),
                    "type": type(e).__name__,
                    "hints": [
                        "Verify camera supports ONVIF",
                        "Check camera network settings",
                        "Review camera documentation for ONVIF setup",
                    ],
                },
            )


@router.post("/ptz/move")
async def ptz_move_direct(
    ip: str = Query(..., description="Camera IP address"),
    username: str = Query(..., description="ONVIF username"),
    password: str = Query(..., description="ONVIF password"),
    x: float = Query(0.0, description="Pan: -1.0 (left) to 1.0 (right)"),
    y: float = Query(0.0, description="Tilt: -1.0 (down) to 1.0 (up)"),
    z: float = Query(0.0, description="Zoom: -1.0 (out) to 1.0 (in)"),
    port: int = Query(80, description="ONVIF port"),
    timeout: int = Query(1, description="Movement timeout in seconds"),
):
    """
    Control PTZ camera movement using direct IP/credentials

    Example:
    POST /api/v1/ptz/move?ip=192.168.1.12&username=admin&password=pass&x=0.5&y=0&z=0
    """
    try:
        logger.info(f"🔌 Direct PTZ move: {username}@{ip}:{port} (x={x}, y={y}, z={z})")

        # Create handler
        config = ONVIFConfig(ip=ip, port=port, username=username, password=password)
        handler = ONVIFHandler(config)

        # Execute move
        success = await handler.ptz_continuous_move(
            pan=x, tilt=y, zoom=z, timeout=timeout
        )

        if not success:
            raise HTTPException(status_code=500, detail="PTZ move command failed")

        logger.info(f"✅ PTZ moved successfully")

        return {
            "success": True,
            "message": f"PTZ moved: x={x}, y={y}, z={z}",
            "ip": ip,
            "pan": x,
            "tilt": y,
            "zoom": z,
        }

    except HTTPException:
        raise
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
        logger.info(f"🛑 Direct PTZ stop: {username}@{ip}:{port}")

        config = ONVIFConfig(ip=ip, port=port, username=username, password=password)
        handler = ONVIFHandler(config)

        success = await handler.ptz_stop()

        if not success:
            raise HTTPException(status_code=500, detail="PTZ stop command failed")

        logger.info(f"✅ PTZ stopped")

        return {"success": True, "message": "PTZ stopped", "ip": ip}

    except HTTPException:
        raise
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
        config = ONVIFConfig(ip=ip, port=port, username=username, password=password)
        handler = ONVIFHandler(config)

        position = await handler.ptz_get_status()

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
        config = ONVIFConfig(ip=ip, port=port, username=username, password=password)
        handler = ONVIFHandler(config)

        presets = await handler.ptz_get_presets()

        return {"success": True, "ip": ip, "presets": presets}

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
        config = ONVIFConfig(ip=ip, port=port, username=username, password=password)
        handler = ONVIFHandler(config)

        success = await handler.ptz_goto_preset(preset_token)

        if not success:
            raise HTTPException(status_code=500, detail="Goto preset command failed")

        return {
            "success": True,
            "ip": ip,
            "preset_token": preset_token,
            "message": f"Moved to preset {preset_token}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Goto preset failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ptz/presets/set")
async def ptz_set_preset_direct(
    ip: str = Query(..., description="Camera IP address"),
    username: str = Query(..., description="ONVIF username"),
    password: str = Query(..., description="ONVIF password"),
    preset_name: str = Query(..., description="Name for the new preset"),
    preset_token: Optional[str] = Query(
        None, description="Optional preset token to update"
    ),
    port: int = Query(80, description="ONVIF port"),
):
    """
    Set/create a preset at current position

    Example:
    POST /api/v1/ptz/presets/set?ip=192.168.1.12&username=admin&password=pass&preset_name=MyPreset
    """
    try:
        config = ONVIFConfig(ip=ip, port=port, username=username, password=password)
        handler = ONVIFHandler(config)

        token = await handler.ptz_set_preset(preset_name, preset_token)

        if not token:
            raise HTTPException(status_code=500, detail="Set preset command failed")

        return {
            "success": True,
            "ip": ip,
            "preset_name": preset_name,
            "preset_token": token,
            "message": f"Preset '{preset_name}' created/updated",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Set preset failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/ptz/presets")
async def ptz_remove_preset_direct(
    ip: str = Query(..., description="Camera IP address"),
    username: str = Query(..., description="ONVIF username"),
    password: str = Query(..., description="ONVIF password"),
    preset_token: str = Query(..., description="Preset token to remove"),
    port: int = Query(80, description="ONVIF port"),
):
    """
    Remove a preset

    Example:
    DELETE /api/v1/ptz/presets?ip=192.168.1.12&username=admin&password=pass&preset_token=1
    """
    try:
        config = ONVIFConfig(ip=ip, port=port, username=username, password=password)
        handler = ONVIFHandler(config)

        success = await handler.ptz_remove_preset(preset_token)

        if not success:
            raise HTTPException(status_code=500, detail="Remove preset command failed")

        return {
            "success": True,
            "ip": ip,
            "preset_token": preset_token,
            "message": f"Preset {preset_token} removed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Remove preset failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# CAMERA_ID ENDPOINTS (Using Database)
# ============================================================================


async def get_onvif_handler(
    camera_id: str, db: AsyncSession
) -> tuple[ONVIFHandler, Any]:
    """Build ONVIF handler from database camera (with caching)"""
    camera = await camera_service.get_camera(db, camera_id)

    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    ip_address = camera.ip_address
    username = camera.username
    password = camera.password
    onvif_port = camera.onvif_port or 80

    if not ip_address:
        raise HTTPException(status_code=400, detail="Camera IP not configured")

    config = ONVIFConfig(
        ip=ip_address, port=onvif_port, username=username, password=password
    )

    # Use cached instance
    handler = ONVIFHandler.get_instance(config)
    return handler, camera


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
    try:
        handler, camera = await get_onvif_handler(camera_id, db)

        success = await handler.ptz_continuous_move(
            pan=request.pan,
            tilt=request.tilt,
            zoom=request.zoom,
            timeout=request.timeout,
        )

        if not success:
            raise HTTPException(status_code=500, detail="PTZ move command failed")

        logger.info(
            f"✅ PTZ moved (camera_id={camera_id}): pan={request.pan}, tilt={request.tilt}"
        )

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "pan": request.pan,
            "tilt": request.tilt,
            "zoom": request.zoom,
        }

    except HTTPException:
        raise
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
    try:
        handler, camera = await get_onvif_handler(camera_id, db)

        success = await handler.ptz_stop()

        if not success:
            raise HTTPException(status_code=500, detail="PTZ stop command failed")

        return {"success": True, "camera_id": camera_id, "camera_name": camera.name}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ptz/{camera_id}/position")
async def ptz_get_position_by_id(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Get PTZ position using camera_id"""
    try:
        handler, camera = await get_onvif_handler(camera_id, db)

        position = await handler.ptz_get_status()

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
    try:
        handler, camera = await get_onvif_handler(camera_id, db)

        presets = await handler.ptz_get_presets()

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "presets": presets,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ptz/{camera_id}/presets/goto")
async def ptz_goto_preset_by_id(
    camera_id: str, request: PTZPresetRequest, db: AsyncSession = Depends(get_db)
):
    """Go to preset using camera_id"""
    try:
        handler, camera = await get_onvif_handler(camera_id, db)

        success = await handler.ptz_goto_preset(request.preset_token)

        if not success:
            raise HTTPException(status_code=500, detail="Goto preset command failed")

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "preset_token": request.preset_token,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ptz/{camera_id}/presets/set")
async def ptz_set_preset_by_id(
    camera_id: str, request: PTZSetPresetRequest, db: AsyncSession = Depends(get_db)
):
    """Set/create preset using camera_id"""
    try:
        handler, camera = await get_onvif_handler(camera_id, db)

        token = await handler.ptz_set_preset(request.preset_name, request.preset_token)

        if not token:
            raise HTTPException(status_code=500, detail="Set preset command failed")

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "preset_name": request.preset_name,
            "preset_token": token,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/ptz/{camera_id}/presets/{preset_token}")
async def ptz_remove_preset_by_id(
    camera_id: str, preset_token: str, db: AsyncSession = Depends(get_db)
):
    """Remove preset using camera_id"""
    try:
        handler, camera = await get_onvif_handler(camera_id, db)

        success = await handler.ptz_remove_preset(preset_token)

        if not success:
            raise HTTPException(status_code=500, detail="Remove preset command failed")

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "preset_token": preset_token,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ptz/{camera_id}/capabilities")
async def get_ptz_capabilities_by_id(
    camera_id: str, db: AsyncSession = Depends(get_db)
):
    """Get PTZ capabilities using camera_id"""
    try:
        handler, camera = await get_onvif_handler(camera_id, db)

        # Get device info to trigger auth discovery
        device_info = await handler.get_device_info()
        capabilities = await handler.get_capabilities()

        # Get profiles to check for audio
        profiles = await handler.get_profiles()
        has_audio = any("audio" in p.get("name", "").lower() for p in profiles)
        capabilities["audio"] = has_audio

        return {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "device": {
                "manufacturer": device_info["manufacturer"],
                "model": device_info["model"],
                "firmware": device_info["firmware"],
            },
            "capabilities": {
                "ptz": capabilities.get("ptz", False),
                "audio": capabilities.get("audio", False),
                "events": capabilities.get("events", False),
                "imaging": capabilities.get("imaging", False),
                "analytics": capabilities.get("analytics", False),
                "device_io": capabilities.get("device_io", False),
                "recording": capabilities.get("recording", False),
                "profiles": len(profiles),
            },
            "connection": {
                "ip": camera.ip_address,
                "onvif_port": camera.onvif_port,
                "auth_method": handler.working_auth.value
                if handler.working_auth
                else "unknown",
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
