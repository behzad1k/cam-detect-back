# app/api/camera_proxy.py
"""
Camera stream proxy with Digest Authentication and browser-compatible MJPEG streaming
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import aiohttp
import logging
from urllib.parse import urlparse, unquote
import asyncio
from aiohttp import BasicAuth
import hashlib
import re

from app.database.base import get_db
from app.services.camera_service import camera_service

router = APIRouter()
logger = logging.getLogger(__name__)


class DigestAuth:
  """Manual implementation of Digest Authentication for aiohttp"""

  def __init__(self, username: str, password: str):
    self.username = username
    self.password = password
    self.nc = 0

  def _parse_challenge(self, www_authenticate: str) -> dict:
    """Parse the WWW-Authenticate header"""
    challenge = {}
    for match in re.finditer(r'(\w+)=("[^"]*"|[^,\s]*)', www_authenticate):
      key, value = match.groups()
      challenge[key] = value.strip('"')
    return challenge

  def _build_digest_header(self, method: str, uri: str, challenge: dict) -> str:
    """Build the Authorization header for Digest auth"""
    realm = challenge.get('realm', '')
    nonce = challenge.get('nonce', '')
    qop = challenge.get('qop', '')
    opaque = challenge.get('opaque', '')
    algorithm = challenge.get('algorithm', 'MD5')

    self.nc += 1
    nc_value = f'{self.nc:08x}'

    import random
    import string
    cnonce = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

    ha1_input = f"{self.username}:{realm}:{self.password}"
    ha1 = hashlib.md5(ha1_input.encode()).hexdigest()

    ha2_input = f"{method}:{uri}"
    ha2 = hashlib.md5(ha2_input.encode()).hexdigest()

    if qop:
      response_input = f"{ha1}:{nonce}:{nc_value}:{cnonce}:{qop}:{ha2}"
    else:
      response_input = f"{ha1}:{nonce}:{ha2}"
    response_hash = hashlib.md5(response_input.encode()).hexdigest()

    auth_header = (
      f'Digest username="{self.username}", '
      f'realm="{realm}", '
      f'nonce="{nonce}", '
      f'uri="{uri}", '
      f'response="{response_hash}"'
    )

    if qop:
      auth_header += f', qop={qop}, nc={nc_value}, cnonce="{cnonce}"'

    if opaque:
      auth_header += f', opaque="{opaque}"'

    if algorithm:
      auth_header += f', algorithm={algorithm}'

    return auth_header


@router.get("/cameras/{camera_id}/stream")
async def proxy_camera_stream(
  camera_id: str,
  request: Request,
  db: AsyncSession = Depends(get_db)
):
  """
  Proxy camera MJPEG stream - browser compatible version
  """
  try:
    camera = await camera_service.get_camera(db, camera_id)

    if not camera:
      raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    if not camera.rtsp_url:
      raise HTTPException(status_code=400, detail="Camera has no stream URL configured")

    original_url = camera.rtsp_url
    logger.info(f"📹 Proxying stream for camera {camera_id}")

    # Parse URL and extract credentials
    parsed = urlparse(original_url)
    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None

    if username:
      logger.info(f"👤 Username: {username}")
    if password:
      logger.info(f"🔑 Password length: {len(password)} chars")

    # Build URL without credentials
    stream_url = f"{parsed.scheme}://{parsed.hostname}"
    if parsed.port:
      stream_url += f":{parsed.port}"
    stream_url += parsed.path
    if parsed.query:
      stream_url += f"?{parsed.query}"

    uri = parsed.path
    if parsed.query:
      uri += f"?{parsed.query}"

    logger.info(f"🔗 Stream URL: {stream_url}")

    # Stream generator that passes through the camera stream
    async def stream_generator():
      timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=30)

      try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
          logger.info(f"📡 Connecting to camera...")

          # Probe for auth type
          headers = {
            'User-Agent': 'Mozilla/5.0',
            'Connection': 'keep-alive',
            'Accept': 'multipart/x-mixed-replace,image/jpeg'
          }

          auth_type = None
          www_auth_header = None

          async with session.get(stream_url, headers=headers, allow_redirects=False) as probe:
            if probe.status == 401:
              www_auth_header = probe.headers.get('WWW-Authenticate', '')
              if 'Digest' in www_auth_header:
                auth_type = 'digest'
                logger.info(f"🔐 Using Digest authentication")
              elif 'Basic' in www_auth_header:
                auth_type = 'basic'
                logger.info(f"🔐 Using Basic authentication")

          # Build auth header
          if auth_type == 'digest' and username and password:
            digest = DigestAuth(username, password)
            challenge = digest._parse_challenge(www_auth_header)
            auth_header = digest._build_digest_header('GET', uri, challenge)
            headers['Authorization'] = auth_header
          elif username and password:
            import base64
            credentials = f"{username}:{password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers['Authorization'] = f'Basic {encoded}'

          # Connect with auth
          async with session.get(stream_url, headers=headers) as response:
            logger.info(f"📊 Status: {response.status}")

            if response.status != 200:
              logger.error(f"❌ Camera returned status {response.status}")
              return

            logger.info(f"✅ Connected! Content-Type: {response.headers.get('Content-Type')}")

            chunk_count = 0
            total_bytes = 0

            # Simply pass through all chunks from camera
            async for chunk in response.content.iter_any():
              # Check if client disconnected
              if await request.is_disconnected():
                logger.info("🔌 Client disconnected")
                break

              if chunk:
                chunk_count += 1
                total_bytes += len(chunk)

                if chunk_count == 1:
                  logger.info(f"✅ First chunk: {len(chunk)} bytes")

                # Yield chunk as-is
                yield chunk

            logger.info(f"🏁 Stream ended: {chunk_count} chunks, {total_bytes / 1024:.1f}KB")

      except asyncio.CancelledError:
        logger.info("⏹️ Stream cancelled")
      except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)

    # Return streaming response with camera's content type
    return StreamingResponse(
      stream_generator(),
      media_type="multipart/x-mixed-replace; boundary=myboundary",  # Match camera's boundary
      headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*"
      }
    )

  except HTTPException:
    raise
  except Exception as e:
    logger.error(f"❌ Setup error: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))


@router.options("/cameras/{camera_id}/stream")
async def proxy_camera_stream_options(camera_id: str):
  """Handle CORS preflight"""
  from fastapi.responses import JSONResponse
  return JSONResponse(
    content={"status": "ok"},
    headers={
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Allow-Headers": "*"
    }
  )