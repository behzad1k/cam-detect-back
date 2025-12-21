# app/services/onvif_handler.py - Multi-Protocol ONVIF Handler

import asyncio
import base64
import hashlib
import logging
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Supported authentication methods"""

    DIGEST = "digest"
    BASIC = "basic"
    SOAP_WSSE = "soap_wsse"
    NONE = "none"


@dataclass
class ONVIFConfig:
    """ONVIF connection configuration"""

    ip: str
    port: int
    username: str
    password: str
    timeout: int = 10


@dataclass
class CameraCapabilities:
    """Camera capabilities"""

    manufacturer: str = "Unknown"
    model: str = "Unknown"
    firmware: str = "Unknown"
    serial: str = "Unknown"
    hardware_id: str = "Unknown"
    capabilities: Dict[str, bool] = None
    profiles: List[Dict] = None
    auth_method: AuthMethod = None

    def __post_init__(self):
        if self.profiles is None:
            self.profiles = []
        if self.capabilities is None:
            self.capabilities = {
                "ptz": False,
                "media": False,
                "imaging": False,
                "analytics": False,
                "events": False,
                "device_io": False,
                "recording": False,
                "replay": False,
                "receiver": False,
                "audio": False,
            }


class ONVIFHandler:
    """
    Comprehensive ONVIF handler supporting multiple protocols:
    - Plain SOAP with WS-Security (UsernameToken)
    - HTTP Digest Authentication
    - HTTP Basic Authentication
    - No authentication (for testing)

    Tries each method until one succeeds.
    """

    # Class-level cache for handler instances
    _instances = {}

    @classmethod
    def get_instance(cls, config: ONVIFConfig) -> "ONVIFHandler":
        """Get or create cached handler instance"""
        cache_key = f"{config.ip}:{config.port}:{config.username}"

        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls(config)

        return cls._instances[cache_key]

    def __init__(self, config: ONVIFConfig):
        self.config = config
        self.base_url = f"http://{config.ip}:{config.port}"
        self.device_url = f"{self.base_url}/onvif/device_service"
        self.media_url = f"{self.base_url}/onvif/media_service"
        self.ptz_url = f"{self.base_url}/onvif/ptz_service"
        self.imaging_url = f"{self.base_url}/onvif/imaging_service"

        self.working_auth: Optional[AuthMethod] = None
        self.profile_token: Optional[str] = None
        self.namespaces = {
            "soap": "http://www.w3.org/2003/05/soap-envelope",
            "tds": "http://www.onvif.org/ver10/device/wsdl",
            "trt": "http://www.onvif.org/ver10/media/wsdl",
            "tptz": "http://www.onvif.org/ver20/ptz/wsdl",
            "tt": "http://www.onvif.org/ver10/schema",
            "wsse": "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd",
            "wsu": "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd",
        }

    # ========================================================================
    # SOAP ENVELOPE BUILDERS
    # ========================================================================

    def _create_wsse_header(self) -> str:
        """Create WS-Security header with UsernameToken"""
        created = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        nonce = base64.b64encode(uuid.uuid4().bytes).decode("utf-8")

        # Create password digest: Base64(SHA-1(nonce + created + password))
        nonce_bytes = base64.b64decode(nonce)
        created_bytes = created.encode("utf-8")
        password_bytes = self.config.password.encode("utf-8")

        digest_input = nonce_bytes + created_bytes + password_bytes
        password_digest = base64.b64encode(hashlib.sha1(digest_input).digest()).decode(
            "utf-8"
        )

        return f'''
        <wsse:Security soap:mustUnderstand="true" xmlns:wsse="{self.namespaces["wsse"]}" xmlns:wsu="{self.namespaces["wsu"]}">
            <wsse:UsernameToken>
                <wsse:Username>{self.config.username}</wsse:Username>
                <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{password_digest}</wsse:Password>
                <wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{nonce}</wsse:Nonce>
                <wsu:Created>{created}</wsu:Created>
            </wsse:UsernameToken>
        </wsse:Security>
        '''

    def _build_soap_envelope(self, body: str, include_wsse: bool = True) -> str:
        """Build complete SOAP envelope"""
        wsse_header = self._create_wsse_header() if include_wsse else ""

        return f'''<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="{self.namespaces["soap"]}"
               xmlns:tds="{self.namespaces["tds"]}"
               xmlns:trt="{self.namespaces["trt"]}"
               xmlns:tptz="{self.namespaces["tptz"]}"
               xmlns:tt="{self.namespaces["tt"]}">
    <soap:Header>{wsse_header}</soap:Header>
    <soap:Body>{body}</soap:Body>
</soap:Envelope>'''

    # ========================================================================
    # HTTP REQUEST METHODS
    # ========================================================================

    def _try_request(
        self, url: str, soap_body: str, auth_method: AuthMethod
    ) -> Optional[requests.Response]:
        """Try a single request with specific auth method"""
        try:
            headers = {
                "Content-Type": "application/soap+xml; charset=utf-8",
                "Connection": "close",
            }

            # Build envelope based on auth method
            if auth_method == AuthMethod.SOAP_WSSE:
                envelope = self._build_soap_envelope(soap_body, include_wsse=True)
                auth = None
            elif auth_method == AuthMethod.DIGEST:
                envelope = self._build_soap_envelope(soap_body, include_wsse=False)
                auth = HTTPDigestAuth(self.config.username, self.config.password)
            elif auth_method == AuthMethod.BASIC:
                envelope = self._build_soap_envelope(soap_body, include_wsse=False)
                auth = HTTPBasicAuth(self.config.username, self.config.password)
            else:  # NONE
                envelope = self._build_soap_envelope(soap_body, include_wsse=False)
                auth = None

            response = requests.post(
                url,
                data=envelope.encode("utf-8"),
                headers=headers,
                auth=auth,
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                logger.info(f"✅ Success with {auth_method.value}")
                return response
            else:
                logger.debug(f"❌ {auth_method.value} failed: {response.status_code}")
                return None

        except Exception as e:
            logger.debug(f"❌ {auth_method.value} exception: {e}")
            return None

    def _send_request(self, url: str, soap_body: str) -> requests.Response:
        """
        Send SOAP request, trying all auth methods until one works
        """
        # If we already know which auth works, try that first
        if self.working_auth:
            response = self._try_request(url, soap_body, self.working_auth)
            if response:
                return response
            logger.warning(
                f"Previously working auth {self.working_auth.value} failed, retrying all methods"
            )

        # Try all auth methods in order of likelihood
        auth_methods = [
            AuthMethod.SOAP_WSSE,  # Most common for ONVIF
            AuthMethod.DIGEST,  # Common for Dahua, Hikvision
            AuthMethod.BASIC,  # Some older cameras
            AuthMethod.NONE,  # For testing/debug
        ]

        for auth_method in auth_methods:
            logger.info(f"🔄 Trying {auth_method.value} authentication...")
            response = self._try_request(url, soap_body, auth_method)

            if response:
                self.working_auth = auth_method
                logger.info(f"✅ Found working auth method: {auth_method.value}")

                # Debug: Log first 500 chars of response for troubleshooting
                logger.debug(f"Response preview: {response.text[:500]}")

                return response

        raise Exception("All authentication methods failed")

    # ========================================================================
    # XML PARSING HELPERS
    # ========================================================================

    def _find_element_by_tag_suffix(
        self, root: ET.Element, suffix: str
    ) -> Optional[ET.Element]:
        """Find first element whose tag ends with the given suffix (namespace-safe)"""
        for elem in root.iter():
            if elem.tag.endswith(suffix):
                return elem
        return None

    def _find_all_elements_by_tag_suffix(
        self, root: ET.Element, suffix: str
    ) -> List[ET.Element]:
        """Find all elements whose tag ends with the given suffix (namespace-safe)"""
        results = []
        for elem in root.iter():
            if elem.tag.endswith(suffix):
                results.append(elem)
        return results

    def _parse_response(
        self, response: requests.Response, tag_name: str
    ) -> Optional[ET.Element]:
        """Parse SOAP response and extract specific tag"""
        try:
            root = ET.fromstring(response.content)

            # Try to find the tag in the body
            for ns_prefix, ns_uri in self.namespaces.items():
                element = root.find(f".//{{{ns_uri}}}{tag_name}")
                if element is not None:
                    return element

            # Also try without namespace
            element = root.find(f".//{tag_name}")
            if element is not None:
                return element

            # Finally, try suffix matching
            return self._find_element_by_tag_suffix(root, tag_name)

        except Exception as e:
            logger.error(f"Parse error: {e}")
            return None

    def _get_text(self, element: ET.Element, tag: str, default: str = "Unknown") -> str:
        """Safely extract text from XML element"""
        if element is None:
            return default

        for ns_uri in self.namespaces.values():
            child = element.find(f"{{{ns_uri}}}{tag}")
            if child is not None and child.text:
                return child.text

        child = element.find(tag)
        if child is not None and child.text:
            return child.text

        return default

    # ========================================================================
    # DEVICE MANAGEMENT
    # ========================================================================

    async def get_device_info(self) -> Dict[str, str]:
        """Get device information"""
        soap_body = "<tds:GetDeviceInformation/>"

        response = self._send_request(self.device_url, soap_body)
        info_element = self._parse_response(response, "GetDeviceInformationResponse")

        return {
            "manufacturer": self._get_text(info_element, "Manufacturer"),
            "model": self._get_text(info_element, "Model"),
            "firmware": self._get_text(info_element, "FirmwareVersion"),
            "serial": self._get_text(info_element, "SerialNumber"),
            "hardware_id": self._get_text(info_element, "HardwareId"),
        }

    async def get_capabilities(self) -> Dict[str, Any]:
        """Get device capabilities"""
        # Request all capability categories
        soap_body = """
        <tds:GetCapabilities>
            <tds:Category>All</tds:Category>
        </tds:GetCapabilities>
        """

        response = self._send_request(self.device_url, soap_body)
        root = ET.fromstring(response.content)

        # Debug: Print response to see structure
        logger.debug(f"GetCapabilities full response:\n{response.text}")

        # Initialize capabilities
        capabilities = {
            "ptz": False,
            "media": False,
            "imaging": False,
            "analytics": False,
            "events": False,
            "device_io": False,
            "recording": False,
            "replay": False,
            "receiver": False,
            "audio": False,
        }

        # Find the Capabilities element
        capabilities_element = None
        for elem in root.iter():
            if elem.tag.endswith("Capabilities"):
                capabilities_element = elem
                logger.debug(f"Found Capabilities element: {elem.tag}")
                break

        if capabilities_element is None:
            logger.warning("Could not find Capabilities element in response")
            logger.debug(
                f"All elements in response: {[elem.tag for elem in root.iter()]}"
            )
            return capabilities

        # Debug: Log all children
        logger.debug(f"Capabilities has {len(list(capabilities_element))} children")

        # Check each child of Capabilities
        for child in capabilities_element:
            tag_name = child.tag.split("}")[-1]  # Remove namespace
            tag_lower = tag_name.lower()

            logger.debug(f"Examining child: {tag_name}")

            # Check if it has an XAddr (service endpoint)
            has_xaddr = False
            xaddr_value = None
            for subchild in child:
                subchild_name = subchild.tag.split("}")[-1]
                if subchild_name == "XAddr" and subchild.text:
                    has_xaddr = True
                    xaddr_value = subchild.text
                    logger.debug(f"  Found XAddr: {xaddr_value}")
                    break

            if has_xaddr:
                if "ptz" in tag_lower:
                    capabilities["ptz"] = True
                    logger.info(f"✅ PTZ capability found at {xaddr_value}")
                elif "media" in tag_lower:
                    capabilities["media"] = True
                    logger.info(f"✅ Media capability found at {xaddr_value}")
                elif "imaging" in tag_lower:
                    capabilities["imaging"] = True
                    logger.info(f"✅ Imaging capability found at {xaddr_value}")
                elif "analytics" in tag_lower:
                    capabilities["analytics"] = True
                    logger.info(f"✅ Analytics capability found at {xaddr_value}")
                elif "events" in tag_lower or "event" in tag_lower:
                    capabilities["events"] = True
                    logger.info(f"✅ Events capability found at {xaddr_value}")
                elif "deviceio" in tag_lower or "device" in tag_lower:
                    capabilities["device_io"] = True
                    logger.info(f"✅ DeviceIO capability found at {xaddr_value}")
                elif "recording" in tag_lower:
                    capabilities["recording"] = True
                    logger.info(f"✅ Recording capability found at {xaddr_value}")
                elif "replay" in tag_lower:
                    capabilities["replay"] = True
                    logger.info(f"✅ Replay capability found at {xaddr_value}")
                elif "receiver" in tag_lower:
                    capabilities["receiver"] = True
                    logger.info(f"✅ Receiver capability found at {xaddr_value}")
                else:
                    logger.debug(f"  Unknown capability type: {tag_name}")
            else:
                logger.debug(f"  No XAddr found for {tag_name}")

        logger.info(f"Final capabilities: {capabilities}")
        return capabilities

    # ========================================================================
    # MEDIA PROFILES
    # ========================================================================

    async def get_profiles(self) -> List[Dict[str, Any]]:
        """Get media profiles"""
        soap_body = "<trt:GetProfiles/>"

        response = self._send_request(self.media_url, soap_body)
        root = ET.fromstring(response.content)

        profiles = []

        # Find all Profile/Profiles elements
        profile_elements = self._find_all_elements_by_tag_suffix(root, "Profiles")

        for profile in profile_elements:
            token = profile.get("token")
            if not token:
                # Try to find token as child element
                for child in profile:
                    if child.tag.endswith("token"):
                        token = child.text
                        break

            if not token:
                continue

            name = self._get_text(profile, "Name", default=f"Profile_{token}")

            # Extract resolution
            width = 0
            height = 0
            fps = 25

            # Look for VideoEncoderConfiguration
            for child in profile.iter():
                if "VideoEncoderConfiguration" in child.tag:
                    # Look for Resolution within this config
                    for subchild in child.iter():
                        if "Resolution" in subchild.tag:
                            w = self._get_text(subchild, "Width", "0")
                            h = self._get_text(subchild, "Height", "0")
                            try:
                                width = int(w) if w and w != "Unknown" else 0
                                height = int(h) if h and h != "Unknown" else 0
                            except:
                                pass

                        if "RateControl" in subchild.tag:
                            fps_text = self._get_text(subchild, "FrameRateLimit", "25")
                            try:
                                fps = (
                                    int(float(fps_text))
                                    if fps_text != "Unknown"
                                    else 25
                                )
                            except:
                                fps = 25

            profiles.append(
                {
                    "token": token,
                    "name": name,
                    "width": width,
                    "height": height,
                    "fps": fps,
                }
            )

            # Store first profile token for PTZ operations
            if not self.profile_token:
                self.profile_token = token

        return profiles

    async def get_stream_uri(self, profile_token: str) -> str:
        """Get RTSP stream URI for a profile"""
        soap_body = f"""
        <trt:GetStreamUri>
            <trt:StreamSetup>
                <tt:Stream>RTP-Unicast</tt:Stream>
                <tt:Transport>
                    <tt:Protocol>RTSP</tt:Protocol>
                </tt:Transport>
            </trt:StreamSetup>
            <trt:ProfileToken>{profile_token}</trt:ProfileToken>
        </trt:GetStreamUri>
        """

        response = self._send_request(self.media_url, soap_body)
        uri_element = self._parse_response(response, "Uri")

        if uri_element is not None and uri_element.text:
            return uri_element.text

        return ""

    # ========================================================================
    # PTZ CONTROL
    # ========================================================================

    async def ptz_continuous_move(
        self, pan: float, tilt: float, zoom: float, timeout: int = 1
    ) -> bool:
        """
        Continuous PTZ move
        pan: -1.0 (left) to 1.0 (right)
        tilt: -1.0 (down) to 1.0 (up)
        zoom: -1.0 (out) to 1.0 (in)
        """
        if not self.profile_token:
            await self.get_profiles()

        logger.info(
            f"🎮 PTZ ContinuousMove: pan={pan:.3f}, tilt={tilt:.3f}, zoom={zoom:.3f}, timeout={timeout}s, profile={self.profile_token}"
        )

        soap_body = f'''
        <tptz:ContinuousMove>
            <tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>
            <tptz:Velocity>
                <tt:PanTilt x="{pan}" y="{tilt}"/>
                <tt:Zoom x="{zoom}"/>
            </tptz:Velocity>
            <tptz:Timeout>PT{timeout}S</tptz:Timeout>
        </tptz:ContinuousMove>
        '''

        try:
            response = self._send_request(self.ptz_url, soap_body)
            logger.info(f"✅ PTZ ContinuousMove command sent successfully")
            logger.debug(f"PTZ Response: {response.text[:500]}")
            return True
        except Exception as e:
            logger.error(f"❌ PTZ move failed: {e}", exc_info=True)
            return False

    async def ptz_stop(self) -> bool:
        """Stop PTZ movement"""
        if not self.profile_token:
            await self.get_profiles()

        soap_body = f"""
        <tptz:Stop>
            <tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>
            <tptz:PanTilt>true</tptz:PanTilt>
            <tptz:Zoom>true</tptz:Zoom>
        </tptz:Stop>
        """

        try:
            self._send_request(self.ptz_url, soap_body)
            return True
        except Exception as e:
            logger.error(f"PTZ stop failed: {e}")
            return False

    async def ptz_get_status(self) -> Dict[str, float]:
        """Get current PTZ position"""
        if not self.profile_token:
            await self.get_profiles()

        soap_body = f"""
        <tptz:GetStatus>
            <tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>
        </tptz:GetStatus>
        """

        try:
            response = self._send_request(self.ptz_url, soap_body)
            root = ET.fromstring(response.content)

            position = {"pan": 0.0, "tilt": 0.0, "zoom": 0.0}

            # Find PanTilt and Zoom elements using iter()
            for elem in root.iter():
                if "PanTilt" in elem.tag:
                    position["pan"] = float(elem.get("x", 0.0))
                    position["tilt"] = float(elem.get("y", 0.0))
                elif "Zoom" in elem.tag and elem.get("x") is not None:
                    position["zoom"] = float(elem.get("x", 0.0))

            return position
        except Exception as e:
            logger.error(f"Get PTZ status failed: {e}")
            return {"pan": 0.0, "tilt": 0.0, "zoom": 0.0}

    async def ptz_get_presets(self) -> List[Dict[str, str]]:
        """Get PTZ presets"""
        if not self.profile_token:
            await self.get_profiles()

        soap_body = f"""
        <tptz:GetPresets>
            <tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>
        </tptz:GetPresets>
        """

        try:
            response = self._send_request(self.ptz_url, soap_body)
            root = ET.fromstring(response.content)

            presets = []

            # Find all Preset elements
            preset_elements = self._find_all_elements_by_tag_suffix(root, "Preset")

            for preset in preset_elements:
                token = preset.get("token")
                if not token:
                    # Try to find token as child element
                    for child in preset:
                        if child.tag.endswith("token"):
                            token = child.text
                            break

                if not token:
                    continue

                name = self._get_text(preset, "Name", default=f"Preset_{token}")
                presets.append({"token": token, "name": name})

            return presets
        except Exception as e:
            logger.error(f"Get presets failed: {e}")
            return []

    async def ptz_goto_preset(self, preset_token: str) -> bool:
        """Go to a preset position"""
        if not self.profile_token:
            await self.get_profiles()

        soap_body = f"""
        <tptz:GotoPreset>
            <tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>
            <tptz:PresetToken>{preset_token}</tptz:PresetToken>
        </tptz:GotoPreset>
        """

        try:
            self._send_request(self.ptz_url, soap_body)
            return True
        except Exception as e:
            logger.error(f"Goto preset failed: {e}")
            return False

    async def ptz_set_preset(
        self, preset_name: str, preset_token: Optional[str] = None
    ) -> Optional[str]:
        """Set/create a preset at current position"""
        if not self.profile_token:
            await self.get_profiles()

        preset_token_xml = (
            f"<tptz:PresetToken>{preset_token}</tptz:PresetToken>"
            if preset_token
            else ""
        )

        soap_body = f"""
        <tptz:SetPreset>
            <tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>
            <tptz:PresetName>{preset_name}</tptz:PresetName>
            {preset_token_xml}
        </tptz:SetPreset>
        """

        try:
            response = self._send_request(self.ptz_url, soap_body)
            preset_token_elem = self._parse_response(response, "PresetToken")
            if preset_token_elem is not None and preset_token_elem.text:
                return preset_token_elem.text
            return preset_token
        except Exception as e:
            logger.error(f"Set preset failed: {e}")
            return None

    async def ptz_remove_preset(self, preset_token: str) -> bool:
        """Remove a preset"""
        if not self.profile_token:
            await self.get_profiles()

        soap_body = f"""
        <tptz:RemovePreset>
            <tptz:ProfileToken>{self.profile_token}</tptz:ProfileToken>
            <tptz:PresetToken>{preset_token}</tptz:PresetToken>
        </tptz:RemovePreset>
        """

        try:
            self._send_request(self.ptz_url, soap_body)
            return True
        except Exception as e:
            logger.error(f"Remove preset failed: {e}")
            return False

    # ========================================================================
    # DISCOVERY & FULL CAPABILITY CHECK
    # ========================================================================

    async def discover(self) -> CameraCapabilities:
        """
        Full camera discovery:
        - Test all auth methods
        - Get device info
        - Get capabilities
        - Get media profiles
        - Get stream URIs
        """
        logger.info(
            f"🔍 Starting ONVIF discovery for {self.config.ip}:{self.config.port}"
        )

        capabilities_obj = CameraCapabilities()

        try:
            # Get device info
            device_info = await self.get_device_info()
            capabilities_obj.manufacturer = device_info["manufacturer"]
            capabilities_obj.model = device_info["model"]
            capabilities_obj.firmware = device_info["firmware"]
            capabilities_obj.serial = device_info["serial"]
            capabilities_obj.hardware_id = device_info["hardware_id"]
            capabilities_obj.auth_method = self.working_auth

            logger.info(
                f"✅ Device: {capabilities_obj.manufacturer} {capabilities_obj.model}"
            )

            # Get capabilities
            caps = await self.get_capabilities()
            capabilities_obj.capabilities = caps

            # Get profiles
            profiles = await self.get_profiles()

            # Check for audio in profiles
            has_audio = any("audio" in p.get("name", "").lower() for p in profiles)
            capabilities_obj.capabilities["audio"] = has_audio

            # Get stream URIs
            for profile in profiles:
                try:
                    uri = await self.get_stream_uri(profile["token"])
                    profile["uri"] = uri
                except Exception as e:
                    logger.warning(
                        f"Could not get URI for profile {profile['name']}: {e}"
                    )
                    profile["uri"] = ""

            capabilities_obj.profiles = profiles

            logger.info(f"✅ Discovery complete:")
            logger.info(
                f"   Auth: {self.working_auth.value if self.working_auth else 'Unknown'}"
            )
            logger.info(f"   PTZ: {'Yes' if caps.get('ptz') else 'No'}")
            logger.info(f"   Events: {'Yes' if caps.get('events') else 'No'}")
            logger.info(f"   Analytics: {'Yes' if caps.get('analytics') else 'No'}")
            logger.info(f"   Profiles: {len(profiles)}")

            return capabilities_obj

        except Exception as e:
            logger.error(f"❌ Discovery failed: {e}", exc_info=True)
            raise


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def discover_camera(
    ip: str, port: int, username: str, password: str, timeout: int = 10
) -> CameraCapabilities:
    """Convenience function for camera discovery"""
    config = ONVIFConfig(
        ip=ip, port=port, username=username, password=password, timeout=timeout
    )
    handler = ONVIFHandler(config)
    return await handler.discover()


async def create_handler(
    ip: str, port: int, username: str, password: str, timeout: int = 10
) -> ONVIFHandler:
    """Create and initialize ONVIF handler"""
    config = ONVIFConfig(
        ip=ip, port=port, username=username, password=password, timeout=timeout
    )
    handler = ONVIFHandler(config)

    # Do initial discovery to find working auth
    await handler.get_device_info()

    return handler
