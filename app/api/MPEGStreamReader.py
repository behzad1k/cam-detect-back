import re

import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth


class MJPEGStreamReader:
    """Alternative MJPEG reader using requests library - MORE RELIABLE"""

    def __init__(self, url: str, username: str = None, password: str = None):
        self.url = url
        self.username = username
        self.password = password
        self.stream = None
        self.running = False

    def start(self):
        """Start the stream"""
        auth = None
        if self.username and self.password:
            # Try digest auth first (Dahua prefers this)
            auth = HTTPDigestAuth(self.username, self.password)

        try:
            self.stream = requests.get(
                self.url,
                auth=auth,
                stream=True,
                timeout=10,
                headers={"Connection": "keep-alive"},
            )

            if self.stream.status_code == 401:
                # Try basic auth
                auth = HTTPBasicAuth(self.username, self.password)
                self.stream = requests.get(
                    self.url,
                    auth=auth,
                    stream=True,
                    timeout=10,
                    headers={"Connection": "keep-alive"},
                )

            self.running = True
            logger.info(f"✅ MJPEG stream started: {self.url}")
            return True

        except Exception as e:
            logger.error(f"Failed to start MJPEG stream: {e}")
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from MJPEG stream"""
        if not self.running or not self.stream:
            return None

        try:
            # Read until we find a JPEG image
            bytes_data = b""

            for chunk in self.stream.iter_content(chunk_size=1024):
                bytes_data += chunk

                # Look for JPEG start (0xFFD8) and end (0xFFD9) markers
                start = bytes_data.find(b"\xff\xd8")
                end = bytes_data.find(b"\xff\xd9")

                if start != -1 and end != -1 and end > start:
                    # Found a complete JPEG image
                    jpg = bytes_data[start : end + 2]
                    bytes_data = bytes_data[end + 2 :]

                    # Decode JPEG to numpy array
                    nparr = np.frombuffer(jpg, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        return frame

        except Exception as e:
            logger.error(f"Error reading MJPEG frame: {e}")
            return None

        return None

    def stop(self):
        """Stop the stream"""
        self.running = False
        if self.stream:
            self.stream.close()
