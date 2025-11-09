"""
FPS Rate Limiter for Camera Streams
Ensures frames are processed at the configured FPS rate
"""
import time
from typing import Optional


class FPSRateLimiter:
    """
    Rate limiter to control frame processing speed based on target FPS.
    """

    def __init__(self, target_fps: float):
        """
        Initialize the rate limiter.

        Args:
            target_fps: Target frames per second (e.g., 15.0)
        """
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        self.last_frame_time: Optional[float] = None
        self.frame_count = 0
        self.start_time = time.time()

    def should_process_frame(self) -> bool:
        """
        Determine if enough time has passed to process the next frame.

        Returns:
            bool: True if frame should be processed, False otherwise
        """
        current_time = time.time()

        # First frame always processes
        if self.last_frame_time is None:
            self.last_frame_time = current_time
            self.frame_count += 1
            return True

        # Check if enough time has elapsed
        elapsed = current_time - self.last_frame_time

        if elapsed >= self.frame_interval:
            self.last_frame_time = current_time
            self.frame_count += 1
            return True

        return False

    def wait_if_needed(self) -> None:
        """
        Sleep if necessary to maintain target FPS.
        More precise than should_process_frame() for active rate limiting.
        """
        current_time = time.time()

        if self.last_frame_time is None:
            self.last_frame_time = current_time
            return

        elapsed = current_time - self.last_frame_time
        sleep_time = self.frame_interval - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

        self.last_frame_time = time.time()
        self.frame_count += 1

    def get_actual_fps(self) -> float:
        """
        Calculate the actual FPS based on frames processed.

        Returns:
            float: Actual frames per second
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return 0.0
        return self.frame_count / elapsed_time

    def reset(self) -> None:
        """Reset the rate limiter."""
        self.last_frame_time = None
        self.frame_count = 0
        self.start_time = time.time()

    def update_target_fps(self, new_fps: float) -> None:
        """
        Update the target FPS.

        Args:
            new_fps: New target frames per second
        """
        self.target_fps = new_fps
        self.frame_interval = 1.0 / new_fps if new_fps > 0 else 0