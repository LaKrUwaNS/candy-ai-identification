"""Camera operations and video streaming lifecycle service."""

import time
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraService:
    """Manages OpenCV webcam initialization, frame grabbing, and lifecycle release."""

    @staticmethod
    def is_camera_available(device_index: int = 0) -> bool:
        """Probe whether a camera index is openable."""
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return bool(ret)

    @staticmethod
    def open_camera(
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> Optional[cv2.VideoCapture]:
        """Open and configure VideoCapture device."""
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    @staticmethod
    def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera and return (success, bgr_frame)."""
        if cap is None or not cap.isOpened():
            return False, None
        return cap.read()

    @staticmethod
    def release_camera(cap: Optional[cv2.VideoCapture]) -> None:
        """Safely release camera handle."""
        if cap is not None and cap.isOpened():
            cap.release()


class FPSCalculator:
    """Helper to compute rolling frames-per-second metric."""

    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.frame_count = 0
        self.last_time = time.time()
        self.current_fps = 0.0

    def tick(self) -> float:
        """Record frame arrival and return current rolling FPS."""
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time
        if elapsed >= self.interval:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = now
        return self.current_fps

