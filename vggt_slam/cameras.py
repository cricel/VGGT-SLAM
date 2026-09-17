"""Camera backends for real-time SLAM."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Camera(ABC):
    """Abstract camera that produces BGR uint8 frames."""

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming."""
        ...

    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """Return a BGR uint8 (H, W, 3) frame, or None if unavailable."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release the device."""
        ...


class RealSenseCamera(Camera):
    """Intel RealSense color stream via pyrealsense2."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = None

    def start(self) -> None:
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        self._pipeline.start(config)

    def capture(self) -> Optional[np.ndarray]:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()) if color else None

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


class WebcamCamera(Camera):
    """Generic USB / built-in webcam via OpenCV."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._cap = None

    def start(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self._device}")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def capture(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


BACKENDS = {
    "realsense": RealSenseCamera,
    "webcam": WebcamCamera,
}
