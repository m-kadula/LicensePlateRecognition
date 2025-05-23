import cv2
import numpy as np
from numpy.typing import NDArray

from ..base import CameraInterface


class DefaultCameraInterface(CameraInterface):
    def __init__(self, device: int = 0):
        self.device = device
        self.cap: cv2.VideoCapture | None = None

    def get_frame(self) -> NDArray:
        assert isinstance(self.cap, cv2.VideoCapture)
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab a frame.")
            return np.zeros(shape=(3, 10, 10))
        return frame

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise IOError(f"Camera {self.device} cannot be accessed.")

    def stop(self) -> None:
        assert isinstance(self.cap, cv2.VideoCapture)
        self.cap.release()
