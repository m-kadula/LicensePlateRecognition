from typing import Self

import cv2
import numpy as np
from numpy.typing import NDArray

from .base import CameraInterface


class DefaultCameraInterface(CameraInterface):
    def __init__(self, device: int = 0):
        self.device = device
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise IOError(f"Camera {self.device} cannot be accessed.")

    @classmethod
    def get_instance(cls, *, device: int) -> Self:
        return cls(device)

    def get_frame(self) -> NDArray:
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab a frame.")
            return np.zeros(shape=(3, 10, 10))
        return frame

    def deactivate(self) -> None:
        self.cap.release()
