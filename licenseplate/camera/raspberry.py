from numpy.typing import NDArray
from picamera2 import Picamera2
import cv2

from . import base


class RaspberryCameraInterface(base.CameraInterface):
    def __init__(self):
        self.picamera = Picamera2()
        self.picamera.configure(self.picamera.create_preview_configuration())
        self.picamera.start()

    def get_frame(self) -> NDArray:
        frame = self.picamera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
