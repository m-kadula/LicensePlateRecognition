from numpy.typing import NDArray
from picamera2 import Picamera2, Preview
import cv2

from . import base


class RaspberryCameraInterface(base.CameraInterface):
    def __init__(self):
        self.picamera = Picamera2()

        config = self.picamera.create_still_configuration(
            main={
                "size": (1920, 1080),
                "format": "RGB888",
            },
            buffer_count=3
        )
        self.picamera.configure(config)

        controls = {
            "AfMode": 2,
            "AfTrigger": 0,
        }
        self.picamera.set_controls(controls)

        self.picamera.start()

    def get_frame(self) -> NDArray:
        frame = self.picamera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
