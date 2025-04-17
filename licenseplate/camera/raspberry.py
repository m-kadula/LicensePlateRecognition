from numpy.typing import NDArray
from picamera2 import Picamera2, Preview

from . import base


class RaspberryCameraInterface(base.CameraInterface):
    def __init__(self):
        self.picamera = Picamera2()

        config = self.picamera.create_still_configuration(
            main={
                "size": (1920, 1080),
                "format": "RGB888",
            },
            buffer_count=3,
        )
        self.picamera.configure(config)

        self.picamera.set_controls(
            {
                "AfMode": 2,
                "AwbMode": 0,
            }
        )

        self.picamera.start()

    def get_frame(self) -> NDArray:
        frame = self.picamera.capture_array()
        return frame
