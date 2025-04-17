from numpy.typing import NDArray
from picamera2 import Picamera2, Preview

from . import base


class RaspberryCameraInterface(base.CameraInterface):
    def __init__(self):
        self.picamera = Picamera2()

        config = self.picamera.create_video_configuration(
            main={
                "size": (1280, 720),
                "format": "RGB888",
            },
            buffer_count=4
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
