from numpy.typing import NDArray
from picamera2 import Picamera2, Preview

from ..base import CameraInterface


class RaspberryCameraInterface(CameraInterface):
    def __init__(self, frame_dim: tuple[int, int], buffer_count: int):
        self.picamera = Picamera2()

        config = self.picamera.create_video_configuration(
            main={
                "size": frame_dim,
                "format": "RGB888",
            },
            buffer_count=buffer_count,
        )
        self.picamera.configure(config)

        self.picamera.set_controls(
            {
                "AfMode": 2,
                "AwbMode": 0,
            }
        )

    def get_frame(self) -> NDArray:
        frame = self.picamera.capture_array()
        return frame

    def start(self) -> None:
        self.picamera.start()

    def stop(self) -> None:
        if self.picamera.started:
            self.picamera.stop()
