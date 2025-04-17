from typing import Any, Self

from numpy.typing import NDArray
from picamera2 import Picamera2, Preview

from . import base


class RaspberryCameraInterface(base.CameraInterface):
    def __init__(self, frame_dim: tuple[int, int], buffer_count: int):
        self.picamera = Picamera2()

        config = self.picamera.create_video_configuration(
            main={
                "size": frame_dim,
                "format": "RGB888",
            },
            buffer_count=buffer_count
        )
        self.picamera.configure(config)

        self.picamera.set_controls(
            {
                "AfMode": 2,
                "AwbMode": 0,
            }
        )

        self.picamera.start()

    @classmethod
    def get_instance(cls, kwargs: dict[str, Any]) -> Self:
        width = kwargs.get("width", 1280)
        height = kwargs.get("height", 720)
        buffer_count = kwargs.get("buffer_count", 4)
        for param in [width, height, buffer_count]:
            if not isinstance(param, int):
                raise TypeError("All kwargs for RaspberryCameraInterface have to be int")
        return cls((width, height), buffer_count)

    def get_frame(self) -> NDArray:
        frame = self.picamera.capture_array()
        return frame
