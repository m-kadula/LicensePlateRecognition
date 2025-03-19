from pathlib import Path
from typing import NoReturn
from datetime import datetime
from time import sleep

from .detection import (
    LicensePlateFinder,
    TextExtractor,
    LicensePlateValidator,
    detect_plates,
)
from .camera.base import CameraInterface
from .action.base import ActionInterface


class DetectionLoop:
    def __init__(
        self,
        finder: LicensePlateFinder,
        extractor: TextExtractor,
        validator: LicensePlateValidator,
        camera_interface: CameraInterface,
        action_interface: ActionInterface,
        max_fps: int = 30,
    ):
        self.finder = finder
        self.extractor = extractor
        self.validator = validator
        self.camera = camera_interface
        self.action = action_interface
        self.max_fps = max_fps

    def loop(self) -> NoReturn:
        fps_sum = 0.0
        fps_count = 0

        while True:
            start = datetime.now()
            frame = self.camera.get_frame()
            plates = detect_plates(frame, self.finder, self.extractor, self.validator)
            if plates:
                self.action.action_if_found(frame, plates)
            lasted = (datetime.now() - start).total_seconds()
            fps_count += 1
            fps_sum += 1 / lasted
            print(f"FPS now: {1 / lasted}, FPS average: {fps_sum / fps_count}")
            if 1 / self.max_fps - lasted > 0:
                sleep(1 / self.max_fps - lasted)

    def __call__(self) -> NoReturn:
        self.loop()


if __name__ == "__main__":
    from .camera.macos import MacOSCameraInterface
    from .action.localsave import LocalSaveInterface

    loop = DetectionLoop(
        LicensePlateFinder(
            Path(__file__).parents[1] / "runs/detect/train/weights/best.pt"
        ),
        TextExtractor(),
        LicensePlateValidator(),
        MacOSCameraInterface(),
        LocalSaveInterface(
            Path(__file__).parents[1] / "detected", show_debug_boxes=True
        ),
    )
    loop()
