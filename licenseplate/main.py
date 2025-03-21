from pathlib import Path
from typing import NoReturn
from datetime import datetime
from time import sleep

from .detection import PlateDetectionModel
from .camera.base import CameraInterface
from .action.base import ActionInterface


def detection_loop(
    detection_model: PlateDetectionModel,
    camera: CameraInterface,
    action: ActionInterface,
    max_fps: int = 30,
) -> NoReturn:
    fps_sum = 0.0
    fps_count = 0

    while True:
        start = datetime.now()
        frame = camera.get_frame()
        plates = detection_model.detect_plates(frame)
        if plates:
            action.action_if_found(frame, plates)
        lasted = (datetime.now() - start).total_seconds()
        fps_count += 1
        fps_sum += 1 / lasted
        print(f"FPS now: {1 / lasted}, FPS average: {fps_sum / fps_count}")
        if 1 / max_fps - lasted > 0:
            sleep(1 / max_fps - lasted)


if __name__ == "__main__":
    from .camera.macos import MacOSCameraInterface
    from .action.localsave import LocalSaveInterface
    from .preprocessing import preprocess_polish_license_plate

    model = PlateDetectionModel(
        Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
        preprocess_polish_license_plate
    )

    detection_loop(
        model,
        MacOSCameraInterface(),
        LocalSaveInterface(Path(__file__).parents[1] / "detected", show_debug_boxes=True),
    )
