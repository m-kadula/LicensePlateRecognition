import logging
from typing import NoReturn
from datetime import datetime
from time import sleep

from .detection import PlateDetectionModel
from .camera.base import CameraInterface
from .action.base import ActionInterface


def detection_iteration(
    detection_model: PlateDetectionModel,
    camera: CameraInterface,
    action: ActionInterface,
) -> tuple[datetime, float, int, int]:
    frame_time = datetime.now()

    frame = camera.get_frame()
    plates = detection_model.detect_plates(frame)
    if plates:
        action.action_if_found(frame, plates, frame_time)
    else:
        action.action_if_not_found(frame, frame_time)

    lasted = (datetime.now() - frame_time).total_seconds()

    return frame_time, lasted, len(plates), sum(len(x[1]) for x in plates)


def detection_loop(
    detection_model: PlateDetectionModel,
    camera: CameraInterface,
    action: ActionInterface,
    logger: logging.Logger | None = None,
    max_fps: int = 30,
) -> NoReturn:
    fps_sum = 0.0
    iteration = 0

    try:
        while True:
            frame_time, lasted, detected_plates, detected_text = detection_iteration(
                detection_model, camera, action
            )

            iteration += 1
            fps_sum += 1 / lasted
            if logger is not None:
                logger.info(
                    f"Iteration: {iteration}, FPS now: {round(1 / lasted, 2)}, FPS average: {round(fps_sum / iteration, 2)}\n"
                    f"Detected plates: {detected_plates}, detected text: {detected_text}.\n"
                )
            if 1 / max_fps - lasted > 0:
                sleep(1 / max_fps - lasted)
    except KeyboardInterrupt:
        if logger is not None:
            logger.info(
                f"Loop ended after {iteration} iterations with the average of {round(fps_sum / iteration, 2)} FPS."
            )
