import logging
import threading
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


class DetectionLoop:
    def __init__(
        self,
        detection_model: PlateDetectionModel,
        camera: CameraInterface,
        action: ActionInterface,
        logger: logging.Logger | None = None,
        max_fps: int = 30,
    ):
        self.detection_model = detection_model
        self.camera = camera
        self.action = action
        self.logger = logger
        self.max_fps = max_fps

        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run)
        self._stop_now = False

    def run(self):
        fps_sum = 0.0
        iteration = 0

        while True:
            with self.lock:
                if self._stop_now:
                    break

            frame_time, lasted, detected_plates, detected_text = detection_iteration(
                self.detection_model, self.camera, self.action
            )

            iteration += 1
            fps_sum += 1 / lasted
            if self.logger is not None and detected_plates > 0:
                self.logger.info(
                    f"Iteration: {iteration}, FPS now: {round(1 / lasted, 2)}, FPS average: {round(fps_sum / iteration, 2)}\n"
                    f"Detected plates: {detected_plates}, detected text: {detected_text}.\n"
                )
            if 1 / self.max_fps - lasted > 0:
                sleep(1 / self.max_fps - lasted)

        if self.logger is not None:
            self.logger.info(
                f"Loop ended after {iteration} iterations with the average of {round(fps_sum / iteration, 2)} FPS."
            )

    def start_thread(self):
        if self.thread.is_alive():
            raise RuntimeError("The thread is already running.")
        with self.lock:
            self._stop_now = False
        self.thread.start()

    def stop_thread(self):
        if not self.thread.is_alive():
            raise RuntimeError("The thread is not running.")
        with self.lock:
            self._stop_now = True
        self.thread.join()
