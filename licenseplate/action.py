from datetime import datetime
from pathlib import Path
from time import sleep

from .base import ActionInterface, CameraInterface
from .detection import PlateDetectionModel


class LocalSave(ActionInterface):
    def __init__(
        self,
        detection_model: PlateDetectionModel,
        camera: CameraInterface,
        max_fps: int,
        logging_root: Path,
        show_debug_boxes: bool = False,
    ):
        super().__init__(detection_model, camera, max_fps)
        self.logging_root = logging_root
        self.debug_boxes = show_debug_boxes

    def loop(self):
        lasted = 1 / self.max_fps

        while not self.stop_signal_initiated():

            frame_time = datetime.now()

            frame = self.camera.get_frame()
            plates = self.detection_model.detect_plates(frame)
            # TODO: log detected plates
            lasted = (datetime.now() - frame_time).total_seconds()

            if 1 / self.max_fps - lasted > 0:
                sleep(1 / self.max_fps - lasted)
