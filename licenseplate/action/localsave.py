from datetime import datetime
from typing import Self, Any, TextIO
from pathlib import Path
from logging import Logger
from time import sleep

from numpy.typing import NDArray
from pydantic import BaseModel
import cv2

from .base import ActionInterface, BaseActionManager
from ..detection import PlateDetectionModel
from ..camera.base import CameraInterface
from ..detection import FinderResult, ExtractorResult, visualise
from ..logger import get_standard_logger


class _Message(BaseModel):
    original_image: NDArray
    detected: list[tuple[FinderResult, list[ExtractorResult]]]
    visualised: NDArray
    time: datetime
    framerate: float

    class Config:
        arbitrary_types_allowed = True


class LocalSave(ActionInterface):
    def __init__(
        self,
        detection_model: PlateDetectionModel,
        camera: CameraInterface,
        max_fps: int,
        show_debug_boxes: bool = False,
        save_all_photos: bool = False,
    ):
        super().__init__(detection_model, camera, max_fps)
        self.debug_boxes = show_debug_boxes
        self.save_all_photos = save_all_photos

    @classmethod
    def get_instance(
        cls,
        detection_model: PlateDetectionModel,
        camera: CameraInterface,
        max_fps: int,
        kwargs: dict[str, Any],
    ) -> Self:
        show_debug_boxes = kwargs["show_debug_boxes"]
        if not isinstance(show_debug_boxes, bool):
            raise TypeError("show_debug_boxes has to be a bool")
        save_all_photos = kwargs.get("save_all_photos", False)
        if not isinstance(save_all_photos, bool):
            raise TypeError("show_debug_boxes has to be a bool")
        return cls(detection_model, camera, max_fps, show_debug_boxes, save_all_photos)

    def loop(self):
        lasted = 1 / self.max_fps

        while True:
            with self.lock:
                if self._stop_now:
                    break

            frame_time = datetime.now()

            frame = self.camera.get_frame()
            plates = self.detection_model.detect_plates(frame)
            if plates or self.save_all_photos:
                visualised = visualise(frame, plates, show_debug_boxes=self.debug_boxes)
                self.report_to_manager(
                    _Message(
                        detected=plates,
                        original_image=frame,
                        visualised=visualised,
                        time=frame_time,
                        framerate=1 / lasted,
                    )
                )
            lasted = (datetime.now() - frame_time).total_seconds()

            if 1 / self.max_fps - lasted > 0:
                sleep(1 / self.max_fps - lasted)


class LocalSaveManager(BaseActionManager):
    def __init__(self, logging_path: Path, log_original: bool = False):
        super().__init__()
        self.logging_path = logging_path
        self.log_original = log_original
        self.loggers: dict[ActionInterface, Logger] = {}
        self.ios: list[TextIO] = []

        if not self.logging_path.exists():
            self.logging_path.mkdir()

    @classmethod
    def get_instance(cls, kwargs: dict[str, Any]) -> Self:
        if "logging_path" not in kwargs:
            raise ValueError("Field 'logging_path' is required (str)")
        logging_path = kwargs["logging_path"]
        if not isinstance(logging_path, str):
            raise TypeError("logging_path has to be a string")
        log_original = kwargs.get("log_original", False)
        if not isinstance(log_original, bool):
            raise TypeError("log_original has to be a bool")
        return cls(Path(logging_path).resolve(), log_original)

    def register_camera(
        self, name: str, action: ActionInterface, kwargs: dict[str, Any]
    ):
        super().register_camera(name, action, kwargs)
        this_action_path = self.logging_path / name
        if not this_action_path.exists():
            this_action_path.mkdir()
        f = open(this_action_path / f"{name}.log", "+a")
        self.loggers[action] = get_standard_logger(name, f)
        self.ios.append(f)

    def stop(self):
        super().stop()
        for logger in self.loggers.values():
            logger.info("Finished!")
        for io in self.ios:
            io.close()

    def raport(self, action_instance: ActionInterface, data: _Message) -> Any:
        logger = self.loggers[action_instance]

        detected_plates = len(data.detected)
        detected_text = sum(len(x[1]) for x in data.detected)
        photo_file_path = (
            self.logging_path
            / self.actions[action_instance]
            / f"{data.time.isoformat()}.jpg"
        )

        cv2.imwrite(str(photo_file_path), data.visualised)

        if self.log_original:
            og_photo_file_path = (
                self.logging_path
                / self.actions[action_instance]
                / f"{data.time.isoformat()}-original.jpg"
            )
            cv2.imwrite(str(og_photo_file_path), data.original_image)

        logger.info(
            f"Detected plates: {detected_plates}, detected text: {detected_text}, FPS: {round(data.framerate, 1)}\n"
            f"Photo saved in: {photo_file_path}\n"
        )
