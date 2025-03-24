from datetime import datetime
from typing import Self, Any, TextIO
from pathlib import Path
from logging import Logger

from numpy.typing import NDArray
import cv2

from .base import ActionInterface, ActionManagerInterface
from ..detection import FinderResult, ExtractorResult, visualise
from ..logger import get_standard_logger


class LocalSave(ActionInterface):

    def __init__(self, show_debug_boxes: bool = False):
        super().__init__()
        self.debug_boxes = show_debug_boxes

    @classmethod
    def get_instance(
        cls, *, show_debug_boxes: bool = False
    ) -> Self:
        return cls(show_debug_boxes)

    def action_if_found(
        self,
        image: NDArray,
        detected_plates: list[tuple[FinderResult, list[ExtractorResult]]],
        time: datetime,
    ):
        visualised = visualise(
            image, detected_plates, show_debug_boxes=self.debug_boxes
        )
        self.report_to_manager(detected=detected_plates, visualised=visualised, time=time)


class LocalSaveManager(ActionManagerInterface):

    def __init__(self, logging_path: Path):
        super().__init__()
        self.logging_path = logging_path
        self.loggers: dict[ActionInterface, Logger] = {}
        self.ios: list[TextIO] = []

        if not self.logging_path.exists():
            self.logging_path.mkdir()

    @classmethod
    def get_instance(cls, *, logging_path: str) -> Self:
        return cls(Path(logging_path).resolve())

    def register_camera(self, name: str, action: ActionInterface, **kwargs):
        super().register_camera(name, action, **kwargs)
        this_action_path = self.logging_path / name
        if not this_action_path.exists():
            this_action_path.mkdir()
        f = open(this_action_path / f'{name}.log', 'w')
        self.loggers[action] = get_standard_logger(name, f)
        self.ios.append(f)

    def destroy(self):
        for logger in self.loggers.values():
            logger.info("Finished!")
        for io in self.ios:
            io.close()

    def raport(self, action_instance: ActionInterface, **kwargs) -> Any:
        detected: list[tuple[FinderResult, list[ExtractorResult]]] = kwargs['detected']
        visualised: NDArray = kwargs['visualised']
        time: datetime = kwargs['time']
        logger = self.loggers[action_instance]

        detected_plates = len(detected)
        detected_text = sum(len(x[1]) for x in detected)
        photo_file_path = self.logging_path / self.actions[action_instance] / f"{time.isoformat()}.jpg"

        cv2.imwrite(str(photo_file_path), visualised)

        logger.info(
            f"Detected plates: {detected_plates}, detected text: {detected_text}.\n"
            f"Photo saved in: {photo_file_path}\n"
        )
