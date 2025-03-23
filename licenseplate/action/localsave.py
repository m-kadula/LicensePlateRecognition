from pathlib import Path
from datetime import datetime
from typing import Self

from numpy.typing import NDArray
import cv2

from .base import ActionInterface
from ..detection import FinderResult, ExtractorResult, visualise


class LocalSaveInterface(ActionInterface):
    def __init__(self, save_directory: Path, show_debug_boxes: bool = False):
        super().__init__()
        self.save_to = save_directory
        self.debug_boxes = show_debug_boxes
        if not self.save_to.exists():
            self.save_to.mkdir()

    @classmethod
    def get_instance(
        cls, *, save_directory: str, show_debug_boxes: bool = False
    ) -> Self:
        return cls(Path(save_directory).resolve(), show_debug_boxes)

    def action_if_found(
        self,
        image: NDArray,
        detected_plates: list[tuple[FinderResult, list[ExtractorResult]]],
        time: datetime,
    ):
        visualised = visualise(
            image, detected_plates, show_debug_boxes=self.debug_boxes
        )
        cv2.imwrite(str(self.save_to / f"{time.isoformat()}.jpg"), visualised)
