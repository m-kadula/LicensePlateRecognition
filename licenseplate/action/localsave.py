from pathlib import Path
from datetime import datetime

from numpy.typing import NDArray
import cv2

from .base import ActionInterface
from ..detection import FinderResult, ExtractorResult, visualise


class LocalSaveInterface(ActionInterface):
    def __init__(self, save_directory: Path | str, show_debug_boxes: bool = False):
        self.save_to = save_directory if isinstance(save_directory, Path) else Path(save_directory).resolve()
        self.debug_boxes = show_debug_boxes
        if not self.save_to.exists():
            self.save_to.mkdir()

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
