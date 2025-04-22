from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from time import sleep
from typing import TextIO, Any
from logging import Logger
import json

import cv2

from .base import ActionInterface, CameraInterface
from .logger import get_standard_logger
from .detection import PlateDetectionModel, DetectionResults


class LocalSave(ActionInterface):
    def __init__(
        self,
        detection_model: PlateDetectionModel,
        camera: CameraInterface,
        max_fps: int,
        logging_root: Path,
        show_debug_boxes: bool = False,
        log_cropped_plates: bool = False,
        log_augmented_plates: bool = False
    ):
        super().__init__(detection_model, camera, max_fps)
        self.logging_root = logging_root
        self.original_image_root = self.logging_root / 'original'
        self.marked_image_root = self.logging_root / 'marked'
        self.cropped_plates_root = self.logging_root / 'plates'
        self.augmented_plates_root = self.logging_root / 'augmented'

        self.debug_boxes = show_debug_boxes
        self.log_cropped_plates = log_cropped_plates
        self.log_augmented_plates = log_augmented_plates

        self.logger: Logger | None = None
        self.logger_io: TextIO | None = None

        self.logging_root.mkdir(exist_ok=True)
        self.original_image_root.mkdir(exist_ok=True)
        self.marked_image_root.mkdir(exist_ok=True)
        if self.log_cropped_plates:
            self.cropped_plates_root.mkdir(exist_ok=True)
        if self.log_augmented_plates:
            self.augmented_plates_root.mkdir(exist_ok=True)

    def log_detection(self, time: datetime, plates: DetectionResults, fps_now: float):
        log_content: dict[str, Any] = {"time": time.isoformat(), "results": [], "FPS": round(fps_now, 2)}

        original_image_path = self.original_image_root / f"{time.isoformat()}.jpg"
        marked_image_path = self.marked_image_root / f"{time.isoformat()}.jpg"
        cropped_plate_path = self.cropped_plates_root / time.isoformat()
        augmented_plate_path = self.augmented_plates_root / time.isoformat()

        cv2.imwrite(str(original_image_path), plates.original_image)
        cv2.imwrite(str(marked_image_path), plates.visualise(self.debug_boxes))

        if self.log_cropped_plates:
            cropped_plate_path.mkdir()
            log_content["cropped_plates_directory"] = str(cropped_plate_path)
        if self.log_augmented_plates:
            augmented_plate_path.mkdir()
            log_content["augmented_plates_directory"] = str(augmented_plate_path)

        log_content["original_image"] = str(original_image_path.relative_to(self.logging_root))
        log_content["marked_image"] = str(marked_image_path.relative_to(self.logging_root))

        detection_summary = []
        for i, detection_result in enumerate(plates.det_results):
            extraction_summary = []
            for extraction_result in detection_result.ext_results:
                extraction_info = {
                    'text': extraction_result.text,
                    'confidence': extraction_result.confidence,
                    'box': str(extraction_result.box)
                }
                extraction_summary.append(extraction_info)
            detection_info: dict[str, Any] = {
                'confidence': detection_result.finder_result.confidence,
                'box': str(detection_result.finder_result.box),
                'detected': extraction_summary
            }
            if self.log_cropped_plates:

                cv2.imwrite(str(cropped_plate_path / f"{i}.jpg"), detection_result.cropped_plate_image)
                detection_info["plate_image"] = str(cropped_plate_path / f"{i}.jpg")
            if self.log_augmented_plates:
                cv2.imwrite(str(augmented_plate_path / f"{i}.jpg"), detection_result.text_preprocessed_image)
                detection_info["augmented_plate_image"] = str(augmented_plate_path / f"{i}.jpg")
            detection_summary.append(detection_info)

        log_content["detected"] = detection_summary

        assert isinstance(self.logger, Logger)
        self.logger.info(json.dumps(log_content, indent=4))

    def loop(self):
        lasted = 1 / self.max_fps

        while not self.stop_signal_initiated():

            frame_time = datetime.now()

            frame = self.camera.get_frame()
            plates = self.detection_model.detect_plates(frame)
            if plates.det_results:
                self.log_detection(frame_time, plates, 1 / lasted)
            lasted = (datetime.now() - frame_time).total_seconds()

            if 1 / self.max_fps - lasted > 0:
                sleep(1 / self.max_fps - lasted)


    def start_thread(self):
        self.logger_io = open(self.logging_root / 'detected-plates.log', '+a')
        self.logger = get_standard_logger(self.logging_root.parts[1], self.logger_io)
        super().start_thread()

    def stop_thread(self):
        assert isinstance(self.logger_io, TextIO)
        self.logger_io.close()
        super().stop_thread()


@dataclass
class LocalSaveManagerArguments:
    name: str
    detection_model: PlateDetectionModel
    camera: CameraInterface
    max_fps: int
    show_debug_boxes: bool = False
    log_cropped_plates: bool = False
    log_augmented_plates: bool = False


class LocalSaveManager:

    def __init__(self, cameras: list[LocalSaveManagerArguments], logging_root: Path):
        self.cameras: dict[str, LocalSave] = {}
        self.logging_root = logging_root.resolve()
        self._is_running = False
        for args in cameras:
            camera = LocalSave(
                detection_model=args.detection_model,
                camera=args.camera,
                max_fps=args.max_fps,
                logging_root=self.logging_root / args.name,
                show_debug_boxes=args.show_debug_boxes,
                log_cropped_plates=args.log_cropped_plates,
                log_augmented_plates=args.log_augmented_plates
            )
            self.cameras[args.name] = camera

    def is_running(self) -> bool:
        return self._is_running

    def start(self):
        if self._is_running:
            raise RuntimeError("The manager has already been stared.")
        for camera in self.cameras.values():
            camera.start_thread()
        self._is_running = True

    def stop(self):
        if not self._is_running:
            raise RuntimeError("Attempted to stop a manager that has not been started")
        for camera in self.cameras.values():
            camera.stop_thread()
        self._is_running = False
