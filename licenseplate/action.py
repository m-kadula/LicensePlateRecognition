from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from time import sleep
from typing import TextIO, Any
from logging import Logger
import json

import cv2

from .base import ActionInterface, CameraInterface, ManagerInterface, DetectionResults
from .logger import get_standard_logger
from .detection import YoloPlateDetectionModel, visualise_all


class LocalSave(ActionInterface):
    def __init__(
        self,
        detection_model: YoloPlateDetectionModel,
        camera: CameraInterface,
        max_fps: int,
        logging_root: Path,
        show_debug_boxes: bool = False,
        log_cropped_plates: bool = False,
        log_augmented_plates: bool = False,
    ):
        super().__init__(detection_model, camera, max_fps)
        self.logging_root = logging_root
        self.original_image_root = self.logging_root / "original"
        self.marked_image_root = self.logging_root / "marked"
        self.cropped_plates_root = self.logging_root / "plates"
        self.augmented_plates_root = self.logging_root / "augmented"

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
        assert self.logger is not None
        log_content: dict[str, Any] = {
            "time": time.isoformat(),
            "FPS": round(fps_now, 2),
            "logger_name": self.logger.name,
        }

        original_image_path = self.original_image_root / f"{time.isoformat()}.jpg"
        marked_image_path = self.marked_image_root / f"{time.isoformat()}.jpg"
        cropped_plate_path = self.cropped_plates_root / time.isoformat()
        augmented_plate_path = self.augmented_plates_root / time.isoformat()

        cv2.imwrite(str(original_image_path), plates.original_image)
        cv2.imwrite(str(marked_image_path), visualise_all(plates, self.debug_boxes))

        if self.log_cropped_plates:
            cropped_plate_path.mkdir()
            log_content["cropped_plates_directory"] = str(
                cropped_plate_path.relative_to(self.logging_root)
            )
        if self.log_augmented_plates:
            augmented_plate_path.mkdir()
            log_content["augmented_plates_directory"] = str(
                augmented_plate_path.relative_to(self.logging_root)
            )

        log_content["original_image"] = str(
            original_image_path.relative_to(self.logging_root)
        )
        log_content["marked_image"] = str(
            marked_image_path.relative_to(self.logging_root)
        )

        detection_summary = []
        for i, detection_result in enumerate(plates.det_results):
            extraction_summary = []
            for extraction_result in detection_result.ext_results:
                extraction_info = {
                    "text": extraction_result.text,
                    "confidence": extraction_result.confidence,
                    "box": str(extraction_result.box),
                }
                extraction_summary.append(extraction_info)
            detection_info: dict[str, Any] = {
                "confidence": detection_result.finder_result.confidence,
                "box": str(detection_result.finder_result.box),
                "detected": extraction_summary,
            }
            if self.log_cropped_plates:
                cv2.imwrite(
                    str(cropped_plate_path / f"{i}.jpg"),
                    detection_result.cropped_plate_image,
                )
                detection_info["plate_image"] = str(
                    (cropped_plate_path / f"{i}.jpg").relative_to(self.logging_root)
                )
            if self.log_augmented_plates:
                cv2.imwrite(
                    str(augmented_plate_path / f"{i}.jpg"),
                    detection_result.text_preprocessed_image,
                )
                detection_info["augmented_plate_image"] = str(
                    (augmented_plate_path / f"{i}.jpg").relative_to(self.logging_root)
                )
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
        self.logger_io = open(self.logging_root / "detected-plates.log", "+a")
        self.logger = get_standard_logger(self.logging_root.parts[-1], self.logger_io)
        super().start_thread()

    def stop_thread(self):
        super().stop_thread()
        assert self.logger_io is not None
        self.logger_io.close()


@dataclass
class LocalSaveManagerArguments:
    name: str
    detection_model: YoloPlateDetectionModel
    camera: CameraInterface
    max_fps: int
    show_debug_boxes: bool = False
    log_cropped_plates: bool = False
    log_augmented_plates: bool = False


class LocalSaveManager(ManagerInterface):
    def __init__(self, cameras: list[LocalSaveManagerArguments], logging_root: Path):
        super().__init__()
        self.logging_root = logging_root.resolve()
        self.logging_root.mkdir(exist_ok=True)
        for args in cameras:
            camera = LocalSave(
                detection_model=args.detection_model,
                camera=args.camera,
                max_fps=args.max_fps,
                logging_root=self.logging_root / args.name,
                show_debug_boxes=args.show_debug_boxes,
                log_cropped_plates=args.log_cropped_plates,
                log_augmented_plates=args.log_augmented_plates,
            )
            self.cameras[args.name] = camera
