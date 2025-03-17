from pathlib import Path
from typing import NoReturn
from datetime import datetime
from time import sleep

import cv2

from .detection import YoloLicensePlateFinder, TextExtractor, detect_plates, visualise, polish_plate_regex
from .camera.base import CameraInterface

class DetectionLoop:

    def __init__(self,
                 finder: YoloLicensePlateFinder,
                 extractor: TextExtractor,
                 camera_interface: CameraInterface,
                 verification_regex=polish_plate_regex,
                 required_confidence: float = 0.7,
                 img_output_directory: Path = Path(__file__).parents[1] / 'detected'
                 ):
        self.finder = finder
        self.extractor = extractor
        self.camera = camera_interface
        self.verification_regex = verification_regex
        self.confidence = required_confidence
        self.img_out = img_output_directory

        if not img_output_directory.exists():
            img_output_directory.mkdir()

    def loop(self) -> NoReturn:
        while True:
            start = datetime.now()
            frame = self.camera.get_frame()
            plates = detect_plates(frame, self.finder, self.extractor, self.verification_regex, self.confidence)
            if plates:
                visualised = visualise(frame, plates)
                now = datetime.now()
                cv2.imwrite(str(self.img_out / f"{now.isoformat()}.jpg"), visualised)
            lasted = (datetime.now() - start).total_seconds()
            print(f"FPS: {1 / lasted}")
            if 1/30 - lasted > 0:
                sleep(1/30 - lasted)

    def __call__(self) -> NoReturn:
        self.loop()


if __name__ == '__main__':
    from .camera.raspberry import RaspberryCameraInterface
    camera = RaspberryCameraInterface()
    camera.initiate()
    loop = DetectionLoop(
        YoloLicensePlateFinder(Path(__file__).parents[1] / 'runs/detect/train/weights/best.pt'),
        TextExtractor(),
        camera,
    )
    loop()
