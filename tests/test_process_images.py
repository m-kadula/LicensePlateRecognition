import sys
from pathlib import Path
from argparse import ArgumentParser
from time import sleep
from string import ascii_uppercase, digits
from datetime import datetime
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import cv2

from licenseplate.action import LocalSaveManager, LocalSaveManagerArguments
from licenseplate.base import CameraInterface, preprocessor_type
from licenseplate.detection import YoloPlateDetectionModel
from licenseplate import preprocessor as prc

preprocessor_choices: dict[str, preprocessor_type] = {
    "identity": prc.preprocess_identity,
    "black-white": prc.preprocess_black_on_white,
    "threshold": prc.simple_threshold,
}


class MockCameraInterface(CameraInterface):
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        assert self.image_dir.exists() and self.image_dir.is_dir()
        self.image_iterator = self.image_dir.iterdir()
        self.call: Callable[[], None] = lambda: None

    def get_frame(self) -> NDArray:
        try:
            image_dir = next(self.image_iterator)
        except StopIteration:
            self.call()
            sleep(2)
            return np.zeros((10, 10, 3))
        return cv2.imread(str(image_dir))


def main():
    parser = ArgumentParser("Detect plates in a directory")
    parser.add_argument(
        "--images", type=Path, default=Path(__file__).parents[1] / "dataset/images/val"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "test_results"
        / ("images_test-" + datetime.now().isoformat()),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="threshold",
    )
    args = parser.parse_args()

    engine_dir: Path = args.weights.resolve()
    image_dir: Path = args.images.resolve()
    results_dir: Path = args.output.resolve()

    if not image_dir.exists() or not image_dir.is_dir():
        print(
            "Error: Provided Path for images does not exist or is not a directory.",
            file=sys.stderr,
        )
        exit(1)

    if results_dir.exists():
        print("Error: Output directory already exists.", file=sys.stderr)
        exit(1)

    if not engine_dir.exists():
        print("Error: Cannot find weights.", file=sys.stderr)
        exit(1)

    preprocessor = preprocessor_choices[args.preprocessor]

    model = YoloPlateDetectionModel(
        engine_dir,
        prc.preprocess_identity,
        preprocessor,
        text_allow_list=ascii_uppercase + digits,
        required_confidence=args.confidence,
    )

    arguments = [
        LocalSaveManagerArguments(
            name="camera",
            detection_model=model,
            camera=MockCameraInterface(image_dir),
            max_fps=30,
            show_debug_boxes=True,
            log_cropped_plates=True,
            log_augmented_plates=True,
        )
    ]

    stop_now = False

    def callback():
        nonlocal stop_now
        stop_now = True

    manager = LocalSaveManager(arguments, results_dir)
    manager.cameras["camera"].camera.call = callback

    manager.start()
    while not stop_now:
        sleep(1)
    manager.stop()


if __name__ == "__main__":
    main()
