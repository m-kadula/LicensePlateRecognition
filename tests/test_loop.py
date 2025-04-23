import sys
from pathlib import Path
from argparse import ArgumentParser
from time import sleep
from string import ascii_uppercase, digits
from datetime import datetime

from numpy.typing import NDArray
import cv2

from licenseplate.action import LocalSaveManager, LocalSaveManagerArguments
from licenseplate.base import CameraInterface
from licenseplate.detection import YoloPlateDetectionModel
from licenseplate.preprocessor import preprocess_identity, preprocess_black_on_white


class MockCameraInterface(CameraInterface):
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        assert self.image_dir.exists() and self.image_dir.is_dir()
        self.image_iterator = None

    def get_frame(self) -> NDArray:
        if self.image_iterator is None:
            self.image_iterator = self.image_dir.iterdir()
        try:
            image_dir = next(self.image_iterator)
        except StopIteration:
            self.image_iterator = self.image_dir.iterdir()
            image_dir = next(self.image_iterator)
        return cv2.imread(image_dir)


def main():
    parser = ArgumentParser("Run three mock cameras using one manager.")
    parser.add_argument(
        "--images", type=Path, default=Path(__file__).parents[1] / "dataset/images/val"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "test_results"
        / ("loop_test-" + datetime.now().isoformat()),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
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

    model1 = YoloPlateDetectionModel(
        engine_dir,
        preprocess_identity,
        preprocess_black_on_white,
        text_allow_list=ascii_uppercase + digits,
        required_confidence=0.0,
    )
    model2 = YoloPlateDetectionModel(
        engine_dir,
        preprocess_identity,
        preprocess_black_on_white,
        text_allow_list=ascii_uppercase + digits,
        required_confidence=0.5,
    )
    model3 = YoloPlateDetectionModel(
        engine_dir,
        preprocess_identity,
        preprocess_black_on_white,
        text_allow_list=ascii_uppercase + digits,
        required_confidence=0.8,
    )
    arguments = [
        LocalSaveManagerArguments(
            name="camera1",
            detection_model=model1,
            camera=MockCameraInterface(image_dir),
            max_fps=30,
            show_debug_boxes=True,
            log_cropped_plates=True,
            log_augmented_plates=True,
        ),
        LocalSaveManagerArguments(
            name="camera2",
            detection_model=model2,
            camera=MockCameraInterface(image_dir),
            max_fps=30,
            show_debug_boxes=True,
            log_cropped_plates=False,
            log_augmented_plates=True,
        ),
        LocalSaveManagerArguments(
            name="camera3",
            detection_model=model3,
            camera=MockCameraInterface(image_dir),
            max_fps=30,
            show_debug_boxes=True,
            log_cropped_plates=False,
            log_augmented_plates=False,
        ),
    ]
    manager = LocalSaveManager(arguments, results_dir)
    manager.start()
    sleep(10)
    manager.stop()

    exit(0)


if __name__ == "__main__":
    main()
