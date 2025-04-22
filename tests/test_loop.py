from pathlib import Path
import shutil
from time import sleep
from string import ascii_uppercase, digits

from numpy.typing import NDArray
import cv2

from licenseplate.action import LocalSaveManager, LocalSaveManagerArguments
from licenseplate.base import CameraInterface
from licenseplate.detection import YoloPlateDetectionModel
from licenseplate.preprocessor import preprocess_identity, preprocess_black_on_white

engine_dir = Path(__file__).parents[1] / "runs/detect/train/weights/best.pt"
results_path = Path(__file__).parents[0] / "results"
image_dir = Path(__file__).parents[1] / "dataset/images"


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


def test_loop():
    model1 = YoloPlateDetectionModel(
        Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
        preprocess_identity,
        preprocess_black_on_white,
        text_allow_list=ascii_uppercase + digits,
        required_confidence=0.0,
    )
    model2 = YoloPlateDetectionModel(
        Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
        preprocess_identity,
        preprocess_black_on_white,
        text_allow_list=ascii_uppercase + digits,
        required_confidence=0.5,
    )
    model3 = YoloPlateDetectionModel(
        Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
        preprocess_identity,
        preprocess_black_on_white,
        text_allow_list=ascii_uppercase + digits,
        required_confidence=0.8,
    )
    arguments = [
        LocalSaveManagerArguments(
            name='camera1',
            detection_model=model1,
            camera=MockCameraInterface(image_dir / "val"),
            max_fps=30,
            show_debug_boxes=True,
            log_cropped_plates=True,
            log_augmented_plates=True,
        ),
        LocalSaveManagerArguments(
            name='camera2',
            detection_model=model2,
            camera=MockCameraInterface(image_dir / "train"),
            max_fps=30,
            show_debug_boxes=True,
            log_cropped_plates=False,
            log_augmented_plates=True,
        ),
        LocalSaveManagerArguments(
            name='camera3',
            detection_model=model3,
            camera=MockCameraInterface(image_dir / "val"),
            max_fps=30,
            show_debug_boxes=True,
            log_cropped_plates=False,
            log_augmented_plates=False,
        ),
    ]
    manager = LocalSaveManager(arguments, results_path)
    manager.start()
    sleep(10)
    manager.stop()


if __name__ == "__main__":
    if results_path.exists():
        shutil.rmtree(results_path)
    test_loop()
