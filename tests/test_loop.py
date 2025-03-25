from pathlib import Path
import shutil
from time import sleep

from numpy.typing import NDArray
import cv2

from licenseplate.action.localsave import LocalSave, LocalSaveManager
from licenseplate.camera.base import CameraInterface
from licenseplate.detection import PlateDetectionModel
from licenseplate.preprocessor.base import IdentityPreprocessor
from licenseplate.preprocessor.polish_plate import PolishLicensePlatePreprocessor

engine_dir = Path(__file__).parents[1] / "runs/detect/train/weights/best.pt"
results_path = Path(__file__).parents[0] / "results"
image_dir = Path(__file__).parents[1] / "dataset/images"


class MockCameraInterface(CameraInterface):
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        assert self.image_dir.exists() and self.image_dir.is_dir()
        self.image_iterator = self.image_dir.iterdir()
        self.first = next(self.image_iterator)

    def get_frame(self) -> NDArray:
        try:
            image_dir = next(self.image_iterator)
        except StopIteration:
            image_dir = self.first
        return cv2.imread(image_dir)


def test_loop():
    model1 = PlateDetectionModel(
        Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
        IdentityPreprocessor(),
        PolishLicensePlatePreprocessor(),
        required_confidence=0.0,
    )
    model2 = PlateDetectionModel(
        Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
        IdentityPreprocessor(),
        PolishLicensePlatePreprocessor(),
        required_confidence=0.5,
    )
    model3 = PlateDetectionModel(
        Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
        IdentityPreprocessor(),
        PolishLicensePlatePreprocessor(),
        required_confidence=0.8,
    )
    action1 = LocalSave(
        model1,
        MockCameraInterface(image_dir / 'val'),
        30,
        True
    )
    action2 = LocalSave(
        model2,
        MockCameraInterface(image_dir / 'train'),
        30,
        True
    )
    action3 = LocalSave(
        model3,
        MockCameraInterface(image_dir / 'val'),
        30,
        False
    )
    manager = LocalSaveManager(results_path)
    manager.register_camera('camera1', action1, {})
    manager.register_camera('camera2', action2, {})
    manager.register_camera('camera3', action3, {})
    manager.finish_registration()
    manager.start()
    sleep(5)
    manager.stop()


if __name__ == "__main__":
    if results_path.exists():
        shutil.rmtree(results_path)
    test_loop()
