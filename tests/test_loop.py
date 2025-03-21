from pathlib import Path
import shutil
from datetime import datetime

from numpy.typing import NDArray
import cv2

from licenseplate.action.localsave import LocalSaveInterface
from licenseplate.camera.base import CameraInterface
from licenseplate.detection import TextExtractor, ExtractorResult, PlateDetectionModel
from licenseplate.preprocessing import preprocess_polish_license_plate, preprocess_identity
from licenseplate.main import detection_loop

engine_dir = Path(__file__).parents[1] / "runs/detect/train/weights/best.pt"
results_path = Path(__file__).parents[0] / "results"
image_dir = Path(__file__).parents[1] / "dataset/images/val"


class MockCameraInterface(CameraInterface):
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir
        assert self.image_dir.exists() and self.image_dir.is_dir()
        self.image_iterator = self.image_dir.iterdir()

    def get_frame(self) -> NDArray:
        image_dir = next(self.image_iterator)
        return cv2.imread(image_dir)


class DebugTextFinder(TextExtractor):
    def run(self, image: NDArray) -> list[ExtractorResult]:
        out = super().run(image)
        if not (results_path.parent / "processed").exists():
            (results_path.parent / "processed").mkdir()
        cv2.imwrite(
            str(
                results_path.parent / "processed" / f"{datetime.now().isoformat()}.jpg"
            ),
            image,
        )
        return out


def test_loop():
    model = PlateDetectionModel(
        Path(__file__).parents[1] / "runs/detect/train/weights/best.pt",
        preprocess_identity,
        preprocess_polish_license_plate,
        required_confidence=0.0,
    )
    try:
        detection_loop(
            model,
            MockCameraInterface(image_dir),
            LocalSaveInterface(results_path, show_debug_boxes=True),
        )
    except StopIteration:
        pass


if __name__ == "__main__":
    if results_path.exists():
        shutil.rmtree(results_path)
    test_loop()
