import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from numpy.typing import NDArray


@dataclass
class FinderResult:
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class ExtractorResult:
    text: str
    confidence: float
    box: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass
class SingleDetectionResult:
    cropped_plate_image: NDArray
    text_preprocessed_image: NDArray
    finder_result: FinderResult
    ext_results: list[ExtractorResult]


@dataclass
class DetectionResults:
    original_image: NDArray
    general_preprocessed_image: NDArray
    det_results: list[SingleDetectionResult]


class PlateDetectionModel(ABC):
    @abstractmethod
    def detect_plates(self, image: NDArray) -> DetectionResults:
        pass

    def __call__(self, image: NDArray) -> DetectionResults:
        return self.detect_plates(image)


class CameraInterface(ABC):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @abstractmethod
    def get_frame(self) -> NDArray:
        pass


preprocessor_type = Callable[[NDArray], NDArray]


class ActionInterface(ABC):
    def __init__(
        self,
        detection_model: PlateDetectionModel,
        camera: CameraInterface,
        max_fps: int,
    ):
        self.detection_model = detection_model
        self.camera = camera
        self.max_fps = max_fps
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self.loop)
        self._stop_now = False

    @abstractmethod
    def loop(self) -> None:
        pass

    def stop_signal_initiated(self) -> bool:
        with self._lock:
            return self._stop_now

    def start_thread(self):
        if self._thread.is_alive():
            raise RuntimeError("The thread is already running.")
        with self._lock:
            self._stop_now = False
        self.camera.start()
        self._thread.start()

    def stop_thread(self):
        if not self._thread.is_alive():
            raise RuntimeError("The thread is not running.")
        with self._lock:
            self._stop_now = True
        self._thread.join()
        self.camera.stop()


class ManagerInterface:
    def __init__(self):
        self.cameras: dict[str, ActionInterface] = {}
        self._is_running = False

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
