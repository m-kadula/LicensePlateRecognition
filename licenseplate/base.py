import threading
from abc import ABC, abstractmethod
from typing import Callable

from numpy.typing import NDArray

from .detection import PlateDetectionModel


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
