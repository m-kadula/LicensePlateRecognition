from typing import Self, Optional, Any
from abc import ABC, abstractmethod
import threading

from ..detection import PlateDetectionModel
from ..camera.base import CameraInterface


class ActionInterface(ABC):
    def __init__(self, detection_model: PlateDetectionModel, camera: CameraInterface, max_fps: int):
        self.manager: Optional["BaseActionManager"] = None
        self.detection_model = detection_model
        self.camera = camera
        self.max_fps = max_fps
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.loop)
        self._stop_now = False

    def register_manager(self, manager: "BaseActionManager"):
        if self.manager is not None:
            raise RuntimeError("Manager has already been registered for this instance.")
        self.manager = manager

    def report_to_manager(self, data: Any) -> Any:
        if self.manager is None:
            raise RuntimeError("Manager for this class is not set.")
        return self.manager.raport(self, data)

    @classmethod
    def get_instance(cls, detection_model: PlateDetectionModel, camera: CameraInterface, max_fps: int, kwargs: dict[str, Any]) -> Self:
        return cls(detection_model, camera, max_fps)

    @abstractmethod
    def loop(self):
        pass

    def start_thread(self):
        if self.manager is None:
            raise RuntimeError("Tried to start an instance without a registered manager")
        if self.thread.is_alive():
            raise RuntimeError("The thread is already running.")
        with self.lock:
            self._stop_now = False
        self.thread.start()

    def stop_thread(self):
        if not self.thread.is_alive():
            raise RuntimeError("The thread is not running.")
        with self.lock:
            self._stop_now = True
        self.thread.join()


class BaseActionManager:
    def __init__(self):
        self.actions: dict[ActionInterface, str] = {}
        self.registration_open = True
        self._is_running = False

    def register_camera(self, name: str, action: ActionInterface, kwargs: dict[str, Any]):
        if not self.registration_open:
            raise RuntimeError("Registration for this manager has been closed.")
        if action in self.actions.keys():
            raise RuntimeError(f"Camera {action} has already been registered.")
        action.register_manager(self)
        self.actions[action] = name

    def finish_registration(self):
        self.registration_open = False

    def start(self):
        if self.registration_open:
            raise RuntimeError("Tried to start without finishing the registration.")
        if self._is_running:
            raise RuntimeError("Loops for this manager are already running.")
        for action in self.actions:
            action.start_thread()
        self._is_running = True

    def stop(self):
        if not self._is_running:
            raise RuntimeError("Tried to stop but loops are not running.")
        for action in self.actions:
            action.stop_thread()

    @classmethod
    def get_instance(cls, kwargs: dict[str, Any]) -> Self:
        return cls()

    def raport(self, action_instance: ActionInterface, data: Any) -> Any:
        return None
