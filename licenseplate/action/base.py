from typing import Self, Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime

from numpy.typing import NDArray

from ..detection import FinderResult, ExtractorResult


class ActionInterface(ABC):

    def __init__(self):
        self.manager: Optional["ActionManagerInterface"] = None

    def register_manager(self, manager: "ActionManagerInterface"):
        if self.manager is not None:
            raise RuntimeError("Manager has already been registered for this instance.")
        self.manager = manager

    def report_to_manager(self, *args, **kwargs) -> Any:
        if self.manager is None:
            raise RuntimeError("Manager for this class is not set.")
        return self.manager.raport(self, *args, **kwargs)

    @classmethod
    def get_instance(cls, **kwargs) -> Self:
        return cls()

    @abstractmethod
    def action_if_found(
        self,
        image: NDArray,
        detected_plated: list[tuple[FinderResult, list[ExtractorResult]]],
        time: datetime,
    ):
        pass

    def action_if_not_found(self, image: NDArray, time: datetime):
        pass


class ActionManagerInterface(ABC):

    def __init__(self):
        self.cameras: set[ActionInterface] = set()

    def register_camera(self, camera: ActionInterface, **kwargs):
        if camera in self.cameras:
            raise RuntimeError(f"Camera {camera} has already been registered")
        camera.register_manager(self)
        self.cameras.add(camera)

    @classmethod
    def get_instance(cls, **kwargs) -> Self:
        return cls()

    @abstractmethod
    def raport(self, action_instance: ActionInterface, *args, **kwargs) -> Any:
        pass
