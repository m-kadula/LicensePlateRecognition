from typing import Self, Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime

from numpy.typing import NDArray
from pydantic import BaseModel

from ..detection import FinderResult, ExtractorResult


class ActionInterface(ABC):
    def __init__(self):
        self.manager: Optional["ActionManagerInterface"] = None

    def register_manager(self, manager: "ActionManagerInterface"):
        if self.manager is not None:
            raise RuntimeError("Manager has already been registered for this instance.")
        self.manager = manager

    def report_to_manager(self, data: BaseModel) -> Any:
        if self.manager is None:
            raise RuntimeError("Manager for this class is not set.")
        return self.manager.raport(self, data)

    @classmethod
    def get_instance(cls, **kwargs) -> Self:
        return cls()

    @abstractmethod
    def action_if_found(
        self,
        image: NDArray,
        detected_plates: list[tuple[FinderResult, list[ExtractorResult]]],
        time: datetime,
    ):
        pass

    def action_if_not_found(self, image: NDArray, time: datetime):
        pass


class ActionManagerInterface(ABC):
    def __init__(self):
        self.actions: dict[ActionInterface, str] = {}
        self.registration_open = True

    def register_camera(self, name: str, action: ActionInterface, **kwargs):
        if not self.registration_open:
            raise RuntimeError("Registration for this manager has been closed")
        if action in self.actions:
            raise RuntimeError(f"Camera {action} has already been registered")
        action.register_manager(self)
        self.actions[action] = name

    def finish_registration(self):
        self.registration_open = False

    def destroy(self):
        pass

    @classmethod
    def get_instance(cls, **kwargs) -> Self:
        return cls()

    @abstractmethod
    def raport(self, action_instance: ActionInterface, data: BaseModel) -> Any:
        pass
